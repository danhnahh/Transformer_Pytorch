import math
import time
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import *
from config import *
from lr_scheduler import MyScheduler
from models.transformer import Transformer
from infer import translate_sentence
import wandb

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Vietnamese-English translation model')
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (e.g., checkpoints/best_transformer1.pt)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs from config'
    )
    
    return parser.parse_args()

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader=None,
                 criterion=None, optimizer=None, scheduler=None,
                 device='cuda', log_step=100, val_step=200,
                 model_save_path='best_transformer.pt', gradient_clip=1.0,
                 src_tokenizer=None, trg_tokenizer=None, test_samples=None,
                 use_wandb=False, start_epoch=0):
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_step = log_step
        self.val_step = val_step
        self.model_save_path = model_save_path
        self.gradient_clip = gradient_clip
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.start_epoch = start_epoch
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.test_samples = test_samples if test_samples else []
        self.use_wandb = use_wandb

        self.model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters")
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}\n")

    def get_loss(self, batch):
        src, trg = batch
        src = src.to(self.device)
        trg = trg.to(self.device)

        output = self.model(src, trg[:, :-1])
        

        if torch.isnan(output).any():
            print("⚠️ NaN in model outputs!")
            return None, None
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = self.criterion(output_reshape, trg)
        pred = output.argmax(dim=-1).view(-1)  # Lấy token có xác suất cao nhất
        mask = (trg != self.model.trg_pad_idx)  # Bỏ qua token padding
        correct = (pred == trg) & mask  # Đúng và không phải padding
        accuracy = correct.sum().item() / (mask.sum().item() + 1e-8)
        return loss, accuracy
    
    def train(self, num_epochs=10, accumulate_steps=1):
        """Main training loop"""
        total_batches = len(self.train_dataloader)
        total_updates = (total_batches + accumulate_steps - 1) // accumulate_steps
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            self.model.train()
            running_loss = 0.0
            running_accuracy = 0.0
            
            loop = tqdm(total=total_updates, 
                       desc=f"Epoch {epoch + 1}/{self.start_epoch + num_epochs}", 
                       unit="it", ncols=120)
            
            update_count = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.global_step += 1
                
                # Get loss and accuracy
                loss, accuracy = self.get_loss(batch)
                
                # Skip batch if NaN detected
                if loss is None:
                    print(f"Skipping batch {batch_idx}")
                    continue
                
                # Gradient accumulation
                loss = loss / accumulate_steps
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == total_batches:
                    # Gradient clipping
                    grad_norm = 0.0  # Khởi tạo mặc định
                    if self.gradient_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip)
                        if isinstance(grad_norm, torch.Tensor):
                            grad_norm = grad_norm.item()
                    else:
                        # Tính grad norm nếu không clip
                        for p in self.model.parameters():
                            if p.grad is not None:
                                grad_norm += (p.grad.data.norm(2).item()) ** 2
                        grad_norm = grad_norm ** 0.5
                    
                    # Check NaN/Inf gradients
                    import math
                    if math.isnan(grad_norm) or math.isinf(grad_norm):
                        print("⚠️ NaN/Inf gradient detected! Skipping update.")
                        self.optimizer.zero_grad()
                        continue
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    running_loss += loss.item() * accumulate_steps
                    running_accuracy += accuracy
                    update_count += 1
                    loop.update(1)
                    
                    # Logging
                    if update_count % self.log_step == 0:
                        avg_loss = running_loss / update_count
                        avg_acc = running_accuracy / update_count
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        tqdm.write(
                            f"Step {self.global_step} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Acc: {avg_acc:.4f} | "
                            f"Grad: {grad_norm:.4f} | "
                            f"LR: {current_lr:.6f}"
                        )
                        
                        # Log to wandb
                        if self.use_wandb:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/accuracy': avg_acc,
                                'train/perplexity': math.exp(min(avg_loss, 20)),  # Cap to prevent overflow
                                'train/grad_norm': grad_norm,
                                'train/learning_rate': current_lr,
                                'step': self.global_step
                            })
                    
                    # Validation
                    if self.val_dataloader is not None and update_count % self.val_step == 0:
                        val_loss, val_acc = self.validate()
                        tqdm.write(
                            f"Validation | "
                            f"Loss: {val_loss:.4f} | "
                            f"Acc: {val_acc:.4f} | "
                            f"PPL: {math.exp(val_loss):.3f}"
                        )
                        
                        # Log to wandb
                        if self.use_wandb:
                            wandb.log({
                                'val/loss': val_loss,
                                'val/accuracy': val_acc,
                                'val/perplexity': math.exp(min(val_loss, 20)),
                                'step': self.global_step
                            })
                        
                        self.translate_samples(num_samples=3, beam_size=3)
                        self.save_best_model(val_loss, val_acc)
            
            # End of epoch
            epoch_train_loss = running_loss / update_count
            epoch_train_acc = running_accuracy / update_count
            history['train_loss'].append(epoch_train_loss)
            history['train_accuracy'].append(epoch_train_acc)
            
            if self.val_dataloader is not None:
                epoch_val_loss, epoch_val_acc = self.validate()
                history['val_loss'].append(epoch_val_loss)
                history['val_accuracy'].append(epoch_val_acc)
                
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{self.start_epoch + num_epochs} Summary:")
                print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Train PPL: {math.exp(epoch_train_loss):.3f}")
                print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Val PPL: {math.exp(epoch_val_loss):.3f}")
                print(f"{'='*60}\n")
                
                # Log epoch metrics to wandb
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'epoch/train_loss': epoch_train_loss,
                        'epoch/train_accuracy': epoch_train_acc,
                        'epoch/train_perplexity': math.exp(min(epoch_train_loss, 20)),
                        'epoch/val_loss': epoch_val_loss,
                        'epoch/val_accuracy': epoch_val_acc,
                        'epoch/val_perplexity': math.exp(min(epoch_val_loss, 20)),
                    })
                
                self.translate_samples(num_samples=3, beam_size=3)
                self.save_best_model(epoch_val_loss, epoch_val_acc)
            else:
                print(f"\nEpoch {epoch + 1}/{self.start_epoch + num_epochs}: Train Loss = {epoch_train_loss:.4f}\n")
                
                # Log epoch metrics to wandb
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'epoch/train_loss': epoch_train_loss,
                        'epoch/train_accuracy': epoch_train_acc,
                        'epoch/train_perplexity': math.exp(min(epoch_train_loss, 20)),
                    })
            
            loop.close()
        
        return self.model, history
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                loss, accuracy = self.get_loss(batch)

                if loss is None:
                    continue
                batch_size = batch[0].size(0)
                total_loss += loss.item() * batch_size
                total_accuracy += accuracy * batch_size
                total_samples += batch_size
        self.model.train()

        if total_samples == 0:
            return float('inf'), 0.0
        
        return total_loss / total_samples, total_accuracy / total_samples
    
    def translate_samples(self, num_samples=3, beam_size=3):
        """Translate sample sentences and display them"""
        if not self.test_samples or not self.src_tokenizer or not self.trg_tokenizer:
            return
        
        print(f"\n{'='*80}")
        print("TRANSLATION SAMPLES")
        print(f"{'='*80}")
        
        samples_to_show = min(num_samples, len(self.test_samples))
        for i in range(samples_to_show):
            src_text, ref_text = self.test_samples[i]
            
            translation = translate_sentence(
                self.model, src_text, self.src_tokenizer, self.trg_tokenizer,
                self.device, max_len=MAX_SEQ_LEN, beam_size=beam_size
            )
            
            print(f"\nSample {i+1}:")
            print(f"  Source:      {src_text}")
            print(f"  Reference:   {ref_text}")
            print(f"  Translation: {translation}")
        
        print(f"{'='*80}\n")
    
    def save_best_model(self, val_loss, val_accuracy):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'epoch': self.global_step
            }, self.model_save_path)
            print(f"Best model saved with val loss: {val_loss:.4f} and val acc: {val_accuracy:.4f}")

def main():
    args = parse_args()
    
    # Override epochs if specified
    epochs = args.epochs if args.epochs is not None else EPOCHS
    
    # Determine dataset name for wandb run name
    if USE_DATASET == 'huggingface':
        dataset_name = HF_DATASET_NAME.split('/')[-1]  # Extract dataset name from path
        print(f"Using Hugging Face dataset: {HF_DATASET_NAME}")
    else:
        dataset_name = 'IWSLT15'
        print(f"Using local dataset: {data_path}")
    
    # Check if resuming from checkpoint
    resume_checkpoint = None
    start_epoch = 0
    if args.resume:
        print(f"\n{'='*60}")
        print("RESUMING FROM CHECKPOINT")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.resume}")
        try:
            resume_checkpoint = torch.load(args.resume, map_location=DEVICE)
            print(f"✓ Checkpoint loaded successfully")
            if 'epoch' in resume_checkpoint:
                # Estimate epoch from global steps
                print(f"  Checkpoint step: {resume_checkpoint['epoch']}")
            if 'val_loss' in resume_checkpoint:
                print(f"  Checkpoint validation loss: {resume_checkpoint['val_loss']:.4f}")
            if 'val_accuracy' in resume_checkpoint:
                print(f"  Checkpoint validation accuracy: {resume_checkpoint['val_accuracy']:.4f}")
        except FileNotFoundError:
            print(f"✗ Checkpoint not found: {args.resume}")
            exit(1)
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            exit(1)
    
    # Initialize wandb if enabled
    if USE_WANDB:
        run_name = f"{dataset_name}-resume" if args.resume else dataset_name
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            resume='allow' if args.resume else False,
            config={
                'dataset': USE_DATASET,
                'dataset_name': HF_DATASET_NAME if USE_DATASET == 'huggingface' else 'IWSLT15',
                'max_seq_len': MAX_SEQ_LEN,
                'num_layers': NUM_LAYERS,
                'd_model': D_MODEL,
                'd_ff': D_FF,
                'num_heads': NUM_HEADS,
                'dropout': DROPOUT,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'epochs': epochs,
                'gradient_accumulation': GRADIENT_ACCUMULATION,
                'vocab_size': VOCAB_SIZE,
                'warmup_ratio': WARMUP_RATIO,
                'weight_decay': WEIGHT_DECAY,
                'resume': args.resume is not None,
            }
        )
        print(f"✓ Weights & Biases initialized with run name: {run_name}")
    
    # Load dataset based on config setting
    if USE_DATASET == 'huggingface':
        print(f"Using Hugging Face dataset: {HF_DATASET_NAME}")
        en_tokenizer, vi_tokenizer, all_train_sequences, all_val_sequences = preprocess_data(
            vocab_size=VOCAB_SIZE,
            use_huggingface=True,
            hf_dataset_name=HF_DATASET_NAME
        )
        # Load test samples from HF dataset
        _, _, test_dataset = load_data_from_huggingface(HF_DATASET_NAME)
        test_src = [example['vi'] for example in test_dataset.select(range(5))]
        test_trg = [example['en'] for example in test_dataset.select(range(5))]
        test_samples = list(zip(test_src, test_trg))
    else:
        print(f"Using local dataset: {data_path}")
        en_tokenizer, vi_tokenizer, all_train_sequences, all_val_sequences = preprocess_data(
            train_src_path=train_data_path + "train.vi.txt",
            train_trg_path=train_data_path + "train.en.txt",
            val_src_path=data_path + "tst2013.vi.txt",
            val_trg_path=data_path + "tst2013.en.txt",
            vocab_size=VOCAB_SIZE,
            use_huggingface=False
        )
        # Load test samples from local files
        test_src, test_trg = load_data(
            data_path + "tst2012.vi.txt",
            data_path + "tst2012.en.txt"
        )
        test_samples = list(zip(test_src[:5], test_trg[:5]))

    train_batches = DataLoader(all_train_sequences, batch_size=BATCH_SIZE, shuffle=True)
    val_batches = DataLoader(all_val_sequences, batch_size=BATCH_SIZE, shuffle=False)

    # Get vocab sizes from SentencePiece tokenizers
    en_vocab_size = en_tokenizer.get_vocab_size()
    vi_vocab_size = vi_tokenizer.get_vocab_size()
    
    # Pad token ID for SentencePiece (we use 3)
    pad_token_id = vi_tokenizer.pad_id
     
    print(f"\n{'='*60}")
    print("DATA INFO")
    print(f"{'='*60}")
    print(f"Train samples: {len(all_train_sequences)}")
    print(f"Val samples: {len(all_val_sequences)}")
    print(f"EN vocab size: {en_vocab_size}")
    print(f"VI vocab size: {vi_vocab_size}")
    print(f"Pad token ID: {pad_token_id}")

    model = Transformer(
        src_pad_idx=pad_token_id,
        trg_pad_idx=pad_token_id,
        d_model=D_MODEL,
        inp_vocab_size=vi_vocab_size,
        trg_vocab_size=en_vocab_size,
        max_len=MAX_SEQ_LEN,
        d_ff=D_FF,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        device=DEVICE,
        use_alignment=True  # Transformer-Align: combines dot-product and additive attention
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_updates = ((len(train_batches) + GRADIENT_ACCUMULATION - 1) // 
                    GRADIENT_ACCUMULATION) * epochs 

    scheduler = MyScheduler(
        optimizer, 
        total_steps=total_updates,
        scheduler_type='cosine',  # hoặc 'linear', 'exponential'
        warmup_ratio=WARMUP_RATIO,         # 10% đầu là warmup
        final_lr_ratio=FINAL_LR_RATIO 
    )
    
    # Load checkpoint states if resuming
    best_val_loss = float('inf')
    if resume_checkpoint:
        print(f"\n{'='*60}")
        print("LOADING CHECKPOINT STATES")
        print(f"{'='*60}")
        try:
            model.load_state_dict(resume_checkpoint['model_state_dict'])
            print("✓ Model state loaded")
            
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded")
            
            if resume_checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
                print("✓ Scheduler state loaded")
            
            if 'val_loss' in resume_checkpoint:
                best_val_loss = resume_checkpoint['val_loss']
                print(f"✓ Best validation loss: {best_val_loss:.4f}")
            
            # Estimate start epoch from global steps
            if 'epoch' in resume_checkpoint:
                updates_per_epoch = (len(train_batches) + GRADIENT_ACCUMULATION - 1) // GRADIENT_ACCUMULATION
                start_epoch = resume_checkpoint['epoch'] // updates_per_epoch
                print(f"✓ Estimated starting epoch: {start_epoch + 1}")
            
            # Free memory
            del resume_checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ Checkpoint memory freed")
            
        except Exception as e:
            print(f"✗ Error loading checkpoint states: {e}")
            print("  Starting fresh training instead")
            start_epoch = 0
            best_val_loss = float('inf')

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    print(f"\n{'='*60}")
    print("TRAINING CONFIG")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {epochs} (starting from epoch {start_epoch + 1})")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Scheduler: cosine with warmup")
    print(f"Total training steps: {total_updates}")

    trainer = Trainer(
        model=model,
        train_dataloader=train_batches,
        val_dataloader=val_batches,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        log_step=BATCH_PRINT,
        val_step=VAL_STEP,
        model_save_path=f'{saved_model_path}best_transformer1.pt',
        gradient_clip=CLIP,
        src_tokenizer=vi_tokenizer,
        trg_tokenizer=en_tokenizer,
        test_samples=test_samples,
        use_wandb=USE_WANDB,
        start_epoch=start_epoch
    )
    
    # Set best_val_loss from checkpoint
    if best_val_loss != float('inf'):
        trainer.best_val_loss = best_val_loss

    print(f"\n{'='*60}")
    print("START TRAINING")
    print(f"{'='*60}\n")

    trained_model, history = trainer.train(
        num_epochs=epochs, 
        accumulate_steps=GRADIENT_ACCUMULATION
    )
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Finish wandb run
    if USE_WANDB:
        wandb.finish()
        print("✓ Weights & Biases run finished")

if __name__ == "__main__":
    main()