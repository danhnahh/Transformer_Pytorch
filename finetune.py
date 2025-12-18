#!/usr/bin/env python3
"""
Fine-tuning script for Vietnamese-English Neural Machine Translation.

This script allows you to:
1. Load a model trained on one dataset (e.g., HuggingFace large dataset)
2. Fine-tune it on another dataset (e.g., IWSLT'15 local dataset)

Usage:
    python finetune.py --checkpoint checkpoints/best_transformer1.pt --target-dataset local
    python finetune.py --checkpoint checkpoints/hf_model.pt --target-dataset huggingface
"""

import argparse
import math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import preprocess_data, load_data_from_huggingface, load_data
from config import *
from lr_scheduler import MyScheduler
from models.transformer import Transformer
from train import Trainer
import wandb

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune a translation model on a different dataset')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the pretrained model checkpoint (e.g., checkpoints/best_transformer1.pt)'
    )
    
    parser.add_argument(
        '--target-dataset',
        type=str,
        choices=['huggingface', 'local'],
        default=None,
        help='Target dataset to fine-tune on. If not specified, uses the opposite of current USE_DATASET'
    )
    
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Name for the fine-tuned model checkpoint (default: finetuned_<target_dataset>.pt)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of fine-tuning epochs (default: 10)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate for fine-tuning (default: 1e-5, much smaller than initial training)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default: use BATCH_SIZE from config)'
    )
    
    parser.add_argument(
        '--freeze-encoder',
        action='store_true',
        help='Freeze encoder layers during fine-tuning'
    )
    
    parser.add_argument(
        '--freeze-decoder',
        action='store_true',
        help='Freeze decoder layers during fine-tuning (not recommended)'
    )
    
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.05,
        help='Warmup ratio for learning rate scheduler (default: 0.05)'
    )
    
    return parser.parse_args()

def load_pretrained_model(checkpoint_path, device):
    """Load a pretrained model from checkpoint."""
    print(f"\n{'='*70}")
    print("Loading Pretrained Model")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✓ Checkpoint loaded successfully")
        
        # Extract information from checkpoint
        if 'val_loss' in checkpoint:
            print(f"  Original validation loss: {checkpoint['val_loss']:.4f}")
        if 'val_accuracy' in checkpoint:
            print(f"  Original validation accuracy: {checkpoint['val_accuracy']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  Trained for steps: {checkpoint['epoch']}")
        
        return checkpoint
    except FileNotFoundError:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Please provide a valid checkpoint path")
        exit(1)
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        exit(1)

def freeze_layers(model, freeze_encoder=False, freeze_decoder=False):
    """Freeze specified layers of the model."""
    if freeze_encoder:
        print("  Freezing encoder layers...")
        for param in model.encoder.parameters():
            param.requires_grad = False
        frozen_encoder = sum(p.numel() for p in model.encoder.parameters())
        print(f"    ✓ Frozen {frozen_encoder:,} encoder parameters")
    
    if freeze_decoder:
        print("  Freezing decoder layers...")
        for param in model.decoder.parameters():
            param.requires_grad = False
        frozen_decoder = sum(p.numel() for p in model.decoder.parameters())
        print(f"    ✓ Frozen {frozen_decoder:,} decoder parameters")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

def main():
    args = parse_args()
    
    # Determine target dataset
    if args.target_dataset is None:
        # Use opposite of current setting
        target_dataset = 'local' if USE_DATASET == 'huggingface' else 'huggingface'
        print(f"No target dataset specified. Using: {target_dataset} (opposite of current: {USE_DATASET})")
    else:
        target_dataset = args.target_dataset
    
    # Determine output name
    if args.output_name is None:
        output_name = f"finetuned_{target_dataset}.pt"
    else:
        output_name = args.output_name
        if not output_name.endswith('.pt'):
            output_name += '.pt'
    
    print(f"\n{'='*70}")
    print("Fine-tuning Configuration")
    print(f"{'='*70}")
    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Target dataset: {target_dataset}")
    print(f"Output name: {output_name}")
    print(f"Fine-tuning epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Freeze decoder: {args.freeze_decoder}")
    
    # Load pretrained checkpoint
    checkpoint = load_pretrained_model(args.checkpoint, DEVICE)
    
    # Load target dataset
    print(f"\n{'='*70}")
    print("Loading Target Dataset")
    print(f"{'='*70}")
    
    if target_dataset == 'huggingface':
        print(f"Using Hugging Face dataset: {HF_DATASET_NAME}")
        en_tokenizer, vi_tokenizer, all_train_sequences, all_val_sequences = preprocess_data(
            vocab_size=VOCAB_SIZE,
            use_huggingface=True,
            hf_dataset_name=HF_DATASET_NAME
        )
        # Load test samples
        _, _, test_dataset = load_data_from_huggingface(HF_DATASET_NAME)
        test_src = [example['vi'] for example in test_dataset.select(range(5))]
        test_trg = [example['en'] for example in test_dataset.select(range(5))]
        test_samples = list(zip(test_src, test_trg))
        dataset_name = HF_DATASET_NAME.split('/')[-1]
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
        # Load test samples
        test_src, test_trg = load_data(
            data_path + "tst2012.vi.txt",
            data_path + "tst2012.en.txt"
        )
        test_samples = list(zip(test_src[:5], test_trg[:5]))
        dataset_name = 'IWSLT15'
    
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    train_batches = DataLoader(all_train_sequences, batch_size=batch_size, shuffle=True)
    val_batches = DataLoader(all_val_sequences, batch_size=batch_size, shuffle=False)
    
    # Get vocab sizes
    en_vocab_size = en_tokenizer.get_vocab_size()
    vi_vocab_size = vi_tokenizer.get_vocab_size()
    pad_token_id = vi_tokenizer.pad_id
    
    print(f"✓ Target dataset loaded")
    print(f"  Train samples: {len(all_train_sequences):,}")
    print(f"  Val samples: {len(all_val_sequences):,}")
    print(f"  EN vocab size: {en_vocab_size}")
    print(f"  VI vocab size: {vi_vocab_size}")
    
    # Initialize model with same architecture
    print(f"\n{'='*70}")
    print("Initializing Model")
    print(f"{'='*70}")
    
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
        use_alignment=True
    ).to(DEVICE)
    
    # Load pretrained weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Pretrained weights loaded successfully")
        
        # Free memory: delete checkpoint to avoid OOM
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✓ Checkpoint memory freed")
    except Exception as e:
        print(f"✗ Error loading pretrained weights: {e}")
        print("  Make sure the model architecture matches the checkpoint")
        exit(1)
    
    # Freeze layers if requested
    if args.freeze_encoder or args.freeze_decoder:
        print("\nApplying layer freezing...")
        freeze_layers(model, args.freeze_encoder, args.freeze_decoder)
    
    # Initialize wandb if enabled
    use_wandb = USE_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"finetune-{dataset_name}",
            config={
                'mode': 'finetuning',
                'source_checkpoint': args.checkpoint,
                'target_dataset': target_dataset,
                'dataset_name': dataset_name,
                'max_seq_len': MAX_SEQ_LEN,
                'num_layers': NUM_LAYERS,
                'd_model': D_MODEL,
                'd_ff': D_FF,
                'num_heads': NUM_HEADS,
                'dropout': DROPOUT,
                'batch_size': batch_size,
                'learning_rate': args.learning_rate,
                'epochs': args.epochs,
                'gradient_accumulation': GRADIENT_ACCUMULATION,
                'vocab_size': VOCAB_SIZE,
                'warmup_ratio': args.warmup_ratio,
                'freeze_encoder': args.freeze_encoder,
                'freeze_decoder': args.freeze_decoder,
            }
        )
        print(f"✓ Weights & Biases initialized with run name: finetune-{dataset_name}")
    
    # Setup optimizer with lower learning rate for fine-tuning
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    
    # Setup scheduler
    total_updates = ((len(train_batches) + GRADIENT_ACCUMULATION - 1) // 
                    GRADIENT_ACCUMULATION) * args.epochs
    
    scheduler = MyScheduler(
        optimizer,
        total_steps=total_updates,
        scheduler_type='cosine',
        warmup_ratio=args.warmup_ratio,
        final_lr_ratio=FINAL_LR_RATIO
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    print(f"\n{'='*70}")
    print("Fine-tuning Configuration")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate} (fine-tuning LR)")
    print(f"Scheduler: cosine with warmup (ratio: {args.warmup_ratio})")
    print(f"Total training steps: {total_updates}")
    
    # Create trainer
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
        model_save_path=f'{saved_model_path}{output_name}',
        gradient_clip=CLIP,
        src_tokenizer=vi_tokenizer,
        trg_tokenizer=en_tokenizer,
        test_samples=test_samples,
        use_wandb=use_wandb
    )
    
    print(f"\n{'='*70}")
    print("START FINE-TUNING")
    print(f"{'='*70}\n")
    
    # Train (fine-tune)
    trained_model, history = trainer.train(
        num_epochs=args.epochs,
        accumulate_steps=GRADIENT_ACCUMULATION
    )
    
    print(f"\n{'='*70}")
    print("FINE-TUNING COMPLETED!")
    print(f"{'='*70}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to: {saved_model_path}{output_name}")
    
    # Finish wandb
    if use_wandb:
        wandb.finish()
        print("✓ Weights & Biases run finished")
    
    # Print comparison with original model
    if 'val_loss' in checkpoint:
        original_loss = checkpoint['val_loss']
        improvement = original_loss - trainer.best_val_loss
        print(f"\nComparison:")
        print(f"  Original validation loss: {original_loss:.4f}")
        print(f"  Fine-tuned validation loss: {trainer.best_val_loss:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement/original_loss*100:.1f}%)")

if __name__ == "__main__":
    main()
