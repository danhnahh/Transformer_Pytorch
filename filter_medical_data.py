# -*- coding: utf-8 -*-
"""
Script loc data dich y te chat luong cao
"""

import re
import sys
import io
from collections import defaultdict
from tqdm import tqdm
import argparse

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Tu khoa y te tieng Anh
MEDICAL_KEYWORDS_EN = {
    # Benh ly
    'disease', 'disorder', 'syndrome', 'infection', 'cancer', 'tumor', 'carcinoma',
    'diabetes', 'hypertension', 'inflammation', 'pneumonia', 'hepatitis', 'covid',
    'virus', 'bacteria', 'fever', 'pain', 'symptom', 'diagnosis', 'prognosis',
    'adenocarcinoma', 'lymphoma', 'leukemia', 'metastasis', 'lesion', 'neoplasm',

    # Giai phau
    'heart', 'lung', 'liver', 'kidney', 'brain', 'blood', 'cell', 'tissue',
    'organ', 'muscle', 'bone', 'nerve', 'artery', 'vein', 'lymph', 'colon',
    'rectum', 'stomach', 'intestine', 'pancreas', 'thyroid', 'breast', 'ovary',
    'uterus', 'cervix', 'prostate', 'bladder', 'esophagus', 'trachea',

    # Dieu tri
    'treatment', 'therapy', 'surgery', 'medication', 'drug', 'dose', 'vaccine',
    'injection', 'transplant', 'chemotherapy', 'radiation', 'antibiotic',
    'surgical', 'operative', 'postoperative', 'preoperative', 'resection',

    # Y te chung
    'patient', 'hospital', 'clinic', 'doctor', 'nurse', 'physician', 'medical',
    'clinical', 'healthcare', 'medicine', 'pharmaceutical', 'laboratory',
    'pathology', 'histopathology', 'immunohistochemistry', 'biopsy',

    # Xet nghiem
    'test', 'examination', 'screening', 'imaging', 'x-ray', 'mri', 'ct',
    'ultrasound', 'endoscopy', 'ecg', 'eeg', 'pet', 'scan', 'tomography',

    # Thong ke y te
    'mortality', 'morbidity', 'prevalence', 'incidence', 'epidemiology',
    'randomized', 'placebo', 'trial', 'study', 'research', 'analysis',
    'statistical', 'significant', 'odds ratio', 'confidence interval',
}

# Tu khoa y te tieng Viet (khong dau)
MEDICAL_KEYWORDS_VI = {
    # Benh ly
    'benh', 'hoi chung', 'nhiem', 'ung thu', 'khoi u', 'tieu duong', 'huyet ap',
    'viem', 'sot', 'dau', 'trieu chung', 'chan doan', 'tien luong', 'virus',
    'di can', 'ton thuong', 'lac noi mac', 'xo gan', 'suy', 'hoai tu',

    # Giai phau
    'tim', 'phoi', 'gan', 'than', 'nao', 'mau', 'te bao', 'mo', 'co quan',
    'co', 'xuong', 'than kinh', 'dong mach', 'tinh mach', 'hach', 'dai trang',
    'truc trang', 'da day', 'ruot', 'tuy', 'tuyen giap', 'vu', 'buong trung',
    'tu cung', 'co tu cung', 'tuyen tien liet', 'bang quang', 'thuc quan',

    # Dieu tri
    'dieu tri', 'phau thuat', 'thuoc', 'lieu', 'vac xin', 'tiem', 'ghep',
    'hoa tri', 'xa tri', 'khang sinh', 'lieu phap', 'cat bo', 'noi soi',

    # Y te chung
    'benh nhan', 'benh vien', 'phong kham', 'bac si', 'y ta', 'dieu duong',
    'y te', 'lam sang', 'y khoa', 'duoc', 'phong thi nghiem', 'giai phau benh',

    # Xet nghiem
    'xet nghiem', 'kham', 'sinh thiet', 'sang loc', 'sieu am', 'noi soi',
    'chup', 'cong huong tu', 'x quang', 'ct', 'pet', 'mri',

    # Thong ke
    'ty le', 'nghien cuu', 'phan tich', 'thu nghiem', 'mau', 'ket qua',
    'thong ke', 'co y nghia', 'doi tuong', 'phuong phap',
}

# Vietnamese accent pattern for detection
VIETNAMESE_PATTERN = (
    u'[\u00e0\u00e1\u1ea1\u1ea3\u00e3'  # a with accents
    u'\u00e2\u1ea7\u1ea5\u1ead\u1ea9\u1eab'  # a circumflex
    u'\u0103\u1eb1\u1eaf\u1eb7\u1eb3\u1eb5'  # a breve
    u'\u00e8\u00e9\u1eb9\u1ebb\u1ebd'  # e with accents
    u'\u00ea\u1ec1\u1ebf\u1ec7\u1ec3\u1ec5'  # e circumflex
    u'\u00ec\u00ed\u1ecb\u1ec9\u0129'  # i with accents
    u'\u00f2\u00f3\u1ecd\u1ecf\u00f5'  # o with accents
    u'\u00f4\u1ed3\u1ed1\u1ed9\u1ed5\u1ed7'  # o circumflex
    u'\u01a1\u1edd\u1edb\u1ee3\u1edf\u1ee1'  # o horn
    u'\u00f9\u00fa\u1ee5\u1ee7\u0169'  # u with accents
    u'\u01b0\u1eeb\u1ee9\u1ef1\u1eed\u1eef'  # u horn
    u'\u1ef3\u00fd\u1ef5\u1ef7\u1ef9'  # y with accents
    u'\u0111]'  # d with stroke
)


def remove_vietnamese_accents(text):
    """Remove Vietnamese accents for keyword matching"""
    replacements = [
        (u'\u00e0\u00e1\u1ea1\u1ea3\u00e3\u00e2\u1ea7\u1ea5\u1ead\u1ea9\u1eab\u0103\u1eb1\u1eaf\u1eb7\u1eb3\u1eb5', 'a'),
        (u'\u00e8\u00e9\u1eb9\u1ebb\u1ebd\u00ea\u1ec1\u1ebf\u1ec7\u1ec3\u1ec5', 'e'),
        (u'\u00ec\u00ed\u1ecb\u1ec9\u0129', 'i'),
        (u'\u00f2\u00f3\u1ecd\u1ecf\u00f5\u00f4\u1ed3\u1ed1\u1ed9\u1ed5\u1ed7\u01a1\u1edd\u1edb\u1ee3\u1edf\u1ee1', 'o'),
        (u'\u00f9\u00fa\u1ee5\u1ee7\u0169\u01b0\u1eeb\u1ee9\u1ef1\u1eed\u1eef', 'u'),
        (u'\u1ef3\u00fd\u1ef5\u1ef7\u1ef9', 'y'),
        (u'\u0111', 'd'),
    ]
    result = text.lower()
    for chars, base in replacements:
        for char in chars:
            result = result.replace(char, base)
    return result


def count_words(text):
    """Dem so tu trong cau"""
    return len(text.split())


def has_medical_keywords(en_text, vi_text, min_keywords=1):
    """Kiem tra cau co chua tu khoa y te khong"""
    en_lower = en_text.lower()
    vi_normalized = remove_vietnamese_accents(vi_text)

    en_count = sum(1 for kw in MEDICAL_KEYWORDS_EN if kw in en_lower)
    vi_count = sum(1 for kw in MEDICAL_KEYWORDS_VI if kw in vi_normalized)

    return (en_count + vi_count) >= min_keywords


def calculate_length_ratio(en_text, vi_text):
    """Tinh ty le do dai EN/VI"""
    en_len = len(en_text)
    vi_len = len(vi_text)
    if vi_len == 0:
        return float('inf')
    return en_len / vi_len


def has_excessive_special_chars(text, threshold=0.3):
    """Kiem tra cau co qua nhieu ky tu dac biet khong"""
    if len(text) == 0:
        return True

    special_chars = re.findall(r'[^\w\s.,;:!?\-\'\"()\[\]%/]', text, re.UNICODE)
    ratio = len(special_chars) / len(text)
    return ratio > threshold


def has_excessive_numbers(text, threshold=0.4):
    """Kiem tra cau co qua nhieu so khong"""
    if len(text) == 0:
        return True

    numbers = re.findall(r'\d', text)
    ratio = len(numbers) / len(text)
    return ratio > threshold


def is_mostly_uppercase(text, threshold=0.7):
    """Kiem tra cau co qua nhieu chu hoa khong"""
    letters = re.findall(r'[a-zA-Z]', text)
    if len(letters) == 0:
        return False

    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > threshold


def contains_vietnamese(text):
    """Kiem tra text co chua ky tu tieng Viet khong"""
    vietnamese_regex = re.compile(VIETNAMESE_PATTERN, re.IGNORECASE)
    return bool(vietnamese_regex.search(text))


def is_valid_pair(en_text, vi_text, config):
    """Kiem tra cap cau co hop le khong"""

    # 1. Kiem tra cau rong
    if not en_text.strip() or not vi_text.strip():
        return False, "empty"

    # 2. Kiem tra do dai
    en_words = count_words(en_text)
    vi_words = count_words(vi_text)

    if en_words < config['min_words'] or vi_words < config['min_words']:
        return False, "too_short"

    if en_words > config['max_words'] or vi_words > config['max_words']:
        return False, "too_long"

    # 3. Kiem tra ty le do dai
    ratio = calculate_length_ratio(en_text, vi_text)
    if ratio < config['min_ratio'] or ratio > config['max_ratio']:
        return False, "bad_ratio"

    # 4. Kiem tra ky tu dac biet
    if has_excessive_special_chars(en_text) or has_excessive_special_chars(vi_text):
        return False, "special_chars"

    # 5. Kiem tra so
    if has_excessive_numbers(en_text) or has_excessive_numbers(vi_text):
        return False, "too_many_numbers"

    # 6. Kiem tra chu hoa
    if is_mostly_uppercase(en_text):
        return False, "uppercase"

    # 7. Kiem tra tieng Viet trong cau VI
    if not contains_vietnamese(vi_text):
        return False, "no_vietnamese"

    return True, "valid"


def load_test_set(test_en_file, test_vi_file):
    """Load test set de loai bo cac cau trung"""
    test_en_set = set()
    test_vi_set = set()

    try:
        with open(test_en_file, 'r', encoding='utf-8') as f:
            test_en_set = {line.strip().lower() for line in f}
        with open(test_vi_file, 'r', encoding='utf-8') as f:
            test_vi_set = {line.strip().lower() for line in f}
        print(f"Loaded test set: {len(test_en_set):,} EN, {len(test_vi_set):,} VI sentences")
    except FileNotFoundError:
        print("Warning: Test files not found, skipping overlap check")

    return test_en_set, test_vi_set


def filter_data(en_file, vi_file, output_prefix, config):
    """Loc du lieu va luu ket qua"""

    print(f"Reading data from {en_file} and {vi_file}...")

    with open(en_file, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f]

    with open(vi_file, 'r', encoding='utf-8') as f:
        vi_lines = [line.strip() for line in f]

    assert len(en_lines) == len(vi_lines), "EN and VI line counts don't match!"

    print(f"Total original pairs: {len(en_lines):,}")

    # Load test set de loai bo overlap
    test_en_set, test_vi_set = set(), set()
    if config.get('test_en_file') and config.get('test_vi_file'):
        test_en_set, test_vi_set = load_test_set(config['test_en_file'], config['test_vi_file'])

    # Thong ke
    stats = defaultdict(int)
    valid_pairs = []
    seen_pairs = set()

    print("Filtering data...")
    for en, vi in tqdm(zip(en_lines, vi_lines), total=len(en_lines)):
        # Kiem tra trung voi test set
        en_lower = en.lower().strip()
        vi_lower = vi.lower().strip()
        if en_lower in test_en_set or vi_lower in test_vi_set:
            stats['overlap_with_test'] += 1
            continue
        # Kiem tra trung lap
        pair_key = (en.lower().strip(), vi.lower().strip())
        if pair_key in seen_pairs:
            stats['duplicate'] += 1
            continue

        # Kiem tra hop le
        is_valid, reason = is_valid_pair(en, vi, config)

        if is_valid:
            has_medical = has_medical_keywords(en, vi, config['min_medical_keywords'])

            if config['medical_only'] and not has_medical:
                stats['no_medical_keyword'] += 1
                continue

            valid_pairs.append((en, vi, has_medical))
            seen_pairs.add(pair_key)
            stats['valid'] += 1
            if has_medical:
                stats['has_medical_keyword'] += 1
        else:
            stats[reason] += 1

    # Stratified sampling theo do dai de phan bo deu
    if config['max_samples'] and len(valid_pairs) > config['max_samples']:
        import random
        random.seed(42)

        # Chia thanh 2 nhom: co va khong co tu khoa y te
        medical_pairs = [p for p in valid_pairs if p[2]]
        non_medical_pairs = [p for p in valid_pairs if not p[2]]

        def stratified_sample(pairs, n_samples):
            """Lay mau phan tang theo do dai"""
            if len(pairs) <= n_samples:
                return pairs

            # Chia thanh cac bins theo do dai (so tu)
            bins = defaultdict(list)
            for p in pairs:
                word_count = count_words(p[0])
                # Bins: 5-10, 11-20, 21-40, 41-80, 81+
                if word_count <= 10:
                    bin_id = 0
                elif word_count <= 20:
                    bin_id = 1
                elif word_count <= 40:
                    bin_id = 2
                elif word_count <= 80:
                    bin_id = 3
                else:
                    bin_id = 4
                bins[bin_id].append(p)

            # Lay deu tu moi bin
            result = []
            samples_per_bin = n_samples // len(bins)

            for bin_id in sorted(bins.keys()):
                bin_pairs = bins[bin_id]
                random.shuffle(bin_pairs)
                n_take = min(len(bin_pairs), samples_per_bin)
                result.extend(bin_pairs[:n_take])

            # Neu chua du, lay them tu cac bins con du
            remaining = n_samples - len(result)
            if remaining > 0:
                leftover = []
                for bin_id in sorted(bins.keys()):
                    bin_pairs = bins[bin_id]
                    n_taken = min(len(bin_pairs), samples_per_bin)
                    leftover.extend(bin_pairs[n_taken:])
                random.shuffle(leftover)
                result.extend(leftover[:remaining])

            return result

        # Uu tien medical truoc
        if config['prioritize_medical']:
            # Lay toi da 70% la medical, 30% la non-medical
            n_medical = min(len(medical_pairs), int(config['max_samples'] * 0.7))
            n_non_medical = config['max_samples'] - n_medical

            sampled_medical = stratified_sample(medical_pairs, n_medical)
            sampled_non_medical = stratified_sample(non_medical_pairs, n_non_medical)

            valid_pairs = sampled_medical + sampled_non_medical
        else:
            valid_pairs = stratified_sample(valid_pairs, config['max_samples'])

        # Shuffle lai de tron deu
        random.shuffle(valid_pairs)

    # Luu ket qua
    output_en = f"{output_prefix}.en.txt"
    output_vi = f"{output_prefix}.vi.txt"

    print(f"\nSaving {len(valid_pairs):,} pairs to {output_en} and {output_vi}...")

    with open(output_en, 'w', encoding='utf-8') as f_en, \
         open(output_vi, 'w', encoding='utf-8') as f_vi:
        for en, vi, _ in valid_pairs:
            f_en.write(en + '\n')
            f_vi.write(vi + '\n')

    # In thong ke
    print("\n" + "="*60)
    print("FILTERING STATISTICS")
    print("="*60)
    print(f"Total original pairs: {len(en_lines):,}")
    print(f"Valid pairs: {stats['valid']:,}")
    print(f"  - With medical keywords: {stats['has_medical_keyword']:,}")
    print(f"\nRemoved pairs:")
    print(f"  - Duplicates: {stats['duplicate']:,}")
    print(f"  - Too short (<{config['min_words']} words): {stats['too_short']:,}")
    print(f"  - Too long (>{config['max_words']} words): {stats['too_long']:,}")
    print(f"  - Bad length ratio: {stats['bad_ratio']:,}")
    print(f"  - Excessive special chars: {stats['special_chars']:,}")
    print(f"  - Too many numbers: {stats['too_many_numbers']:,}")
    print(f"  - Mostly uppercase: {stats['uppercase']:,}")
    print(f"  - No Vietnamese chars: {stats['no_vietnamese']:,}")
    print(f"  - No medical keywords: {stats['no_medical_keyword']:,}")
    print(f"  - Empty: {stats['empty']:,}")
    print(f"  - Overlap with test: {stats['overlap_with_test']:,}")
    print(f"\nFinal result: {len(valid_pairs):,} pairs")
    print(f"Retention rate: {len(valid_pairs)/len(en_lines)*100:.1f}%")

    return valid_pairs, stats


def main():
    parser = argparse.ArgumentParser(description='Filter medical translation data')
    parser.add_argument('--en_file', type=str, default='data/VLSP_MT_dataset/train.en.txt',
                        help='English file')
    parser.add_argument('--vi_file', type=str, default='data/VLSP_MT_dataset/train.vi.txt',
                        help='Vietnamese file')
    parser.add_argument('--output_prefix', type=str, default='data/VLSP_MT_dataset/train_filtered',
                        help='Output file prefix')
    parser.add_argument('--min_words', type=int, default=5,
                        help='Minimum words')
    parser.add_argument('--max_words', type=int, default=150,
                        help='Maximum words')
    parser.add_argument('--min_ratio', type=float, default=0.4,
                        help='Minimum EN/VI length ratio')
    parser.add_argument('--max_ratio', type=float, default=2.5,
                        help='Maximum EN/VI length ratio')
    parser.add_argument('--medical_only', action='store_true',
                        help='Keep only sentences with medical keywords')
    parser.add_argument('--min_medical_keywords', type=int, default=1,
                        help='Minimum medical keywords required')
    parser.add_argument('--prioritize_medical', action='store_true',
                        help='Prioritize sentences with medical keywords')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples (None = unlimited)')
    parser.add_argument('--test_en_file', type=str, default='data/VLSP_MT_dataset/test.en.txt',
                        help='Test English file (to remove overlap)')
    parser.add_argument('--test_vi_file', type=str, default='data/VLSP_MT_dataset/test.vi.txt',
                        help='Test Vietnamese file (to remove overlap)')
    parser.add_argument('--no_test_filter', action='store_true',
                        help='Disable filtering overlap with test set')

    args = parser.parse_args()

    config = {
        'min_words': args.min_words,
        'max_words': args.max_words,
        'min_ratio': args.min_ratio,
        'max_ratio': args.max_ratio,
        'medical_only': args.medical_only,
        'min_medical_keywords': args.min_medical_keywords,
        'prioritize_medical': args.prioritize_medical,
        'max_samples': args.max_samples,
        'test_en_file': None if args.no_test_filter else args.test_en_file,
        'test_vi_file': None if args.no_test_filter else args.test_vi_file,
    }

    filter_data(args.en_file, args.vi_file, args.output_prefix, config)


if __name__ == '__main__':
    main()
