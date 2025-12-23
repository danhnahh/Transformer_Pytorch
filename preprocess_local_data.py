#!/usr/bin/env python3
"""
Preprocess local IWSLT'15 dataset to clean HTML entities and fix spacing.
This script:
1. Replaces HTML entities like &apos; &quot; &amp; etc. with their actual characters
2. Fixes unnecessary spaces before punctuation marks
"""

import html
import os
import re
from pathlib import Path


def clean_html_entities(text):
    """Replace HTML entities with their corresponding characters."""
    # First use html.unescape for standard entities
    text = html.unescape(text)
    
    # Handle additional numeric entities that might not be covered
    replacements = {
        '&#91;': '[',
        '&#93;': ']',
    }
    
    for entity, char in replacements.items():
        text = text.replace(entity, char)
    
    return text


def fix_punctuation_spacing(text):
    """Remove unnecessary spaces before punctuation marks and inside quotes."""
    # Remove space before common punctuation marks
    # Handles: comma, period, semicolon, colon, exclamation, question mark, closing brackets
    text = re.sub(r'\s+([,.:;!?)\]}])', r'\1', text)
    
    # Remove space before apostrophes that are part of contractions (e.g., "don 't" -> "don't")
    text = re.sub(r'\s+\'(\w)', r"'\1", text)
    
    # Fix spaces around dashes and hyphens (multiple spaces to single space)
    text = re.sub(r'\s*(--)\s*', r' \1 ', text)
    
    # Fix spaces inside quotation marks: " text " -> "text"
    # Handle double quotes
    text = re.sub(r'"\s+', r'"', text)  # Remove space after opening quote
    text = re.sub(r'\s+"', r'"', text)  # Remove space before closing quote
    
    # Handle single quotes (be careful not to affect contractions)
    text = re.sub(r"'\s+(\w)", r"'\1", text)  # Remove space after opening single quote before word
    text = re.sub(r'(\w)\s+\'', r"\1'", text)  # Remove space before closing single quote after word
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_text(text):
    """Apply all text cleaning operations."""
    text = clean_html_entities(text)
    text = fix_punctuation_spacing(text)
    return text


def process_file(input_path, output_path=None, backup=True):
    """
    Process a single file to clean HTML entities and fix spacing.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file (if None, overwrites input)
        backup: Whether to create a backup before overwriting
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    
    # Create backup if overwriting and backup is requested
    if backup and output_path == input_path:
        backup_path = input_path.with_suffix(input_path.suffix + '.bak')
        if backup_path.exists():
            print(f"Backup already exists: {backup_path}")
        else:
            print(f"Creating backup: {backup_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    # Read, clean, and write
    print(f"Processing: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Count issues before cleaning
    entities_count = 0
    spacing_issues = 0
    for line in lines:
        entities_count += line.count('&apos;')
        entities_count += line.count('&quot;')
        entities_count += line.count('&amp;')
        entities_count += line.count('&#91;')
        entities_count += line.count('&#93;')
        entities_count += line.count('&lt;')
        entities_count += line.count('&gt;')
        
        # Count spacing issues (space before punctuation)
        spacing_issues += len(re.findall(r'\s+[,.:;!?)\]}]', line))
        spacing_issues += len(re.findall(r'\s+\'', line))
    
    if entities_count > 0:
        print(f"  Found {entities_count} HTML entities")
    if spacing_issues > 0:
        print(f"  Found {spacing_issues} spacing issues")
    
    # Clean the lines
    cleaned_lines = [clean_text(line) + '\n' if not line.endswith('\n') else clean_text(line.rstrip('\n')) + '\n' for line in lines]
    
    # Write cleaned content
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    print(f"  Cleaned and saved to: {output_path}")
    return entities_count, spacing_issues


def main():
    """Process all IWSLT'15 data files."""
    data_dir = Path(__file__).parent / "data" / "archive" / "IWSLT'15 en-vi"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Files to process
    files_to_process = [
        'train.en.txt',
        'train.vi.txt',
        'tst2012.en.txt',
        'tst2012.vi.txt',
        'tst2013.en.txt',
        'tst2013.vi.txt',
    ]
    
    total_entities = 0
    total_spacing = 0
    processed_files = 0
    
    print("=" * 70)
    print("Preprocessing IWSLT'15 Dataset")
    print("- Cleaning HTML Entities")
    print("- Fixing Punctuation Spacing")
    print("=" * 70)
    print()
    
    for filename in files_to_process:
        file_path = data_dir / filename
        if file_path.exists():
            entities, spacing = process_file(file_path, backup=True)
            total_entities += entities
            total_spacing += spacing
            processed_files += 1
            print()
        else:
            print(f"Warning: File not found: {file_path}")
            print()
    
    print("=" * 70)
    print(f"Preprocessing Complete!")
    print(f"  Files processed: {processed_files}")
    print(f"  Total HTML entities cleaned: {total_entities}")
    print(f"  Total spacing issues fixed: {total_spacing}")
    print(f"  Backups saved with .bak extension")
    print("=" * 70)
    
    # Show some examples
    if total_entities > 0 or total_spacing > 0:
        print()
        print("Transformations applied:")
        if total_entities > 0:
            print("  1. HTML entities:")
            print("     &apos;  →  '")
            print("     &quot;  →  \"")
            print("     &amp;   →  &")
            print("     &#91;   →  [")
            print("     &#93;   →  ]")
        if total_spacing > 0:
            print("  2. Punctuation spacing:")
            print("     'text , word'  →  'text, word'")
            print("     'word .'       →  'word.'")
            print("     'don 't'       →  'don't'")
            print("     'word : next'  →  'word: next'")


if __name__ == "__main__":
    main()
