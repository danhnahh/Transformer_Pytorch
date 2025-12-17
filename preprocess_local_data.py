#!/usr/bin/env python3
"""
Preprocess local IWSLT'15 dataset to clean HTML entities.
This script replaces HTML entities like &apos; &quot; &amp; etc. with their actual characters.
"""

import html
import os
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


def process_file(input_path, output_path=None, backup=True):
    """
    Process a single file to clean HTML entities.
    
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
    
    # Count entities before cleaning
    entities_count = 0
    for line in lines:
        entities_count += line.count('&apos;')
        entities_count += line.count('&quot;')
        entities_count += line.count('&amp;')
        entities_count += line.count('&#91;')
        entities_count += line.count('&#93;')
        entities_count += line.count('&lt;')
        entities_count += line.count('&gt;')
    
    if entities_count > 0:
        print(f"  Found {entities_count} HTML entities")
    
    # Clean the lines
    cleaned_lines = [clean_html_entities(line) for line in lines]
    
    # Write cleaned content
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    print(f"  Cleaned and saved to: {output_path}")
    return entities_count


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
    processed_files = 0
    
    print("=" * 60)
    print("Preprocessing IWSLT'15 Dataset - Cleaning HTML Entities")
    print("=" * 60)
    print()
    
    for filename in files_to_process:
        file_path = data_dir / filename
        if file_path.exists():
            entities = process_file(file_path, backup=True)
            total_entities += entities
            processed_files += 1
            print()
        else:
            print(f"Warning: File not found: {file_path}")
            print()
    
    print("=" * 60)
    print(f"Preprocessing Complete!")
    print(f"  Files processed: {processed_files}")
    print(f"  Total HTML entities cleaned: {total_entities}")
    print(f"  Backups saved with .bak extension")
    print("=" * 60)
    
    # Show some examples
    if total_entities > 0:
        print()
        print("Common replacements made:")
        print("  &apos;  →  '")
        print("  &quot;  →  \"")
        print("  &amp;   →  &")
        print("  &#91;   →  [")
        print("  &#93;   →  ]")
        print("  &lt;    →  <")
        print("  &gt;    →  >")


if __name__ == "__main__":
    main()
