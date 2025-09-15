#!/usr/bin/env python3

import csv
import os
from pathlib import Path
from collections import Counter, defaultdict
import sys

def get_csv_files(datasets_path):
    """Get all CSV files from the datasets directory."""
    datasets_dir = Path(datasets_path)
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_path}")
    
    csv_files = list(datasets_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != ".DS_Store"]  # Filter out system files
    csv_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file.name}")
    print()
    
    return csv_files

def get_column_info(csv_file):
    """Extract column information from a CSV file."""
    try:
        # Read only the first row to get column names
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            columns = next(reader)  # Read the header row
        
        # Get file size for additional info
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)
        
        return {
            'filename': csv_file.name,
            'columns': columns,
            'num_columns': len(columns),
            'file_size_mb': round(file_size_mb, 2)
        }
    except Exception as e:
        return {
            'filename': csv_file.name,
            'columns': None,
            'num_columns': 0,
            'file_size_mb': 0,
            'error': str(e)
        }

def compare_columns(file_infos):
    """Compare column structures across all files."""
    print("=" * 70)
    print("COLUMN COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Extract all column sets
    column_sets = []
    valid_files = []
    
    for info in file_infos:
        if info['columns'] is not None:
            column_sets.append(set(info['columns']))
            valid_files.append(info)
        else:
            print(f"⚠️  ERROR reading {info['filename']}: {info.get('error', 'Unknown error')}")
    
    if not column_sets:
        print("❌ No valid CSV files found!")
        return False
    
    # Check if all column sets are identical
    first_column_set = column_sets[0]
    all_identical = all(col_set == first_column_set for col_set in column_sets)
    
    print(f"\n📊 FILES ANALYZED: {len(valid_files)}")
    print(f"🏷️  COLUMNS PER FILE: {valid_files[0]['num_columns']}")
    
    if all_identical:
        print("✅ RESULT: All files have IDENTICAL column structures!")
        print("✅ FILES CAN BE SAFELY MERGED")
        
        # Display the common column structure
        print(f"\n📋 COMMON COLUMN STRUCTURE ({len(first_column_set)} columns):")
        for i, col in enumerate(valid_files[0]['columns'], 1):
            print(f"  {i:2d}. {col}")
            
    else:
        print("❌ RESULT: Column structures are NOT identical!")
        print("❌ FILES CANNOT BE MERGED WITHOUT PREPROCESSING")
        
        # Find differences
        all_columns = set()
        for col_set in column_sets:
            all_columns.update(col_set)
        
        # Show which columns are missing from which files
        print(f"\n🔍 DETAILED ANALYSIS:")
        column_file_matrix = defaultdict(list)
        
        for info in valid_files:
            for col in all_columns:
                if col in info['columns']:
                    column_file_matrix[col].append(info['filename'])
        
        # Find columns that are not in all files
        inconsistent_columns = []
        for col, files in column_file_matrix.items():
            if len(files) != len(valid_files):
                inconsistent_columns.append((col, files))
        
        if inconsistent_columns:
            print(f"\n⚠️  INCONSISTENT COLUMNS ({len(inconsistent_columns)}):")
            for col, files in inconsistent_columns:
                missing_files = [f['filename'] for f in valid_files if f['filename'] not in files]
                print(f"  - '{col}':")
                print(f"    Present in: {', '.join(files)}")
                if missing_files:
                    print(f"    Missing from: {', '.join(missing_files)}")
    
    # Show file details
    print(f"\n📁 FILE DETAILS:")
    for info in valid_files:
        print(f"  {info['filename']:25} | {info['num_columns']:2d} columns | {info['file_size_mb']:6.2f} MB")
    
    return all_identical



def main():
    """Main function to run the column comparison analysis."""
    # Define the datasets path
    datasets_path = Path(__file__).parent.parent / "datasets"
    
    try:
        print("🚂 TRAIN DISRUPTION DATASETS - COLUMN ANALYSIS")
        print("=" * 70)
        
        # Get all CSV files
        csv_files = get_csv_files(datasets_path)
        
        if not csv_files:
            print("❌ No CSV files found in the datasets directory!")
            return
        
        # Analyze each file
        print("📖 Reading column information from each file...")
        file_infos = []
        for csv_file in csv_files:
            print(f"   Processing {csv_file.name}...")
            info = get_column_info(csv_file)
            file_infos.append(info)
        
        # Compare columns
        can_merge = compare_columns(file_infos)
        

        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE ✨")
        
        return can_merge
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
