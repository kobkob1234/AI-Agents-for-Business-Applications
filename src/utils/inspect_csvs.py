import glob
import os
import csv

def inspect_csvs():
    files = glob.glob("data/*.csv")
    formats = {} # key: (has_header_guess, num_cols), value: count
    
    print(f"Inspecting {len(files)} files...")
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8', errors='replace') as csvfile:
                reader = csv.reader(csvfile)
                row1 = next(reader, None)
                if not row1: continue
                
                num_cols = len(row1)
                has_acn = "ACN" in row1[0] if row1 else False
                
                key = (has_acn, num_cols)
                if key not in formats:
                    formats[key] = []
                formats[key].append(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print("\nSummary of formats (Has Header 'ACN', Num Cols):")
    for key, f_list in formats.items():
        print(f"Format {key}: {len(f_list)} files. Example: {f_list[0]}")

if __name__ == "__main__":
    inspect_csvs()
