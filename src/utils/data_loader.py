import pandas as pd
import os
import glob
from src.utils.columns import COLUMN_NAMES

def load_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads all CSV files from the specified directory and merges them into a single DataFrame.
    
    Args:
        data_dir (str): Path to the directory containing ASRS CSV files.
        
    Returns:
        pd.DataFrame: Merged DataFrame containing all reports.
    """
    if not os.path.exists(data_dir):
        print(f"WARNING: Data directory '{data_dir}' not found. Running without local data.")
        print("Note: Semantic search will use vector database only.")
        return pd.DataFrame()
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files. Loading...")
    
    dfs = []
    for file in csv_files:
        try:
            # Check for header
            has_header = False
            with open(file, 'r', encoding='utf-8', errors='replace') as f:
                first_line = f.readline()
                if "ACN" in first_line:
                    has_header = True
            
            # Read CSV
            if has_header:
                df = pd.read_csv(file, on_bad_lines='skip', encoding='utf-8', low_memory=False)
                # Ensure columns match our expected logical names if possible, or just trust them.
                # If they have different columns, concat will fill NaNs.
            else:
                # Use our hardcoded names
                # Note: If the file has fewer columns than we have names, pandas will fill NaNs?
                # Or if it has MORE?
                # We should assume 126 cols.
                df = pd.read_csv(file, header=None, names=COLUMN_NAMES, on_bad_lines='skip', encoding='utf-8', low_memory=False)
            
            # Basic cleanup
            # Filter out rows that might be headers repeated or empty
            if 'ACN' in df.columns:
                 df = df[pd.to_numeric(df['ACN'], errors='coerce').notna()]
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
            
    if not dfs:
        return pd.DataFrame()
        
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Successfully loaded {len(merged_df)} records.")
    
    return merged_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic preprocessing on the ASRS DataFrame.
    """
    # 1. Combine Narratives
    # 'Narrative' and 'Narrative.1' might both exist
    # 'Synopsis' exists
    
    # helper to clean string
    def clean_text(x):
        return str(x) if pd.notna(x) else ""
    
    # Create a single 'Full_Narrative' column
    # Check which columns actually exist in the merged df (concat might produce duplications or misses)
    
    narrative_cols = [c for c in df.columns if 'Narrative' in c or 'Synopsis' in c]
    
    # Initialize Full_Narrative column
    df['Full_Narrative'] = ""
    
    # Safely combine narrative columns if they exist
    if 'Narrative' in df.columns:
        df['Full_Narrative'] = df['Narrative'].apply(clean_text)
    
    if 'Narrative.1' in df.columns:
         df['Full_Narrative'] += " " + df['Narrative.1'].apply(clean_text)
         
    if 'Synopsis' in df.columns:
         df['Full_Narrative'] += " Synopsis: " + df['Synopsis'].apply(clean_text)
         
    # 2. Date parsing
    if 'Date' in df.columns:
        # ASRS Dates are YYYYMM (int or str)
        # Convert to datetime (use 1st of month)
        df['Event_Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m', errors='coerce')
    
    return df

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    print(df.head())
    print(df.info())
