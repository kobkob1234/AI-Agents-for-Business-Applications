import pandas as pd
import os
import glob
from src.utils.columns import COLUMN_NAMES

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

def load_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads data from Supabase (Primary) or local CSV files (Fallback).
    """
    # 1. Try Supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if HAS_SUPABASE and supabase_url and supabase_key:
        try:
            print("Attempting to load data from Supabase (Primary)...")
            client: Client = create_client(supabase_url, supabase_key)
            
            # Fetch all rows with pagination
            all_rows = []
            batch_size = 1000
            start = 0
            
            while True:
                print(f"Fetching rows {start} to {start + batch_size}...")
                response = client.table("asrs_reports").select("*").order("ACN").range(start, start + batch_size - 1).execute()
                data = response.data
                if not data:
                    break
                all_rows.extend(data)
                
                if len(data) < batch_size:
                    break
                start += batch_size
            
            if len(all_rows) > 0:
                print(f"✅ Successfully loaded {len(all_rows)} records from Supabase.")
                df = pd.DataFrame(all_rows)
                # Ensure columns match expected schema if needed
                return df
            else:
                 print("⚠️ Supabase table 'asrs_reports' empty or not found. Falling back to local CSVs.")
        except Exception as e:
             print(f"⚠️ Supabase load failed: {e}. Falling back to local CSVs.")
    
    # 2. Fallback to Local CSV
    return load_from_csv(data_dir)

def load_from_csv(data_dir: str) -> pd.DataFrame:
    """
    Loads all CSV files from the specified directory and merges them into a single DataFrame.
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
            else:
                df = pd.read_csv(file, header=None, names=COLUMN_NAMES, on_bad_lines='skip', encoding='utf-8', low_memory=False)
            
            # Basic cleanup
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
        
        # Log warning if any dates failed to parse
        invalid_dates = df['Event_Date'].isna().sum()
        if invalid_dates > 0:
            print(f"Warning: {invalid_dates} rows had invalid dates (coerced to NaT)")
    
    return df

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    print(df.head())
    print(df.info())
