import os
import glob
import pandas as pd
import sys
from src.utils.supabase_manager import supabase_manager
from src.utils.columns import COLUMN_NAMES
from datetime import datetime

def backfill_year(target_year):
    if not supabase_manager.is_connected():
        print("❌ Supabase not connected. check env vars.")
        return

    data_dir = "data"
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    target_files = []
    for f in all_files:
        try:
            # Expected format: asrs_YYYY_Month.csv
            year = int(os.path.basename(f).split('_')[1])
            if year == target_year:
                target_files.append(f)
        except:
            continue
            
    if not target_files:
        print(f"No files found for year {target_year}")
        return

    api = supabase_manager.client
    table = "asrs_reports"
    
    print(f"Found {len(target_files)} CSV files for {target_year}.")
    
    total_uploaded = 0
    
    for file_path in sorted(target_files):
        print(f"Processing {os.path.basename(file_path)}...")
        try:
            has_header = False
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                first_line = f.readline()
                if "ACN" in first_line:
                    has_header = True
            
            if has_header:
                df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
            else:
                df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES, on_bad_lines='skip', encoding='utf-8', low_memory=False)
                
            records = []
            for _, row in df.iterrows():
                try:
                    acn = str(row.get('ACN', '')).strip()
                    if not acn or acn.lower() == 'nan': continue
                    
                    raw_date = str(row.get('Date', '')) 
                    event_date = None
                    try:
                        dt = datetime.strptime(raw_date, '%Y%m')
                        event_date = dt.strftime('%Y-%m-%d')
                    except:
                        pass
                        
                    narrative = str(row.get('Narrative', '')) 
                    if 'Narrative.1' in row: narrative += " " + str(row['Narrative.1'])
                    
                    rec = {
                        "acn": acn,
                        "event_date": event_date,
                        "report_year_month": raw_date,
                        "location": str(row.get('Locale Reference', '')),
                        "state": str(row.get('State', '')),
                        "aircraft_make_model": str(row.get('Make Model Name', '')),
                        "operator": str(row.get('Operator', '')),
                        "flight_phase": str(row.get('Flight Phase', '')),
                        "event_type": str(row.get('Event Type', '')),
                        "narrative": narrative[:10000], 
                        "raw_data": row.replace({float('nan'): None}).to_dict()
                    }
                    records.append(rec)
                except:
                    continue
            
            # Upsert
            batch_size = 100
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                try:
                    api.table(table).upsert(batch, on_conflict="acn").execute()
                    total_uploaded += len(batch)
                    print(f"   Uploaded {total_uploaded} records...", end='\r')
                except Exception as e:
                    if "21000" not in str(e):
                         print(f"   Batch error: {e}")
                    
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            
    print(f"\n✅ Backfill Complete for {target_year}. Uploaded {total_uploaded} records.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m src.scripts.backfill_year <YEAR>")
    else:
        backfill_year(int(sys.argv[1]))
