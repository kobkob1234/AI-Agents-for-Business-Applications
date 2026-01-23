import os
import glob
import pandas as pd
import time
from src.utils.supabase_manager import supabase_manager
from src.utils.columns import COLUMN_NAMES
from datetime import datetime

# ESTIMATED SIZE PER ROW (Bytes)
# Text narrative is main factor. Avg ~2KB.
BYTES_PER_ROW = 2048 
MAX_DB_SIZE_MB = 480 # Safety buffer below 500MB

def get_existing_years(api):
    try:
        # Fetch unique years from DB
        # Note: listing all rows is expensive, we rely on year aggregation if possible
        # Or simpler: we pass the years we KNOW we did (2013-2026)
        # But better to query. 
        # Since we can't do distinct efficiently on huge non-indexed columns without timeout sometimes,
        # We will assume years based on row counts check or just try to upload and upsert (safe).
        # Actually, let's just query row count to estimate size.
        res = api.table("asrs_reports").select("id", count="exact").execute()
        return res.count
    except Exception as e:
        print(f"Error checking count: {e}")
        return 0

def backfill_remaining():
    if not supabase_manager.is_connected():
        print("❌ Supabase not connected.")
        return

    api = supabase_manager.client
    table = "asrs_reports"

    # 1. Get current status
    total_rows = get_existing_years(api)
    if total_rows is None: total_rows = 0
    current_size_mb = (total_rows * BYTES_PER_ROW) / (1024*1024)
    print(f"📉 Current Status: {total_rows} rows. Est Size: {current_size_mb:.2f} MB")

    # 2. Identify years to process (2012 down to 1988)
    years_to_backfill = list(range(2012, 1987, -1))
    
    data_dir = "data"
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))

    for year in years_to_backfill:
        # Check capacity
        if current_size_mb > MAX_DB_SIZE_MB:
            print(f"⚠️ Reached Max Size Limit ({current_size_mb:.2f} MB > {MAX_DB_SIZE_MB} MB). Stopping.")
            break

        print(f"\nProcessing Year: {year}")
        
        # Find files for this year
        target_files = []
        for f in all_files:
            try:
                y = int(os.path.basename(f).split('_')[1])
                if y == year:
                    target_files.append(f)
            except:
                continue
        
        if not target_files:
            print(f"   No files found.")
            continue

        # Upload files
        year_records = 0
        for file_path in sorted(target_files):
            # ... (Reuse processing logic from backfill_year.py) ...
            try:
                # READ CSV
                has_header = False
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    if "ACN" in f.readline(): has_header = True
                
                if has_header:
                    df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
                else:
                    df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES, on_bad_lines='skip', encoding='utf-8', low_memory=False)

                # EXTRACT RECORDS
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
                        except: pass
                        narrative = str(row.get('Narrative', ''))
                        if 'Narrative.1' in row: narrative += " " + str(row['Narrative.1'])
                        
                        records.append({
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
                            # "raw_data": column dropped for optimization
                        })
                    except: continue

                # UPSERT
                batch_size = 100
                for i in range(0, len(records), batch_size):
                    batch = records[i:i+batch_size]
                    try:
                        api.table(table).upsert(batch, on_conflict="acn").execute()
                        year_records += len(batch)
                        total_rows += len(batch)
                    except Exception as e:
                        pass
                print(f"   Processed {os.path.basename(file_path)}: +{len(records)} recs", end='\r')

            except Exception as e:
                print(f"   Error file {file_path}: {e}")

        # Post-Year Check
        current_size_mb = (total_rows * BYTES_PER_ROW) / (1024*1024)
        print(f"\n✅ Completed {year}. Added {year_records} rows. New Total: {total_rows}. Est Size: {current_size_mb:.2f} MB")
        
        # Artificial sleep to prevent rate limits
        time.sleep(1)

if __name__ == "__main__":
    backfill_remaining()
