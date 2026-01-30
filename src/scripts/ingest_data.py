
import os
import sys
import argparse
import time
import pandas as pd
from dotenv import load_dotenv

# Load env before imports
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.data_loader import load_data, preprocess_data
from src.tools.semantic_search import SemanticSearch

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("⚠️ tiktoken not found. Cost estimation will be approximate.")

COST_PER_1M_TOKENS = 0.02  # $0.02 per 1M tokens for text-embedding-3-small
Budget_Limit = 2.00  # Pause if cost exceeds this

def estimate_cost(df: pd.DataFrame, text_col: str = "Full_Narrative"):
    """
    Counts tokens and estimates cost.
    Returns (total_tokens, estimated_cost_usd)
    """
    if not HAS_TIKTOKEN:
        # Fallback: Approx 4 chars per token
        total_chars = df[text_col].str.len().sum()
        tokens = total_chars / 4
        return int(tokens), (tokens / 1_000_000) * COST_PER_1M_TOKENS
    
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    total_tokens = 0
    
    # Process in chunks to avoid memory spike
    for text in df[text_col]:
        if isinstance(text, str):
            total_tokens += len(enc.encode(text))
            
    cost = (total_tokens / 1_000_000) * COST_PER_1M_TOKENS
    return total_tokens, cost

def main():
    parser = argparse.ArgumentParser(description="Ingest ASRS data into Pinecone")
    parser.add_argument("--full", action="store_true", help="Ingest FULL dataset (overrides limit)")
    parser.add_argument("--limit", type=int, default=150, help="Number of records to ingest (default: 150)")
    parser.add_argument("--dry-run", action="store_true", help="Calculate cost without ingesting")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for ingestion")
    
    args = parser.parse_args()
    
    print("🚀 Starting Ingestion Script...")
    
    # 1. Load Data
    data_path = "data"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "../../data")
    
    print(f"Loading data from '{data_path}'...")
    df = load_data(data_path)
    if df.empty:
        print("❌ No data found.")
        return

    print("Preprocessing data...")
    df = preprocess_data(df)
    
    # Filter empty narratives
    df = df[df['Full_Narrative'].notna() & (df['Full_Narrative'].str.strip() != "")]
    print(f"✅ Loaded {len(df)} valid records.")

    # 2. Apply Limits
    if args.full:
        print("⚠️  MODE: FULL DATASET INGESTION")
        df_to_ingest = df
    else:
        print(f"⚠️  MODE: LIMITED ({args.limit} records). Use --full to ingest all.")
        # Sort by date if possible
        if 'Event_Date' in df.columns:
            df['Event_Date'] = pd.to_datetime(df['Event_Date'], errors='coerce')
            df = df.sort_values(by='Event_Date', ascending=False)
        df_to_ingest = df.head(args.limit)

    count = len(df_to_ingest)
    print(f"Preparing to ingest {count} documents.")

    # 3. Cost Estimation
    print("\n💰 Calculating Cost Estimate...")
    tokens, cost = estimate_cost(df_to_ingest)
    
    print(f"--------------------------------------------------")
    print(f"Total Documents : {count}")
    print(f"Total Tokens    : {tokens:,}")
    print(f"Estimated Cost  : ${cost:.4f}")
    print(f"--------------------------------------------------")

    if args.dry_run:
        print("✅ Dry run complete. No data ingested.")
        return

    # Safety Check
    if cost > Budget_Limit:
        print(f"\n⚠️  WARNING: Estimated cost (${cost:.4f}) exceeds safety limit (${Budget_Limit}).")
        confirm = input("Are you SURE you want to proceed? (type 'YES' to confirm): ")
        if confirm != "YES":
            print("❌ Aborted by user.")
            return

    # 4. Ingest
    if not os.environ.get("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY missing.")
        return

    try:
        print(f"\n📡 Connecting to Vector Store...")
        search_tool = SemanticSearch(collection_name="asrs-reports")
        search_tool.ingest_data(df_to_ingest, batch_size=args.batch_size)
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")

if __name__ == "__main__":
    main()
