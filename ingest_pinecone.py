
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Load env before imports to ensure keys are ready
load_dotenv()

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import load_data, preprocess_data
from src.tools.semantic_search import SemanticSearch

def main():
    print("🚀 Starting Pinecone Ingestion Script...")
    
    # 1. Load Data
    print("Loading data from 'data/' directory...")
    df = load_data("data")
    
    if df.empty:
        print("❌ No data found! Please ensure 'data/' directory has CSV files.")
        return

    # PREPROCESS to create 'Full_Narrative'
    print("Preprocessing data...")
    df = preprocess_data(df)

    print(f"✅ Loaded {len(df)} total records.")

    # 2. Sample Data (To save cost/time)
    # Try to sort by date if possible
    if 'Event_Date' in df.columns:
        print("Sorting by date to get recent reports...")
        try:
            df['Event_Date'] = pd.to_datetime(df['Event_Date'], errors='coerce')
            df = df.sort_values(by='Event_Date', ascending=False)
        except Exception as e:
            print(f"⚠️ Date sorting failed: {e}. Using default order.")
    
    SAMPLE_SIZE = 150
    df_sample = df.head(SAMPLE_SIZE)
    print(f"Taking top {SAMPLE_SIZE} records for ingestion.")

    # 3. Initialize Semantic Search (connects to Pinecone via Manager)
    # Force Pinecone check by ensuring env var is visible (it is loaded by dotenv)
    if not os.environ.get("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY not found in environment!")
        return

    try:
        search_tool = SemanticSearch(collection_name="asrs-reports")
        
        # 4. Ingest
        print(f"📡 Ingesting into Pinecone index: '{search_tool.manager.index_name}'...")
        search_tool.ingest_data(df_sample, batch_size=50)
        
        print("🎉 Ingestion Complete Check your Pinecone dashboard.")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")

if __name__ == "__main__":
    main()
