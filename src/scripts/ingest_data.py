import argparse
from src.utils.data_loader import load_data, preprocess_data
from src.tools.semantic_search import SemanticSearch
import pandas as pd

def ingest(filter_query=None):
    print("Loading data...")
    df = load_data()
    df = preprocess_data(df)
    
    if filter_query:
        print(f"Filtering data for ingestion with query: '{filter_query}' (in columns)")
        # Simple substring match across all columns for simplicity
        # Or just 'Locale Reference'
        mask = df.astype(str).apply(lambda x: x.str.contains(filter_query, case=False, na=False)).any(axis=1)
        df = df[mask]
        print(f"Filtered down to {len(df)} records.")
        
    search_tool = SemanticSearch()
    search_tool.ingest_data(df, batch_size=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, help="Substring filter to ingest only subset (e.g. 'SAN')")
    args = parser.parse_args()
    
    ingest(args.filter)
