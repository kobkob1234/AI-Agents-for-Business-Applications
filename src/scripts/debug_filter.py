
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.data_loader import load_data, preprocess_data
from src.tools.filtering import StructuredFilter

def debug_data():
    print("Loading data...")
    try:
        raw_df = load_data()
        print(f"Raw data shape: {raw_df.shape}")
        print(f"Raw columns: {list(raw_df.columns)[:10]}...")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\nPreprocessing data...")
    try:
        df = preprocess_data(raw_df)
        print(f"Preprocessed data shape: {df.shape}")
        print(f"Preprocessed columns: {list(df.columns)}")
        
        if 'Event_Date' in df.columns:
             print(f"\nEvent_Date sample:\n{df['Event_Date'].head()}")
             print(f"Event_Date dtype: {df['Event_Date'].dtype}")
        else:
             print("\nERROR: Event_Date not found in preprocessed data!")

    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return

    print("\nTesting StructuredFilter...")
    filter_tool = StructuredFilter(df)
    
    # Test 1: Date Filter
    print("\n--- Test 1: Date Filter (2024-01-01 onwards) ---")
    try:
        res = filter_tool.filter_data({'Date_Start': '2024-01-01'})
        print(f"Filtered count: {len(res)}")
    except Exception as e:
        print(f"Date filter failed: {e}")

    # Test 2: Make Model Filter
    print("\n--- Test 2: Make Model Filter ('B737') ---")
    try:
        # Find a make model that exists
        sample_model = df['Make Model Name'].dropna().iloc[0] if 'Make Model Name' in df.columns else "Unknown"
        print(f"Using sample model: {sample_model}")
        res = filter_tool.filter_data({'Make_Model': sample_model})
        print(f"Filtered count: {len(res)}")
    except Exception as e:
        print(f"Make Model filter failed: {e}")

    # Test 3: Location Filter
    print("\n--- Test 3: Location Filter ('ZZZ') ---")
    try:
         # Find a location that exists
        sample_loc = df['Locale Reference'].dropna().iloc[0] if 'Locale Reference' in df.columns else "Unknown"
        print(f"Using sample location: {sample_loc}")
        res = filter_tool.filter_data({'Airport': sample_loc})
        print(f"Filtered count: {len(res)}")
    except Exception as e:
        print(f"Location filter failed: {e}")

if __name__ == "__main__":
    debug_data()
