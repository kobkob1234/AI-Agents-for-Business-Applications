import pandas as pd
import numpy as np
from typing import Dict, Any

class TrendAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def detect_anomalies(self, df_subset: pd.DataFrame, time_col: str = 'Event_Date', metric_col: str = 'ACN') -> Dict[str, Any]:
        """
        Analyzes the trend of report counts over time in the subset.
        Returns a dictionary indicating if there's a surge.
        """
        if len(df_subset) < 5:
             return {"status": "insufficient_data", "details": "Less than 5 reports"}
             
        if time_col not in df_subset.columns:
             return {"status": "error", "details": "Time column missing"}
             
        # Resample by month
        df_subset = df_subset.sort_values(time_col)
        monthly_counts = df_subset.set_index(time_col).resample('ME')[metric_col].count()
        
        # Simple Z-Score based anomaly detection on the count
        if len(monthly_counts) < 3:
             return {"status": "insufficient_data"}
             
        mean = monthly_counts.mean()
        std = monthly_counts.std()
        
        if std == 0:
             return {"status": "stable", "z_score": 0}
             
        # Check specific recent behavior?
        # Or just return the statistics
        
        # Calculate recent trend (slope of last few points)
        
        return {
            "status": "analyzed",
            "mean_monthly_reports": float(mean),
            "std_dev": float(std),
            "max_monthly": float(monthly_counts.max()),
            "last_3_months_avg": float(monthly_counts.iloc[-3:].mean()) if len(monthly_counts) >=3 else float(mean)
        }
