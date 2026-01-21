import pandas as pd
from typing import Dict, Any, List

class StructuredFilter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def filter_data(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Filters the dataset based on structured criteria.
        
        Args:
            filters: Dictionary of column name -> value (or condition)
            Supported inputs:
            - 'Make_Model': str, substring match
            - 'Airport': str (Location), substring match
            - 'Date_Start': datetime/str
            - 'Date_End': datetime/str
        """
        result = self.df.copy()
        
        for key, value in filters.items():
            if not value: continue
            
            if key == 'Make_Model':
                # 'Make Model Name'
                result = result[result['Make Model Name'].astype(str).str.contains(value, case=False, na=False)]
                
            elif key == 'Airport':
                # 'Locale Reference' or 'State Reference' or 'Location'
                # Let's search 'Locale Reference' (e.g. SAN)
                result = result[result['Locale Reference'].astype(str).str.contains(value, case=False, na=False)]
                
            elif key == 'Date_Start':
                if 'Event_Date' in result.columns:
                    pd_date = pd.to_datetime(value)
                    result = result[result['Event_Date'] >= pd_date]
                    
            elif key == 'Date_End':
                if 'Event_Date' in result.columns:
                    pd_date = pd.to_datetime(value)
                    result = result[result['Event_Date'] <= pd_date]
                    
            elif key == 'Event_Date':
                # Exact match or month match
                pass
                
        return result

    def get_statistics(self, df: pd.DataFrame, column: str) -> Dict[str, int]:
        """
        Returns counts of values in a column (e.g. topAnomalies)
        """
        if column in df.columns:
            return df[column].value_counts().head(10).to_dict()
        return {}
