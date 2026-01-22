import pandas as pd
from typing import Dict, Any, List

class StructuredFilter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def _get_column(self, df: pd.DataFrame, *possible_names: str) -> str:
        """Find the first matching column name from possible options."""
        for name in possible_names:
            if name in df.columns:
                return name
        return possible_names[0]  # Fallback to first option
        
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
                # Support both 'Make Model Name' and 'make_model_name'
                col = self._get_column(result, 'Make Model Name', 'make_model_name', 'Make_Model_Name')
                if col in result.columns:
                    result = result[result[col].astype(str).str.contains(value, case=False, na=False)]
                
            elif key == 'Airport':
                # Support both 'Locale Reference' and 'locale_reference'
                col = self._get_column(result, 'Locale Reference', 'locale_reference', 'Locale_Reference')
                if col in result.columns:
                    result = result[result[col].astype(str).str.contains(value, case=False, na=False)]
                
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
