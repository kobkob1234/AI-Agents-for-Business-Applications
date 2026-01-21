"""
Supabase Database Manager for ASI Agent
Handles logging of agent executions to meet the "Supabase: primary database" requirement.
"""
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

# Try importing Supabase, handle if missing
try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    print("WARNING: supabase package not installed. Run: pip install supabase")


class SupabaseManager:
    """
    Manages Supabase connection for logging agent executions.
    Table schema (create in Supabase dashboard):
    
    CREATE TABLE agent_executions (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT now(),
        prompt TEXT NOT NULL,
        entities JSONB,
        steps JSONB,
        final_report TEXT,
        status TEXT DEFAULT 'completed',
        execution_time_ms INTEGER
    );
    """
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.table_name = "agent_executions"
        self._initialize()
        
    def _initialize(self):
        """Initialize Supabase client if credentials are available."""
        if not HAS_SUPABASE:
            print("Supabase library not available - logging disabled")
            return
            
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            print("Supabase credentials not found - logging disabled")
            print("Set SUPABASE_URL and SUPABASE_KEY environment variables")
            return
            
        try:
            self.client = create_client(url, key)
            print(f"Supabase connected: {url}")
        except Exception as e:
            print(f"Failed to connect to Supabase: {e}")
            self.client = None
            
    def is_connected(self) -> bool:
        """Check if Supabase is available."""
        return self.client is not None
        
    def log_execution(
        self,
        prompt: str,
        entities: Dict[str, Any],
        steps: List[Dict[str, Any]],
        final_report: str,
        execution_time_ms: int,
        status: str = "completed"
    ) -> Optional[Dict]:
        """
        Log an agent execution to Supabase.
        
        Args:
            prompt: User's input prompt
            entities: Extracted entities from the report
            steps: Execution trace steps
            final_report: Generated RCA report
            execution_time_ms: Time taken in milliseconds
            status: 'completed' or 'error'
            
        Returns:
            Inserted record or None if logging failed
        """
        if not self.client:
            return None
            
        try:
            data = {
                "prompt": prompt[:5000],  # Limit prompt size
                "entities": entities,
                "steps": steps,
                "final_report": final_report[:50000] if final_report else None,
                "status": status,
                "execution_time_ms": execution_time_ms
            }
            
            result = self.client.table(self.table_name).insert(data).execute()
            print(f"Logged execution to Supabase: {result.data[0]['id'] if result.data else 'unknown'}")
            return result.data[0] if result.data else None
            
        except Exception as e:
            print(f"Failed to log to Supabase: {e}")
            return None
            
    def get_recent_executions(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent agent executions.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of execution records
        """
        if not self.client:
            return []
            
        try:
            result = self.client.table(self.table_name)\
                .select("*")\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            return result.data
        except Exception as e:
            print(f"Failed to fetch from Supabase: {e}")
            return []


# Global instance
supabase_manager = SupabaseManager()
