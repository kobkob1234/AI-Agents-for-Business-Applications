import os
import sys
from src.utils.supabase_manager import supabase_manager

def verify_supabase():
    print("Verifying Supabase Connection...")
    if not supabase_manager.is_connected():
        print("❌ Supabase NOT connected.")
        sys.exit(1)
    
    print("✅ Supabase Manager initialized.")
    
    try:
        # Check table access
        print("Checking 'agent_executions' table access...")
        executions = supabase_manager.get_recent_executions(limit=1)
        print(f"✅ Access successful. Found {len(executions)} records.")
        
        # Verify schema columns by checking one record if it exists
        if executions:
            rec = executions[0]
            required = ["prompt", "entities", "steps", "final_report", "status", "execution_time_ms"]
            missing = [col for col in required if col not in rec]
            if missing:
                print(f"❌ Missing columns in fetch: {missing}")
            else:
                print("✅ Schema columns verified in record.")
                print(f"   Last execution ID: {rec.get('id')}")
        else:
             print("⚠️ Table is empty, cannot verify columns strictly, but query succeeded.")
             
    except Exception as e:
        print(f"❌ Failed to query Supabase: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_supabase()
