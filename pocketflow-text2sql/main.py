import sys
import os
from pathlib import Path
from flow import create_text_to_sql_flow
from populate_db import populate_database

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_FILE = PROJECT_ROOT / "dhis2.sqlite"
SAMPLE_DB_FILE = PROJECT_ROOT / "ecommerce.db"

def run_text_to_sql(natural_query, db_path=DEFAULT_DB_FILE, max_debug_retries=3, force_sql_model=False):
    db_path = Path(db_path).expanduser().resolve()

    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        if db_path == SAMPLE_DB_FILE.resolve():
            print(f"Sample database at {db_path} missing or empty. Populating...")
            populate_database(str(db_path))
        else:
            raise FileNotFoundError(
                f"Database at {db_path} is missing or empty. "
                "Place dhis2.sqlite in the project root or pass a valid database path to run_text_to_sql()."
            )

    shared = {
        "db_path": str(db_path),
        "natural_query": natural_query,
        "max_debug_attempts": max_debug_retries,
        "debug_attempts": 0,
        "query_analysis": None,
        "final_result": None,
        "final_error": None,
        "analytics_payload": None,
        "analytics_context": None,
        "sql_trace": {},
        "force_sql_model": bool(force_sql_model),
    }

    print(f"\n=== Starting Text-to-SQL Workflow ===")
    print(f"Query: '{natural_query}'")
    print(f"Database: {db_path}")
    print(f"Max Debug Retries on SQL Error: {max_debug_retries}")
    print("=" * 45)

    flow = create_text_to_sql_flow()
    flow.run(shared) # Let errors inside the loop be handled by the flow logic

    # Check final state based on shared data
    if shared.get("final_error"):
            print("\n=== Workflow Completed with Error ===")
            print(f"Error: {shared['final_error']}")
    elif shared.get("final_result") is not None:
            print("\n=== Workflow Completed Successfully ===")
            if shared.get("analytics_payload"):
                print("Analytics payload prepared for downstream answer generation.")
            # Result already printed by ExecuteSQL node
    else:
            # Should not happen if flow logic is correct and covers all end states
            print("\n=== Workflow Completed (Unknown State) ===")

    print("=" * 36)
    return shared

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "How many organisation units are there at each hierarchy level?"

    run_text_to_sql(query) 
