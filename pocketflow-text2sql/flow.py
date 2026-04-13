from pocketflow import Flow, Node
from nodes import AnalyzeQuestion, GetSchema, GenerateSQL, ExecuteSQL, BuildAnalytics, DebugSQL, Finish

def create_text_to_sql_flow():
    """Creates the text-to-SQL workflow with a debug loop."""
    analyze_question_node = AnalyzeQuestion()
    get_schema_node = GetSchema()
    generate_sql_node = GenerateSQL()
    execute_sql_node = ExecuteSQL()
    build_analytics_node = BuildAnalytics()
    debug_sql_node = DebugSQL()
    finish_node = Finish()

    # Define the main flow sequence using the default transition operator
    analyze_question_node >> get_schema_node >> generate_sql_node >> execute_sql_node

    # --- Define the debug loop connections using the correct operator ---
    # If ExecuteSQL returns "error_retry", go to DebugSQL
    execute_sql_node - "error_retry" >> debug_sql_node
    execute_sql_node - "success" >> build_analytics_node
    execute_sql_node - "done" >> finish_node
    build_analytics_node >> finish_node

    # If DebugSQL returns "default", go back to ExecuteSQL
    # debug_sql_node - "default" >> execute_sql_node # Explicitly for "default"
    # OR using the shorthand for default:
    debug_sql_node >> execute_sql_node

    # Create the flow
    text_to_sql_flow = Flow(start=analyze_question_node)
    return text_to_sql_flow
