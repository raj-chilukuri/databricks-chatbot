# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from databricks import sql

# --- 1. SETUP & DATABASE CONNECTION ---
load_dotenv() # This will be used for local testing, Streamlit Cloud uses its own secret management

@st.cache_resource
def get_db_connection():
    """Establishes a connection to the Databricks SQL Warehouse."""
    # Streamlit Cloud will get secrets from its own manager, not a .env file
    # For local testing, it will use your .env file
    hostname = os.environ.get("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH")
    access_token = os.environ.get("DATABRICKS_TOKEN")

    if not all([hostname, http_path, access_token]):
        st.error("Databricks credentials not found. Please configure your secrets in Streamlit Community Cloud.")
        st.stop()
        
    return sql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=access_token
    )

def run_ai_query(question: str) -> str:
    """Sends the user's question to the Databricks AI Function and returns the answer."""
    # Sanitize the user's question to prevent SQL injection issues
    sanitized_question = question.replace("'", "''")
    
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            # We use the catalog/schema names we verified work
            catalog = "workspace" 
            schema = "default"
            query = f"SELECT {catalog}.{schema}.ask_my_data('{sanitized_question}')"
            cursor.execute(query)
            result = cursor.fetchone()
            return result[0] if result else "No answer found."

# --- 2. THE STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Databricks Chatbot", layout="wide")
st.title("🤖 Chat with Your Databricks Data")
st.caption("Hosted on Streamlit Community Cloud | Backend on Databricks")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat logic
if prompt := st.chat_input("Ask about employees, expenses, or vendors..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Databricks AI is thinking and querying..."):
            try:
                answer = run_ai_query(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})