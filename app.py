# app.py - FINAL WORKING VERSION for DEPLOYMENT

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent

# --- 1. SETUP ---
load_dotenv() # Not used by Streamlit Cloud secrets, but good practice

# --- 2. THE AI AGENT & DATABASE CONNECTION ---

@st.cache_resource(show_spinner="Initializing AI Agent...")
def setup_agent():
    # In Streamlit Cloud, these will be loaded from the Secrets manager
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    db_host = os.environ.get("DATABRICKS_SERVER_HOSTNAME")
    db_path = os.environ.get("DATABRICKS_HTTP_PATH")
    db_token = os.environ.get("DATABRICKS_TOKEN")
    db_catalog = os.environ.get("DATABRICKS_CATALOG")
    db_schema = os.environ.get("DATABRICKS_SCHEMA")
    
    # Check if secrets are loaded
    if not all([openai_api_key, db_host, db_path, db_token, db_catalog, db_schema]):
        raise ValueError("One or more required secrets are missing. Please check your Streamlit Cloud secrets configuration.")

    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

    db_uri = (
        f"databricks://token:{db_token}@{db_host}?"
        f"http_path={db_path}&"
        f"catalog={db_catalog}&"
        f"schema={db_schema}"
    )

    db = SQLDatabase.from_uri(db_uri, include_tables=[
        "workday_employees", "workday_departments", "sap_vendors", 
        "expensify_reports", "expensify_line_items"
    ])

    return create_sql_agent(llm=llm, db=db, agent_type="openai-tools")

# --- 3. THE STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Databricks Chatbot", layout="wide")
st.title("🤖 Chat with Your Databricks Data")
st.caption("Final Architecture: Hosted Agent connected to Databricks Cluster")

try:
    agent_executor = setup_agent()
    st.success("AI Agent is initialized and connected to Databricks. Ask away!")
except Exception as e:
    st.error(f"Could not initialize AI Agent. Please ensure your All-Purpose Cluster is running and your Streamlit secrets are correct. Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("The AI is thinking and querying..."):
            try:
                response = agent_executor.invoke({"input": prompt})
                answer = response.get("output", "Sorry, I couldn't find an answer.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})