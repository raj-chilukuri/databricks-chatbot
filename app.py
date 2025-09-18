# app.py — Streamlit + LangChain SQL Agent for Databricks (regenerated)
# - Works with Python 3.9–3.13
# - Uses SQLAlchemy 2.x and databricks-sqlalchemy 2.x
# - Fixes the "TypingOnly" AssertionError by requiring SQLAlchemy >= 2.0.31 via requirements.txt

import os
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# --------------------------
# App / Page Configuration
# --------------------------
st.set_page_config(page_title="Databricks SQL Chat", page_icon="🤖", layout="wide")
load_dotenv()  # allows local .env; Streamlit Cloud users can put these in st.secrets

st.title("🤖 Chat with Databricks (SQL Agent)")
st.caption("Natural-language questions → SQL → answers. Backed by Databricks + SQLAlchemy 2.x + LangChain.")

# --------------------------
# Helper: Build SQLAlchemy engine for Databricks
# --------------------------
def build_engine() -> Engine:
    """Create a SQLAlchemy engine for Databricks using env vars or Streamlit secrets."""
    token = os.getenv("DATABRICKS_TOKEN", st.secrets.get("DATABRICKS_TOKEN", ""))
    host = os.getenv("DATABRICKS_SERVER_HOSTNAME", st.secrets.get("DATABRICKS_SERVER_HOSTNAME", ""))
    http_path = os.getenv("DATABRICKS_HTTP_PATH", st.secrets.get("DATABRICKS_HTTP_PATH", ""))
    catalog = os.getenv("DATABRICKS_CATALOG", st.secrets.get("DATABRICKS_CATALOG", ""))
    schema = os.getenv("DATABRICKS_SCHEMA", st.secrets.get("DATABRICKS_SCHEMA", ""))

    missing = [k for k, v in {
        "DATABRICKS_TOKEN": token,
        "DATABRICKS_SERVER_HOSTNAME": host,
        "DATABRICKS_HTTP_PATH": http_path,
        "DATABRICKS_CATALOG": catalog,
        "DATABRICKS_SCHEMA": schema,
    }.items() if not v]

    if missing:
        raise RuntimeError(f"Missing required Databricks settings: {', '.join(missing)}")

    url = (
        f"databricks://token:{token}@{host}"
        f"?http_path={http_path}&catalog={catalog}&schema={schema}"
    )
    return create_engine(url)


# --------------------------
# Cached resources
# --------------------------
@st.cache_resource(show_spinner=False)
def get_resources(model_name: str, temperature: float, include_tables: List[str] | None):
    """Initialize LLM, DB, and Agent (cached across reruns)."""
    engine = build_engine()
    db = SQLDatabase(engine=engine, include_tables=include_tables)
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # Create agent that uses OpenAI tool-calling under the hood
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        verbose=False,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )
    return engine, db, agent


# --------------------------
# Sidebar: Connection & Model Settings
# --------------------------
with st.sidebar:
    st.header("Settings")
    st.markdown("**Connection** values are read from environment variables or `st.secrets`.\n"
                "Set: `DATABRICKS_TOKEN`, `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_HTTP_PATH`, "
                "`DATABRICKS_CATALOG`, `DATABRICKS_SCHEMA`.")
    default_tables = st.text_input(
        "Restrict to these tables (comma-separated, optional)",
        value="",
        help="Optional: Restrict schema context to these table names to improve accuracy and safety."
    )
    include_tables = [t.strip() for t in default_tables.split(",") if t.strip()] or None

    model = st.selectbox(
        "OpenAI model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    show_debug = st.checkbox("Show generated SQL & intermediate steps", value=True)


# --------------------------
# Initialize resources
# --------------------------
engine = None
db = None
agent = None
conn_ok = False
err_text = None

try:
    engine, db, agent = get_resources(model, temperature, include_tables)
    # simple connectivity check
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    conn_ok = True
except Exception as e:
    err_text = str(e)

status_col1, status_col2 = st.columns([1, 3])
with status_col1:
    if conn_ok:
        st.success("Connected to Databricks ✅")
    else:
        st.error("Not connected ❌")
with status_col2:
    if not conn_ok and err_text:
        with st.expander("Connection error details"):
            st.code(err_text)


# --------------------------
# Chat UI
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list[dict(role, content)]

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your Databricks data…")
if prompt and agent:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking, writing SQL, and querying Databricks…"):
            try:
                result: Dict[str, Any] = agent.invoke({"input": prompt})
                answer = result.get("output") or "I couldn't produce an answer."
                st.markdown(answer)

                if show_debug:
                    steps = result.get("intermediate_steps", [])
                    if steps:
                        with st.expander("See generated SQL & intermediate steps"):
                            for i, step in enumerate(steps, start=1):
                                tool = getattr(step, "tool", getattr(step, "action", None))
                                # `step` may be a tuple(tool, observation) in some versions
                                tool_input = getattr(step, "tool_input", None)
                                log = getattr(step, "log", None)
                                observation = getattr(step, "observation", None)
                                if isinstance(step, tuple) and len(step) == 2:
                                    tool, observation = step

                                st.markdown(f"**Step {i}: {tool}**")
                                if tool_input:
                                    st.code(str(tool_input))
                                if log:
                                    st.code(str(log))
                                if observation:
                                    st.markdown(f"_Observation:_\n\n{str(observation)}")
                st.session_state.history.append({"role": "assistant", "content": answer})
            except Exception as e:
                err = f"Error while running agent: {e}"
                st.error(err)
                st.session_state.history.append({"role": "assistant", "content": err})

# --------------------------
# Utility: Quick Schema Peek
# --------------------------
st.divider()
st.subheader("🔍 Quick Schema Peek")
if db:
    cols = st.columns(3)
    with cols[0]:
        if st.button("List tables"):
            try:
                tables = sorted(list(db.get_usable_table_names()))
                if not tables:
                    st.info("No tables found (or permissions restricted).")
                else:
                    st.write(tables)
            except Exception as e:
                st.error(f"Could not list tables: {e}")
    with cols[1]:
        table_name = st.text_input("Preview table (name)")
    with cols[2]:
        limit = st.number_input("Rows", min_value=1, max_value=1000, value=25, step=5)
    if table_name:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {int(limit)}")).fetchall()
            if rows:
                st.dataframe(rows)
            else:
                st.info("No rows returned.")
        except Exception as e:
            st.error(f"Preview failed: {e}")
