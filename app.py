# app.py — Streamlit + Knowledge Graph (Databricks) + OpenAI
import os, json, ast, time
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# =======================
# Config (tables/catalog)
# =======================
CATALOG = os.getenv("DATABRICKS_CATALOG", st.secrets.get("DATABRICKS_CATALOG", "main"))
SCHEMA  = os.getenv("DATABRICKS_SCHEMA",  st.secrets.get("DATABRICKS_SCHEMA",  "default"))
NODES_T = os.getenv("KG_NODES_TABLE",     st.secrets.get("KG_NODES_TABLE",     "kg_nodes"))
EDGES_T = os.getenv("KG_EDGES_TABLE",     st.secrets.get("KG_EDGES_TABLE",     "kg_edges"))
EMB_T   = os.getenv("KG_EMB_TABLE",       st.secrets.get("KG_EMB_TABLE",       "kg_embeddings"))

FULL_NODES = f"{CATALOG}.{SCHEMA}.{NODES_T}"
FULL_EDGES = f"{CATALOG}.{SCHEMA}.{EDGES_T}"
FULL_EMB   = f"{CATALOG}.{SCHEMA}.{EMB_T}"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"))

# ================
# Helper functions
# ================
def _get_env_or_secret(key: str, default=None):
    if os.getenv(key):
        return os.getenv(key)
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def resolve_dbx_config() -> Tuple[str, str, str]:
    host = _get_env_or_secret("DATABRICKS_SERVER_HOSTNAME")
    http_path = _get_env_or_secret("DATABRICKS_HTTP_PATH")
    token = _get_env_or_secret("DATABRICKS_PERSONAL_ACCESS_TOKEN")
    return host, http_path, token

def _coerce_embedding_cell(x) -> np.ndarray:
    """Normalize ARRAY<FLOAT> cell to np.array(float32)."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype="float32")
    if isinstance(x, str):
        try:
            return np.array(json.loads(x), dtype="float32")
        except Exception:
            return np.array(ast.literal_eval(x), dtype="float32")
    return np.array([], dtype="float32")

# ==========================================
# Connection: SQLAlchemy then DB-API fallback
# ==========================================
@st.cache_resource(show_spinner=False)
def dbx_conn(host: str, http_path: str, token: str):
    """
    Return ("sqlalchemy", engine) or ("dbapi", conn).
    Tries SQLAlchemy dialect first; falls back to DB-API.
    """
    # Try SQLAlchemy dialect
    try:
        import databricks.sqlalchemy as _  # registers the dialect
        from sqlalchemy import create_engine
        url = f"databricks+connector://token:{token}@{host}?http_path={http_path}"
        eng = create_engine(url)
        # light test
        with eng.connect() as _c:
            pass
        return ("sqlalchemy", eng)
    except Exception as e:
        st.warning(f"SQLAlchemy dialect not available or failed ({e}). Falling back to DB-API connector…")

    # Fallback: DB-API
    from databricks import sql as dbsql
    conn = dbsql.connect(server_hostname=host, http_path=http_path, access_token=token)
    return ("dbapi", conn)

def _read_sql_df(mode_handle, query: str) -> pd.DataFrame:
    mode, handle = mode_handle
    if mode == "sqlalchemy":
        from sqlalchemy import text
        return pd.read_sql(text(query), handle)
    else:
        return pd.read_sql(query, handle)

# =========================
# Fast, limited KG loading
# =========================
@st.cache_data(show_spinner=True, ttl=600)
def load_kg(mode_handle, max_rows: int, max_emb: int, _refresh_key: int = 0):
    # Only select needed columns + LIMITs
    nodes_sql = f"""
        SELECT node_id, table, label,
               CASE WHEN column_exists('{CATALOG}','{SCHEMA}','{NODES_T}','props_json')
                    THEN props_json ELSE NULL END AS props_json
        FROM {FULL_NODES}
        LIMIT {int(max_rows)}
    """
    edges_sql = f"""
        SELECT src, dst, rel, src_table, dst_table
        FROM {FULL_EDGES}
        LIMIT {int(max_rows)}
    """
    emb_sql = f"""
        SELECT node_id, embedding, dim
        FROM {FULL_EMB}
        LIMIT {int(max_emb)}
    """

    t0 = time.time()
    nodes = _read_sql_df(mode_handle, nodes_sql)
    edges = _read_sql_df(mode_handle, edges_sql)
    emb   = _read_sql_df(mode_handle, emb_sql)
    t1 = time.time()

    # Normalize properties
    if "props_json" in nodes.columns:
        nodes["props_dict"] = nodes["props_json"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    elif "props" in nodes.columns:
        nodes["props_dict"] = nodes["props"].apply(lambda v: v if isinstance(v, dict) else {})
    else:
        nodes["props_dict"] = [{} for _ in range(len(nodes))]

    # Normalize embeddings
    emb["embedding"] = emb["embedding"].apply(_coerce_embedding_cell)
    emb = emb[emb["embedding"].apply(lambda a: a.size > 0)].reset_index(drop=True)

    # Align embeddings with nodes
    if len(emb) == 0:
        joined = pd.DataFrame(columns=["node_id", "embedding", "dim", "table", "label"])
        X = np.zeros((0, 1), dtype="float32")
        ids = []
    else:
        joined = pd.merge(
            emb[["node_id", "embedding", "dim"]],
            nodes[["node_id", "table", "label"]],
            on="node_id",
            how="inner",
        )
        X = np.vstack(joined["embedding"].to_list()).astype("float32") if len(joined) else np.zeros((0, 1), dtype="float32")
        ids = joined["node_id"].tolist()

    load_secs = round(t1 - t0, 2)
    return nodes, edges, joined, X, ids, load_secs

# =====================
# Retrieval + LLM call
# =====================
def expand_neighborhood(seed_ids: List[str], edges_df: pd.DataFrame, hop_k: int = 1, per_hop: int = 6) -> List[str]:
    if hop_k <= 0 or edges_df.empty:
        return seed_ids
    g_from = edges_df.groupby("src")["dst"].apply(list).to_dict()
    g_to   = edges_df.groupby("dst")["src"].apply(list).to_dict()
    seen, frontier = list(seed_ids), list(seed_ids)
    for _ in range(hop_k):
        nxt = []
        for nid in frontier:
            neighbors = (g_from.get(nid, []) + g_to.get(nid, []))[:per_hop]
            for m in neighbors:
                if m not in seen:
                    seen.append(m)
                    nxt.append(m)
        frontier = nxt
        if not frontier:
            break
    return seen

def format_facts(df: pd.DataFrame, limit: int = 50) -> str:
    lines = []
    for _, r in df.head(limit).iterrows():
        props = {k: v for k, v in r.get("props_dict", {}).items() if k not in ("__fulltext",)}
        lines.append(f"- [{r.get('table','')}] {r.get('label', r.get('table',''))} — {props}")
    return "\n".join(lines)

def llm_answer(question: str, facts_block: str) -> str:
    client = OpenAI()
    prompt = f"""Use the knowledge-graph facts to answer succinctly. When referencing data, cite the table names in [brackets].

Question: {question}

Facts:
{facts_block}
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# =========
# UI / App
# =========
st.set_page_config(page_title="Graph-RAG (Databricks + OpenAI)", page_icon="🕸️", layout="wide")
st.title("🕸️ Graph-RAG on Databricks (OpenAI)")

# Sidebar: connection + load controls + retrieval settings
with st.sidebar:
    st.header("Databricks Connection")
    host, http_path, token = resolve_dbx_config()
    need_manual = not (host and http_path and token)
    manual = st.checkbox("Enter connection details manually", value=need_manual)

    if manual:
        host = st.text_input("Server Hostname", value=host or "", placeholder="adb-...azuredatabricks.net")
        http_path = st.text_input("HTTP Path", value=http_path or "", placeholder="/sql/1.0/warehouses/...")
        token = st.text_input("Personal Access Token", value=token or "", type="password")
        st.caption("Tip: save these in Streamlit Secrets for convenience.")

    st.divider()
    st.subheader("Load Controls")
    MAX_ROWS = st.number_input("Max rows per nodes/edges", 1000, 200_000, 20_000, step=1000)
    MAX_EMB  = st.number_input("Embeddings cap (rows)",   1000, 500_000, 50_000, step=1000)
    PING     = st.button("🏓 Ping warehouse")

    st.divider()
    st.subheader("Retrieval Settings")
    k      = st.slider("Top-K nodes", 3, 30, 8, 1)
    hops   = st.slider("Graph hops", 0, 2, 1, 1)
    perhop = st.slider("Per-hop expansion", 2, 25, 6, 1)

    refresh = st.button("🔄 Connect / Refresh")

if refresh and not (host and http_path and token):
    st.error("Please provide Hostname, HTTP Path, and PAT.")
    st.stop()

# Connect (cached) with fallback
try:
    conn_tuple = dbx_conn(host, http_path, token)
except Exception as e:
    st.error(f"Failed to connect to Databricks: {e}")
    st.stop()

# Optional ping
if PING:
    try:
        mode, h = conn_tuple
        if mode == "sqlalchemy":
            from sqlalchemy import text
            df_ping = pd.read_sql(text("SELECT 1 AS ok"), h)
        else:
            df_ping = pd.read_sql("SELECT 1 AS ok", h)
        st.success(f"Ping OK: {df_ping.iloc[0]['ok']}")
    except Exception as e:
        st.error(f"Ping failed: {e}")

# Load KG (limited) and show timing
if refresh:
    st.session_state["_refresh_key"] = int(time.time())
refresh_key = st.session_state.get("_refresh_key", 0)

with st.spinner("Loading knowledge graph tables…"):
    nodes, edges, joined, X, ids, secs = load_kg(conn_tuple, MAX_ROWS, MAX_EMB, refresh_key)
st.caption(f"Loaded in {secs}s (nodes≤{MAX_ROWS}, edges≤{MAX_ROWS}, embeddings≤{MAX_EMB})")
st.success(f"{len(joined)} embeddings • {len(nodes)} nodes • {len(edges)} edges loaded.")

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

# Main chat input
question = st.chat_input("Ask about vendors, departments, expenses…")

def node_rows_from_ids(all_nodes_df: pd.DataFrame, id_list: List[str]) -> pd.DataFrame:
    return all_nodes_df.merge(pd.DataFrame({"node_id": id_list}), on="node_id", how="inner")

if question:
    st.session_state.chat.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    if len(joined) == 0:
        st.warning("No embeddings found in the KG tables.")
        st.stop()

    # Seed selection: cheap label overlap (works without embedding the query)
    q = question.lower()
    labels = joined["label"].fillna("").str.lower().tolist()
    label_sig = np.array([sum(w in lbl for w in q.split()) for lbl in labels], dtype="float32")
    prior = np.linalg.norm(X, axis=1) if len(X) else np.zeros(len(labels), dtype="float32")
    if prior.ptp() > 0:
        prior = (prior - prior.min()) / (prior.ptp() + 1e-9)
    score = 0.9 * label_sig + 0.1 * prior
    idxs = np.argsort(-score)[:min(k, len(score))].tolist()

    seed_ids = [ids[i] for i in idxs]
    expanded_ids = expand_neighborhood(seed_ids, edges, hop_k=hops, per_hop=perhop)
    facts_df = node_rows_from_ids(nodes[["node_id", "table", "label", "props_dict"]], expanded_ids)
    facts_block = format_facts(facts_df, limit=max(50, k * (1 + hops) * 6))

    # LLM answer via OpenAI
    try:
        answer = llm_answer(question, facts_block)
    except Exception as e:
        answer = f"(OpenAI error: {e})\n\nHere are the retrieved facts:\n\n{facts_block}"

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Show retrieved graph facts"):
            st.code(facts_block, language="markdown")

    st.session_state.chat.append(("assistant", answer))
