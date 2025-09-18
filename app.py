# app.py — Streamlit + Knowledge Graph (Databricks) + OpenAI
import os, json, ast, time
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from openai import OpenAI

# -----------------------
# Config (tables/catalog)
# -----------------------
CATALOG = os.getenv("DATABRICKS_CATALOG", st.secrets.get("DATABRICKS_CATALOG", "main"))
SCHEMA  = os.getenv("DATABRICKS_SCHEMA",  st.secrets.get("DATABRICKS_SCHEMA",  "default"))

NODES_T = os.getenv("KG_NODES_TABLE", st.secrets.get("KG_NODES_TABLE", "kg_nodes"))
EDGES_T = os.getenv("KG_EDGES_TABLE", st.secrets.get("KG_EDGES_TABLE", "kg_edges"))
EMB_T   = os.getenv("KG_EMB_TABLE",   st.secrets.get("KG_EMB_TABLE",   "kg_embeddings"))

FULL_NODES = f"{CATALOG}.{SCHEMA}.{NODES_T}"
FULL_EDGES = f"{CATALOG}.{SCHEMA}.{EDGES_T}"
FULL_EMB   = f"{CATALOG}.{SCHEMA}.{EMB_T}"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"))

# -----------------------
# Small helpers
# -----------------------
def _get_env_or_secret(key: str, default=None):
    if os.getenv(key): 
        return os.getenv(key)
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

@st.cache_resource(show_spinner=False)
def dbx_engine(host: str, http_path: str, token: str):
    """Create and cache a Databricks SQLAlchemy engine."""
    url = f"databricks+connector://token:{token}@{host}?http_path={http_path}"
    return create_engine(url)

def resolve_dbx_config() -> Tuple[str, str, str]:
    host = _get_env_or_secret("DATABRICKS_SERVER_HOSTNAME")
    http_path = _get_env_or_secret("DATABRICKS_HTTP_PATH")
    token = _get_env_or_secret("DATABRICKS_PERSONAL_ACCESS_TOKEN")
    return host, http_path, token

def _coerce_embedding_cell(x) -> np.ndarray:
    """Databricks ARRAY<FLOAT> may arrive as list or JSON-ish string; normalize to np.array(float32)."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype="float32")
    if isinstance(x, str):
        try:
            return np.array(json.loads(x), dtype="float32")
        except Exception:
            return np.array(ast.literal_eval(x), dtype="float32")
    return np.array([], dtype="float32")

@st.cache_data(show_spinner=True, ttl=600)
def load_kg(eng, _refresh_key: int = 0):
    """Load KG tables and prepare joined embedding matrix X and id list."""
    nodes = pd.read_sql(text(f"SELECT * FROM {FULL_NODES}"), eng)
    edges = pd.read_sql(text(f"SELECT * FROM {FULL_EDGES}"), eng)
    emb   = pd.read_sql(text(f"SELECT * FROM {FULL_EMB}"),   eng)

    # Normalize properties column
    if "props_json" in nodes.columns:
        nodes["props_dict"] = nodes["props_json"].apply(lambda s: json.loads(s) if isinstance(s,str) and s else {})
    elif "props" in nodes.columns:  # if stored as MAP
        nodes["props_dict"] = nodes["props"].apply(lambda v: v if isinstance(v, dict) else {})
    else:
        nodes["props_dict"] = [{} for _ in range(len(nodes))]

    # Normalize embeddings
    emb["embedding"] = emb["embedding"].apply(_coerce_embedding_cell)
    emb = emb[emb["embedding"].apply(lambda a: a.size > 0)].reset_index(drop=True)

    # Align embeddings with node metadata
    joined = pd.merge(
        emb[["node_id", "embedding", "dim"]],
        nodes[["node_id", "table", "label"]],
        on="node_id",
        how="inner",
    )
    if len(joined) == 0:
        X = np.zeros((0, 1), dtype="float32")
        ids = []
    else:
        X = np.vstack(joined["embedding"].to_list()).astype("float32")
        ids = joined["node_id"].tolist()

    return nodes, edges, joined, X, ids

def expand_neighborhood(seed_ids: List[str], edges_df: pd.DataFrame, hop_k: int = 1, per_hop: int = 6) -> List[str]:
    """Undirected-ish expansion using edges; limits per-hop fanout for readability."""
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
                    seen.append(m); nxt.append(m)
        frontier = nxt
        if not frontier: break
    return seen

def format_facts(df: pd.DataFrame, limit: int = 50) -> str:
    """Compact bullet list of facts for LLM prompt."""
    lines = []
    for _, r in df.head(limit).iterrows():
        props = {k: v for k, v in r.get("props_dict", {}).items() if k not in ("__fulltext",)}
        lines.append(f"- [{r.get('table','')}] {r.get('label', r.get('table',''))} — {props}")
    return "\n".join(lines)

def llm_answer(question: str, facts_block: str) -> str:
    """Call OpenAI chat model with facts augmented prompt."""
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

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Graph-RAG (Databricks + OpenAI)", page_icon="🕸️", layout="wide")
st.title("🕸️ Graph-RAG on Databricks (OpenAI)")

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
    st.header("Retrieval Settings")
    k = st.slider("Top-K nodes", 3, 30, 8, 1)
    hops = st.slider("Graph hops", 0, 2, 1, 1)
    per_hop = st.slider("Per-hop expansion", 2, 25, 6, 1)

    refresh = st.button("🔄 Connect / Refresh")

if refresh and not (host and http_path and token):
    st.error("Please provide all three values: Hostname, HTTP Path, and PAT.")
    st.stop()

# Connect once (cached)
try:
    eng = dbx_engine(host, http_path, token)
except Exception as e:
    st.error(f"Failed to create Databricks engine: {e}")
    st.stop()

# Load KG
if refresh:
    st.session_state["_refresh_key"] = int(time.time())
refresh_key = st.session_state.get("_refresh_key", 0)

with st.spinner("Loading knowledge graph tables from Databricks..."):
    nodes, edges, joined, X, ids = load_kg(eng, refresh_key)
st.success(f"Loaded {len(joined)} embeddings • {len(nodes)} nodes • {len(edges)} edges.")

# Chat history
if "chat" not in st.session_state: st.session_state.chat = []
for role, msg in st.session_state.chat:
    with st.chat_message(role): st.markdown(msg)

question = st.chat_input("Ask about vendors, departments, expenses…")
def node_rows_from_ids(all_nodes_df: pd.DataFrame, id_list: List[str]) -> pd.DataFrame:
    return all_nodes_df.merge(pd.DataFrame({"node_id": id_list}), on="node_id", how="inner")

if question:
    st.session_state.chat.append(("user", question))
    with st.chat_message("user"): st.markdown(question)

    # --- Retrieval ---
    # NOTE: Your KG embeddings are [node2vec | MiniLM]. We don't recompute the query embedding here.
    # Instead we use a lightweight heuristic over labels to pick seeds, then expand by edges.
    if len(joined) == 0:
        st.warning("No embeddings found in the KG tables.")
        st.stop()

    q = question.lower()
    # Simple label/props overlap signal
    label_sig = joined["label"].fillna("").str.lower().apply(lambda s: sum(w in s for w in q.split())).values
    # Add a tiny prior to break ties
    prior = np.linalg.norm(np.vstack(joined["embedding"].to_list()), axis=1)
    prior = (prior - prior.min()) / (prior.ptp() + 1e-9)
    score = 0.9 * label_sig + 0.1 * prior
    idxs = np.argsort(-score)[:min(k, len(score))].tolist()

    seed_ids = [ids[i] for i in idxs]
    expanded_ids = expand_neighborhood(seed_ids, edges, hop_k=hops, per_hop=per_hop)
    facts_df = node_rows_from_ids(nodes[["node_id","table","label","props_dict"]], expanded_ids)
    facts_block = format_facts(facts_df, limit=max(50, k * (1 + hops) * 6))

    # --- LLM answer (OpenAI) ---
    try:
        answer = llm_answer(question, facts_block)
    except Exception as e:
        answer = f"(OpenAI error: {e})\n\nHere are the retrieved facts:\n\n{facts_block}"

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Show retrieved graph facts"):
            st.code(facts_block, language="markdown")

    st.session_state.chat.append(("assistant", answer))
