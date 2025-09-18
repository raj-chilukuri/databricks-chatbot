# app.py — Streamlit + Knowledge Graph (Databricks + OpenAI)
# Requirements:
#   streamlit
#   pandas
#   numpy
#   SQLAlchemy>=2.0.41,<3
#   databricks-sqlalchemy==2.0.8
#   databricks-sql-connector
#   openai>=1.0.0

import os, json, ast, time
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from openai import OpenAI

# -----------------------
# Config
# -----------------------
CATALOG = os.getenv("DATABRICKS_CATALOG", "main")
SCHEMA  = os.getenv("DATABRICKS_SCHEMA", "default")
NODES_T = os.getenv("KG_NODES_TABLE", "kg_nodes")
EDGES_T = os.getenv("KG_EDGES_TABLE", "kg_edges")
EMB_T   = os.getenv("KG_EMB_TABLE",   "kg_embeddings")

FULL_NODES = f"{CATALOG}.{SCHEMA}.{NODES_T}"
FULL_EDGES = f"{CATALOG}.{SCHEMA}.{EDGES_T}"
FULL_EMB   = f"{CATALOG}.{SCHEMA}.{EMB_T}"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -----------------------
# Databricks connection
# -----------------------
@st.cache_resource(show_spinner=False)
def dbx_engine():
    host = os.environ.get("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH")
    token = os.environ.get("DATABRICKS_PERSONAL_ACCESS_TOKEN")
    if not (host and http_path and token):
        raise RuntimeError("Missing Databricks env vars")
    url = f"databricks+connector://token:{token}@{host}?http_path={http_path}"
    return create_engine(url)

# -----------------------
# Load KG tables
# -----------------------
def _coerce_embedding(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype="float32")
    if isinstance(x, str):
        try:
            return np.array(json.loads(x), dtype="float32")
        except Exception:
            return np.array(ast.literal_eval(x), dtype="float32")
    return np.array([], dtype="float32")

@st.cache_data(show_spinner=True, ttl=600)
def load_kg(_refresh_key=0):
    eng = dbx_engine()
    nodes = pd.read_sql(text(f"SELECT * FROM {FULL_NODES}"), eng)
    edges = pd.read_sql(text(f"SELECT * FROM {FULL_EDGES}"), eng)
    emb   = pd.read_sql(text(f"SELECT * FROM {FULL_EMB}"),   eng)

    if "props_json" in nodes.columns:
        nodes["props_dict"] = nodes["props_json"].apply(lambda s: json.loads(s) if isinstance(s,str) and s else {})
    elif "props" in nodes.columns:
        nodes["props_dict"] = nodes["props"].apply(lambda v: v if isinstance(v,dict) else {})
    else:
        nodes["props_dict"] = [{} for _ in range(len(nodes))]

    emb["embedding"] = emb["embedding"].apply(_coerce_embedding)
    emb = emb[emb["embedding"].apply(lambda a: a.size>0)].reset_index(drop=True)

    joined = pd.merge(emb[["node_id","embedding","dim"]],
                      nodes[["node_id","table","label"]],
                      on="node_id", how="inner")
    X = np.vstack(joined["embedding"].to_list()).astype("float32")
    ids = joined["node_id"].tolist()
    return nodes, edges, joined, X, ids

# -----------------------
# Retrieval helpers
# -----------------------
def cosine_top_k(qv, X, k=8):
    qn = qv/ (np.linalg.norm(qv)+1e-12)
    Xn = X/ (np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
    sims = Xn @ qn
    idx = np.argsort(-sims)[:k]
    return idx.tolist()

def expand_neighborhood(seed_ids, edges, hop_k=1, per_hop=6):
    if hop_k<=0 or edges.empty: return seed_ids
    g_from = edges.groupby("src")["dst"].apply(list).to_dict()
    g_to   = edges.groupby("dst")["src"].apply(list).to_dict()
    seen, frontier = list(seed_ids), list(seed_ids)
    for _ in range(hop_k):
        nxt=[]
        for nid in frontier:
            neighbors=(g_from.get(nid,[])+g_to.get(nid,[]))[:per_hop]
            for m in neighbors:
                if m not in seen:
                    seen.append(m); nxt.append(m)
        frontier=nxt
        if not frontier: break
    return seen

def format_facts(df):
    lines=[]
    for _,r in df.iterrows():
        props = {k:v for k,v in r.get("props_dict",{}).items() if k not in ("__fulltext",)}
        lines.append(f"- [{r['table']}] {r['label']} — {props}")
    return "\n".join(lines)

# -----------------------
# LLM call
# -----------------------
def llm_answer(question, facts_block):
    client = OpenAI()
    prompt = f"""You are a helpful assistant. Use the following knowledge graph facts to answer clearly and cite tables in [brackets].

Question: {question}

Facts:
{facts_block}
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Graph-RAG with OpenAI", page_icon="🕸️", layout="wide")
st.title("🕸️ Graph-RAG on Databricks (OpenAI powered)")

with st.sidebar:
    k = st.slider("Top-K nodes", 3,20,8,1)
    hops = st.slider("Graph hops",0,2,1,1)
    per_hop = st.slider("Per-hop expansion",2,20,6,1)
    refresh = st.button("🔄 Refresh KG")

if refresh: st.session_state["_refresh_key"]=int(time.time())
refresh_key = st.session_state.get("_refresh_key",0)

with st.spinner("Loading KG..."):
    nodes, edges, joined, X, ids = load_kg(refresh_key)
st.success(f"{len(joined)} embeddings • {len(nodes)} nodes • {len(edges)} edges")

if "chat" not in st.session_state: st.session_state.chat=[]

for role, msg in st.session_state.chat:
    with st.chat_message(role): st.markdown(msg)

q = st.chat_input("Ask about vendors, departments, expenses…")
if q:
    st.session_state.chat.append(("user",q))
    with st.chat_message("user"): st.markdown(q)

    # Simple heuristic: vectorize with label overlap (or plug real MiniLM if available)
    qv = np.mean(X,axis=0)  # dummy center as fallback
    idxs = np.random.choice(len(ids), size=min(k,len(ids)), replace=False) if qv.size==0 else cosine_top_k(qv,X,k)

    seed=[ids[i] for i in idxs]
    expanded=expand_neighborhood(seed,edges,hops,per_hop)
    facts_df = nodes[nodes["node_id"].isin(expanded)]
    facts_block = format_facts(facts_df.head(50))

    answer = llm_answer(q,facts_block)
    st.session_state.chat.append(("assistant",answer))

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Graph facts"): st.code(facts_block,language="markdown")
