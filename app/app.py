"""Frontend Streamlit - Recherche de tickets support (RAG)."""

import streamlit as st
from rag_engine import search, get_filter_options


def score_circle(score, max_score=None, size=50):
    """Genere un cercle SVG de progression. 1.0 = cercle complet."""
    if max_score and max_score > 0:
        pct = min(score / max_score, 1.0)
    else:
        pct = min(score, 1.0)
    r = 18
    circ = 2 * 3.14159 * r
    filled = circ * pct
    color = "#4CAF50" if pct >= 0.7 else "#FF9800" if pct >= 0.4 else "#F44336"
    return f"""
    <div style="display:inline-flex;align-items:center;gap:8px;">
        <svg width="{size}" height="{size}" viewBox="0 0 44 44">
            <circle cx="22" cy="22" r="{r}" fill="none" stroke="#333" stroke-width="4"/>
            <circle cx="22" cy="22" r="{r}" fill="none" stroke="{color}" stroke-width="4"
                stroke-dasharray="{filled:.1f} {circ:.1f}"
                stroke-linecap="round" transform="rotate(-90 22 22)"/>
            <text x="22" y="22" text-anchor="middle" dominant-baseline="central"
                fill="white" font-size="10" font-weight="bold">{pct:.0%}</text>
        </svg>
    </div>
    """

# --- Config page ---
st.set_page_config(page_title="LogiStore - Recherche Tickets", layout="wide")
st.title("LogiStore - Recherche Tickets Support")

# --- Chargement des options de filtres (cache) ---
@st.cache_data(ttl=300)
def load_filters():
    return get_filter_options()

filters = load_filters()

# --- Sidebar : filtres ---
with st.sidebar:
    st.header("Filtres")

    method = st.selectbox("Methode de recherche", ["hybrid", "dense", "sparse"])

    type_filter = st.selectbox("Type", [""] + filters["types"])
    queue_filter = st.selectbox("Queue", [""] + filters["queues"])
    priority_filter = st.selectbox("Priority", [""] + filters["priorities"])
    tag_filter = st.selectbox("Tag", [""] + filters["tags"])

    limit = st.slider("Nombre de resultats", min_value=3, max_value=20, value=10)

# --- Barre de recherche ---
query = st.text_input("Rechercher un ticket", placeholder="Ex: printer not connecting, billing issue, security breach...")

# --- Resultats ---
if query:
    with st.spinner("Recherche en cours..."):
        results = search(
            query_text=query,
            method=method,
            limit=limit,
            type_=type_filter or None,
            queue=queue_filter or None,
            priority=priority_filter or None,
            tag=tag_filter or None,
        )

    max_score = results[0]["score"] if results else 1.0

    st.markdown(f"**{len(results)} resultats** pour *\"{query}\"*")
    st.divider()

    for i, r in enumerate(results, 1):
        title = r["subject"] or r["body"][:80] + "..."

        with st.container():
            col1, col2 = st.columns([0.93, 0.07])
            with col1:
                st.markdown(f"### {i}. {title}")
            with col2:
                st.markdown(score_circle(r["score"], max_score), unsafe_allow_html=True)

            # Metadata badges
            tags_str = " ".join(f"`{t}`" for t in r["tags"])
            st.markdown(
                f"**{r['type']}** | {r['queue']} | Priority: **{r['priority']}** | {tags_str}"
            )

            tab_body, tab_answer = st.tabs(["Ticket", "Reponse"])
            with tab_body:
                st.text(r["body"])
            with tab_answer:
                st.text(r["answer"])

            st.divider()
