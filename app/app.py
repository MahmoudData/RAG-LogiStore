"""Frontend Streamlit - Recherche de tickets support (RAG)."""

import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from rag_engine import search, get_filter_options

load_dotenv()


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


def synthesize(query, results, model="google/gemini-2.0-flash-001"):
    """Synthetise une reponse a partir des resultats les mieux classes."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "OPENROUTER_API_KEY manquante."

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "logistore"},
    )

    # Construction du contexte a partir des top resultats
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"--- Ticket {i} ---\n"
            f"Sujet: {r['subject']}\n"
            f"Type: {r['type']} | Queue: {r['queue']} | Priority: {r['priority']}\n"
            f"Contenu: {r['body'][:500]}\n"
            f"Reponse: {r['answer'][:500]}"
        )
    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Simply summarize the results of the support ticket search below. "
                    "Do not conduct any external searches. Do not make any assumptions. "
                    "Base your response solely on the information provided."
                ),
            },
            {
                "role": "user",
                "content": f"Question : {query}\n\nTickets retrouves :\n{context}",
            },
        ],
        max_tokens=600,
    )
    return response.choices[0].message.content


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

    method = st.selectbox("Methode de recherche", ["hybrid_rerank", "hybrid", "dense", "sparse"])

    type_filter = st.selectbox("Type", [""] + filters["types"])
    queue_filter = st.selectbox("Queue", [""] + filters["queues"])
    priority_filter = st.selectbox("Priority", [""] + filters["priorities"])
    tag_filter = st.selectbox("Tag", [""] + filters["tags"])

    limit = st.slider("Nombre de resultats", min_value=3, max_value=20, value=7)


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

    # --- Bouton synthese IA (sous la barre de recherche, avant les resultats) ---
    if results:
        if st.button("Generer une synthese IA"):
            with st.spinner("Synthese en cours..."):
                answer = synthesize(query, results[:3])
            st.info(answer)

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
