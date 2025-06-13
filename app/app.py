# app.py
import sys
import os
import streamlit as st
from rag_contrats.core.retriever import RAGRetriever
from rag_contrats.core.generator import generate_answer


st.set_page_config(page_title=" Système RAG ", layout="wide")
st.title(" Démonstrateur RAG Contrats PDF 🇫🇷")

query = st.text_area(" Entrez votre question :", height=120)
top_k = st.slider(" Nombre de chunks à récupérer", 1, 10, 5)

if st.button("Lancer la recherche") and query:
    with st.spinner("Recherche et génération en cours..."):
        retriever = RAGRetriever()
        results = retriever.retrieve(query, top_k)

        chunks = [res[0] for res in results]
        scores = [res[1] for res in results]

        answer = generate_answer(results, query)

        st.markdown("###  Réponse générée")
        st.write(answer)

        st.markdown("###  Sources des chunks utilisés")
        for i, (chunk, score) in enumerate(results):
            st.markdown(f"**Chunk {i+1} — Document : `{chunk['doc_id']}` — Similarité : `{score:.4f}`**")
            st.write(chunk['text'])
            st.markdown("---")
