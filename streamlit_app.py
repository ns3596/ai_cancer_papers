import streamlit as st
import pandas as pd
from rank_bm25 import BM25Okapi
import torch
import json
import os
import numpy as np
from google.oauth2 import service_account
from google.cloud import bigquery
from google.cloud import bigquery_storage
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from sentence_transformers import CrossEncoder
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials, project=project_id)

dataset_name = st.secrets["bigquery"]["dataset_name"]
table_name = st.secrets["bigquery"]["table_name"]
reference_table_name = st.secrets["bigquery"]["reference_table_name"]
citation_table_name = st.secrets["bigquery"]["citation_table_name"]
cross_encoder  = st.secrets["bigquery"]["cross_encoder"]

client = bigquery.Client(credentials=credentials, project=credentials.project_id)
bq_storage_client = bigquery_storage.BigQueryReadClient()
st.write("Reached further of script!")
@st.cache_data
def load_papers():
    query = f"""
        SELECT *
        FROM `{client.project}.{dataset_name}.{table_name}`
        WHERE abstract IS NOT NULL and openalex_data_fetched = 'Yes' and language = 'en'
    """
    df = client.query(query).result().to_dataframe(bqstorage_client=bq_storage_client)
    return df

@st.cache_data
def load_citation_data():
    citation_query = f"""
        SELECT id, source_openalex_id, citation_openalex_id, title
        FROM `{client.project}.{dataset_name}.{citation_table_name}`
    """
    citation_df = client.query(citation_query).result().to_dataframe(bqstorage_client=bq_storage_client)
    return citation_df

@st.cache_data
def load_reference_data():
    reference_query = f"""
        SELECT id, source_openalex_id, referenced AS reference_openalex_id, title
        FROM `{client.project}.{dataset_name}.{reference_table_name}`
    """
    reference_df = client.query(reference_query).result().to_dataframe(bqstorage_client=bq_storage_client)
    return reference_df

df = load_papers()
citation_data = load_citation_data()
reference_data = load_reference_data()

if 'view' not in st.session_state:
    st.session_state['view'] = 'main'
if 'selected_paper_id' not in st.session_state:
    st.session_state['selected_paper_id'] = None

tokenized_corpus = [doc.split(" ") for doc in df['abstract']]
bm25 = BM25Okapi(tokenized_corpus)
@st.cache_resource
def load_cross_encoder():
    return CrossEncoder(cross_encoder)
cross_encoder_load = load_cross_encoder()

def bm25_with_crossencoder_ranking(query, top_n=10):
    query_tokens = query.split(" ")
    bm25_scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(bm25_scores)[::-1][:top_n * 5]  
    bm25_candidates = df.iloc[top_indices].copy()
    bm25_candidates['bm25_score'] = bm25_scores[top_indices]

    pairs = [(query, row['abstract']) for _, row in bm25_candidates.iterrows()]
    cross_encoder_scores = cross_encoder.predict(pairs)
    bm25_candidates['cross_encoder_score'] = cross_encoder_scores

    ranked_candidates = bm25_candidates.sort_values(by='cross_encoder_score', ascending=False).head(top_n)
    return ranked_candidates

def influential_ranking(query, top_n=10):
    bm25_candidates = bm25_with_crossencoder_ranking(query, top_n * 5)
    if "influential_score" not in bm25_candidates.columns:
        st.warning("No 'influential_score' column found.")
        return bm25_candidates.head(top_n)
    influential_threshold = bm25_candidates['influential_score'].quantile(0.7)
    filtered_candidates = bm25_candidates[bm25_candidates['influential_score'] > influential_threshold]
    return filtered_candidates.sort_values(by='influential_score', ascending=False).head(top_n)

def groundbreaking_ranking(query, top_n=10):
    bm25_candidates = bm25_with_crossencoder_ranking(query, top_n * 5)
    if "groundbreaking_recent_score" not in bm25_candidates.columns:
        st.warning("No 'groundbreaking_recent_score' column found.")
        return bm25_candidates.head(top_n)
    groundbreaking_threshold = bm25_candidates['groundbreaking_recent_score'].quantile(0.7)
    filtered_candidates = bm25_candidates[bm25_candidates['groundbreaking_recent_score'] > groundbreaking_threshold]
    return filtered_candidates.sort_values(by='groundbreaking_recent_score', ascending=False).head(top_n)

def personalized_ranking(query, user_weights, top_n=10):
    bm25_candidates = bm25_with_crossencoder_ranking(query, top_n * 5)
    total_weight = sum(user_weights.values())
    normalized_weights = {k: v / total_weight for k, v in user_weights.items()}

    for col in ["normalized_novelty_score", "citation_score", "social_media_score",
                "author_expertise_score", "source_score"]:
        if col not in bm25_candidates.columns:
            st.warning(f"Column '{col}' not found.")
            bm25_candidates[col] = 0 

    bm25_candidates['personalized_score'] = (
        normalized_weights.get('novelty', 0) * bm25_candidates['normalized_novelty_score'] +
        normalized_weights.get('citation', 0) * bm25_candidates['citation_score'] +
        normalized_weights.get('social', 0)   * bm25_candidates['social_media_score'] +
        normalized_weights.get('author', 0)   * bm25_candidates['author_expertise_score'] +
        normalized_weights.get('source', 0)   * bm25_candidates['source_score']
    )

    return bm25_candidates.sort_values(by='personalized_score', ascending=False).head(top_n)

def show_main_page():
    st.title("AI Cancer Paper Search Engine (Demo)")
    user_role = st.selectbox("Select your role:", ["Academic", "Researcher", "Student"])
    query = st.text_input("Enter your search query:", value=st.session_state.get('query', ''))
    ranking_method = st.selectbox("Choose a ranking method:", 
        ["Normal Ranking (BM25 & Cross-Encoder)", "Influential Ranking", "Groundbreaking Ranking", "Personalized Ranking"])
    st.session_state['query'] = query

    if st.button("Search"):
        st.session_state['view'] = 'results'
        st.session_state['search_query'] = query
        st.session_state['user_role'] = user_role
        st.session_state['ranking_method'] = ranking_method
        st.rerun()

def filter_by_role(df, user_role):
    if user_role == "Academic":
        min_citation_count = st.sidebar.slider("Minimum Citation Count", min_value=0, max_value=30, step=1, value=5)
        return df[df.get('citationCount', 0) >= min_citation_count]

    elif user_role == "Researcher":
        current_year = pd.to_datetime('today').year
        recent_years = st.sidebar.slider("Publication Year Range", min_value=2000, max_value=current_year, value=(current_year - 5, current_year))
        filtered_df = df[(df.get('year', 1900) >= recent_years[0]) & (df.get('year', 1900) <= recent_years[1])]
        if st.sidebar.checkbox("Show only open-access papers"):
            if 'isOpenAccess' in df.columns:
                filtered_df = filtered_df[filtered_df['isOpenAccess'] == True]
        return filtered_df

    elif user_role == "Student":
        max_influential_citations = st.sidebar.slider("Maximum Influential Citation Count", min_value=0, max_value=10, step=1, value=3)
        if 'influentialCitationCount' in df.columns:
            return df[df['influentialCitationCount'] <= max_influential_citations]
        else:
            st.warning("No 'influentialCitationCount' column found.")
            return df
    return df

def show_search_results():
    query = st.session_state.get('search_query', '')
    ranking_method = st.session_state.get('ranking_method', 'Normal Ranking (BM25 & Cross-Encoder)')
    user_role = st.session_state.get('user_role', 'Academic')

    filtered_df = filter_by_role(df, user_role)

    if ranking_method == "Normal Ranking (BM25 & Cross-Encoder)":
        papers = bm25_with_crossencoder_ranking(query)
    elif ranking_method == "Influential Ranking":
        papers = influential_ranking(query)
    elif ranking_method == "Groundbreaking Ranking":
        papers = groundbreaking_ranking(query)
    elif ranking_method == "Personalized Ranking":
        novelty_weight = st.sidebar.slider("Novelty Weight:", 0.0, 1.0, 0.5)
        citation_weight = st.sidebar.slider("Citation Weight:", 0.0, 1.0, 0.2)
        social_weight = st.sidebar.slider("Social Media Weight:", 0.0, 1.0, 0.1)
        author_weight = st.sidebar.slider("Author Expertise Weight:", 0.0, 1.0, 0.1)
        source_weight = st.sidebar.slider("Source Quality Weight:", 0.0, 1.0, 0.1)

        user_weights = {
            'novelty': novelty_weight,
            'citation': citation_weight,
            'social': social_weight,
            'author': author_weight,
            'source': source_weight
        }
        papers = personalized_ranking(query, user_weights)

    if st.button("Back to Search Bar"):
        st.session_state['view'] = 'main'
        st.session_state['selected_paper_id'] = None
        st.rerun()

    for i, row in papers.iterrows():
        st.markdown(f"<div style='border-bottom: 1px solid #ddd; padding: 15px;'>", unsafe_allow_html=True)
        title_link = row['title'] if 'title' in row else "Unknown Title"
        open_access_pdf = row.get('openAccessPdf', '#')
        st.markdown(f"<h3 style='margin: 0;'>{i+1}. <a href='{open_access_pdf}' target='_blank'>{title_link}</a></h3>", unsafe_allow_html=True)

        year = row.get('year', 'N/A')
        citation_count = row.get('citationCount', 'N/A')
        ref_count = row.get('referenceCount', 'N/A')
        paper_id = row.get('id', 'N/A')

        st.markdown(
            f"<p style='color: #666; font-size: 0.9em;'>"
            f"<strong>ID:</strong> {paper_id} | "
            f"<strong>Publication Year:</strong> {year} | "
            f"<strong>Citations:</strong> {citation_count} | "
            f"<strong>References:</strong> {ref_count}"
            "</p>",
            unsafe_allow_html=True,
        )
        if 'abstract' in row and isinstance(row['abstract'], str):
            summary = (row['abstract'][:200] + "...") if len(row['abstract']) > 200 else row['abstract']
        else:
            summary = "No Abstract"
        st.markdown(f"<p><strong>Abstract:</strong> {summary}</p>", unsafe_allow_html=True)

        if st.button(f"View Details for {title_link}", key=f"details_{paper_id}"):
            st.session_state['view'] = 'details'
            st.session_state['selected_paper_id'] = paper_id
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def show_paper_details(paper_id):
    paper = df[df['id'] == paper_id].iloc[0] if not df[df['id'] == paper_id].empty else None
    if paper is None:
        st.write("Paper not found.")
        return
    st.title(f"Paper Details - {paper.get('title', 'Unknown Title')}")
    st.write(f"Abstract: {paper.get('abstract', 'No abstract')}")
    st.write(f"Citations: {paper.get('citationCount', 'N/A')}")

    if st.button("Back to Search Results"):
        st.session_state['view'] = 'results'
        st.session_state['selected_paper_id'] = None
        st.rerun()

if st.session_state['view'] == 'main':
    show_main_page()
elif st.session_state['view'] == 'results':
    show_search_results()
elif st.session_state['view'] == 'details' and st.session_state['selected_paper_id'] is not None:
    show_paper_details(st.session_state['selected_paper_id'])
