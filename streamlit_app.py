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
import plotly.express as px
from streamlit.components.v1 import html as st_html

st.set_page_config(layout="wide")

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)
project_id = st.secrets["gcp_service_account"]["project_id"]
dataset_name = st.secrets["bigquery"]["dataset_name"]
table_name = st.secrets["bigquery"]["table_name"]
reference_table_name = st.secrets["bigquery"]["reference_table_name"]
citation_table_name = st.secrets["bigquery"]["citation_table_name"]
cross_encoder_model  = st.secrets["bigquery"]["cross_encoder"]

@st.cache_data
def load_papers():
    query = f"""
        SELECT id, title, abstract,combined_text, summary_v3, citationCount, influentialCitationCount, authors_list, referenceCount,
               fieldsOfStudy, safe_cast(safe_cast(year as float64) as int64) as year, isOpenAccess, source_type,
               publicationDate, authors, openAccessPdf,  openalex_id, round(influential_score,2) as influential_score, round(groundbreaking_recent_score,2) as groundbreaking_recent_score,
               citation_count, round(citation_score,2) as citation_score, round(normalized_novelty_score,2) as normalized_novelty_score, round(social_media_score,2) as social_media_score, counts_by_year
        FROM `{project_id}.{dataset_name}.{table_name}`
        WHERE abstract IS NOT NULL and openalex_data_fetched = 'Yes' and language = 'en'
    """
    df = client.query(query).result().to_dataframe()
    return df
df = load_papers()
@st.cache_data
def load_citation_data():
    citation_query = f"""
        SELECT *
        FROM `{project_id}.{dataset_name}.{citation_table_name}`
    """
    citation_df = client.query(citation_query).result().to_dataframe()
    return citation_df
df_citation = load_citation_data()
@st.cache_data
def load_reference_data():
    reference_query = f"""
        SELECT id, source_openalex_id, referenced AS reference_openalex_id, title
        FROM `{project_id}.{dataset_name}.{reference_table_name}`
    """
    reference_df = client.query(reference_query).result().to_dataframe()
    return reference_df

if 'view' not in st.session_state:
    st.session_state['view'] = 'main'
if 'selected_paper_id' not in st.session_state:
    st.session_state['selected_paper_id'] = None

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder(cross_encoder_model)
@st.cache_resource
def create_bm25(abstracts):
    tokenized = [doc.split(" ") for doc in abstracts]
    return BM25Okapi(tokenized)


def bm25_with_crossencoder_ranking(query, top_n=100):
    bm25 = create_bm25(filtered_df['title'])
    cross_encoder = load_cross_encoder()
    query_tokens = query.split(" ")
    bm25_scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(bm25_scores)[::-1][:top_n * 2]  
    bm25_candidates = df.iloc[top_indices].copy()
    bm25_candidates['bm25_score'] = bm25_scores[top_indices]

    pairs = [(query, row['title']) for _, row in bm25_candidates.iterrows()]
    cross_encoder_scores = cross_encoder.predict(pairs)
    bm25_candidates['cross_encoder_score'] = cross_encoder_scores

    ranked_candidates = bm25_candidates.sort_values(by='cross_encoder_score', ascending=False).head(top_n)
    return ranked_candidates


def influential_ranking(query, final_list=30,top_n=200):
    bm25_candidates = bm25_with_crossencoder_ranking(query, top_n)
    if "influential_score" not in bm25_candidates.columns:
        st.warning("No 'influential_score' column found.")
        return bm25_candidates.head(top_n)
    influential_threshold = bm25_candidates['influential_score'].quantile(0.7)
    filtered_candidates = bm25_candidates[bm25_candidates['influential_score'] > influential_threshold]
    return filtered_candidates.sort_values(by='influential_score', ascending=False).head(final_list)

def groundbreaking_ranking(query, final_list=30,top_n=200):
    bm25_candidates = bm25_with_crossencoder_ranking(query, top_n)
    if "groundbreaking_recent_score" not in bm25_candidates.columns:
        st.warning("No 'groundbreaking_recent_score' column found.")
        return bm25_candidates.head(top_n)
    groundbreaking_threshold = bm25_candidates['groundbreaking_recent_score'].quantile(0.7)
    filtered_candidates = bm25_candidates[bm25_candidates['groundbreaking_recent_score'] > groundbreaking_threshold]
    return filtered_candidates.sort_values(by='groundbreaking_recent_score', ascending=False).head(final_list)

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
    st.title("AI Cancer Paper Search Engine")
    user_role = st.selectbox("Select your role:", ["Anyone","Academic", "Researcher", "Student"])
    query = st.text_input("Enter your search query:", value=st.session_state.get('query', ''))
    ranking_method = st.selectbox("Choose a ranking method:", 
        ["Normal Ranking (BM25 & Cross-Encoder)", "Influential Ranking", "Groundbreaking Recent Ranking", "Personalised Ranking"])
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
            filtered_df = df[df['influentialCitationCount'] <= max_influential_citations]
            return filtered_df
        else:
            st.warning("No 'influentialCitationCount' column found.")
            return df
    elif user_role == "Anyone":
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
    elif ranking_method == "Groundbreaking Recent Ranking":
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
    for i, (index, row) in enumerate(papers.iterrows(), start=1):
        st.markdown(f"<div style='border-bottom: 1px solid #ddd; padding: 15px;'>", unsafe_allow_html=True)

        st.markdown(
            f"<h4 style='margin:0; font-size:20px;'>"
            f"{i}. <a href='{row['openAccessPdf']}' target='_blank'>{row['title']}</a>"
            f"</h4>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p style='color: #666; font-size: 0.9em;'>"
            f"<strong>Publication Year:</strong> {row['year']} | "
            f"<strong>Citations:</strong> {row['citationCount']} | "
            f"<strong>References:</strong> {row['referenceCount']}"
            "</p>",
            unsafe_allow_html=True,
        )


        col_left, col_right = st.columns([3, 1])

        with col_left:

            st.markdown(f"<p><strong>Summary:</strong> {row['summary_v3']}</p>", unsafe_allow_html=True)

        with col_right:

            inf_score = row.get('influential_score', 0)
            gnd_score = row.get('groundbreaking_recent_score', 0)
            nov_score = row.get('normalized_novelty_score', 0)

            table_html = f"""
            <table 
              style='
                border-collapse: separate; 
                border: 0px solid #bbb; 
                border-radius: 0px;
                margin-top: 10px; 
                font-size: 0.9em;
                overflow: hidden;
              '
            >
              <thead style='background-color: #292727;'>
                <tr>
                  <th style='padding: 8px 12px; text-align: center; border-bottom: 1px solid #ddd;'>Influential Score</th>
                  <th style='padding: 8px 12px; text-align: center; border-bottom: 1px solid #ddd;'>Groundbreaking Score</th>
                  <th style='padding: 8px 12px; text-align: center; border-bottom: 1px solid #ddd;'>Novelty Score</th>
                </tr>
              </thead style='background-color: #292727;'>
              <tbody>
                <tr>
                  <td style='padding: 8px 12px; text-align: center; border-bottom: 0px solid #ddd;'>{inf_score:.2f}</td>
                  <td style='padding: 8px 12px; text-align: center; border-bottom: 0px solid #ddd;'>{gnd_score:.2f}</td>
                  <td style='padding: 8px 12px; text-align: center; border-bottom: 0px solid #ddd;'>{nov_score:.2f}</td>
                </tr>
              </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)


        if st.button(f"View More Details", key=f"details_{row['id']}"):
            st.session_state['view'] = 'details'
            st.session_state['selected_paper_id'] = row['id']
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)    
def show_citation_map(df_citation, root_openalex_id, max_hop, min_citation_count):

    sub_df = df_citation[
        (df_citation['openalex_id'] == root_openalex_id) &
        (df_citation['hop'] <= max_hop) &
        (df_citation['citationCount'] >= min_citation_count)
    ]

    if sub_df.empty:
        st.info("No edges found for these settings.")
        return

    G = nx.DiGraph()
    for _, row in sub_df.iterrows():
        src = row['source_openalex_id']
        tgt = row['citation_openalex_id']
        G.add_node(src)
        G.add_node(tgt)
        G.add_edge(src, tgt, hop=row['hop'])

    if root_openalex_id not in G.nodes:
        st.warning("Root paper not in graph!")
        return

    dist_map = nx.single_source_shortest_path_length(G, root_openalex_id, cutoff=max_hop)
    sub_nodes = set(dist_map.keys())

    net = Network(height="600px", width="100%", directed=True)
    net.force_atlas_2based()

    distance_colors = {
        0: "#FF6666",
        1: "#FFA500",
        2: "#FFFF00",
        3: "#66FF66",
    }

    for node in sub_nodes:
        dist = dist_map[node]
        color = distance_colors.get(dist, "#cccccc")
        size = 20 if dist == 0 else 15
        net.add_node(str(node), label=str(node), color=color, size=size)

    for (source, target) in G.edges():
        if source in sub_nodes and target in sub_nodes:
            net.add_edge(str(source), str(target), color="#999999")

    html_str = net.generate_html()
    st_html(html_str, height=600, scrolling=True)
    
def show_paper_details(paper_id):
    paper_df = df[df['id'] == paper_id]
    if paper_df.empty:
        st.write("Paper not found.")
        return

    paper = paper_df.iloc[0]

    st.title(f"Paper Details - {paper.get('title', 'Unknown Title')}")


    st.subheader("Key Scores")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Novelty Score", paper.get("normalized_novelty_score", "N/A"))
    with col2:
        st.metric("Groundbreaking Score", paper.get("groundbreaking_recent_score", "N/A"))
    with col3:
        st.metric("Citation Score", paper.get("citation_score", "N/A"))
    with col4:
        st.metric("Influential Score", paper.get("influential_score", "N/A"))


    st.subheader("Totals")
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Total Citations", paper.get("citation_count", "N/A"))
    with col6:
        st.metric("Total Social Media Score", paper.get("social_media_score", "N/A"))


    st.subheader("Abstract")
    abstract_text = paper.get("abstract", "No abstract available.")
    st.write(abstract_text)

    proto_list = paper.get('counts_by_year', [])

    counts_by_year_data = []
    for item in proto_list:
        year = item["year"]
        cited_by_count = item["cited_by_count"]
        counts_by_year_data.append({
            "year": year,
            "cited_by_count": cited_by_count
        })

    if counts_by_year_data:
        st.subheader("Cumulative Citations by Year")
        cby_df = pd.DataFrame(counts_by_year_data).sort_values("year")
        cby_df["year"] = cby_df["year"].astype(int)
        cby_df["cumulative_citations"] = cby_df["cited_by_count"].cumsum()
        cby_df["cumulative_citations"] = cby_df["cumulative_citations"]

        fig = px.line(
            cby_df,
            x="year",
            y="cumulative_citations",
            title="Cumulative Citations Over Time",
            labels={"year": "Year", "cumulative_citations": "Cumulative Citations"}
        )

        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                dtick=1
            )
        )

        fig.update_traces(
            hovertemplate="Year: %{x}<br>Cumulative Citations: %{y:.2f}"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No year-by-year citation data available.")
    st.subheader("Citation Network")
    root_openalex_id = paper.get('openalex_id', None)

    if not root_openalex_id:
        st.warning("No OpenAlex ID found for this paper. Can't build citation map.")
    else:

        max_hop = st.slider("Maximum Hop Distance", 1, 5, 3)
        min_citations = st.slider("Minimum Citation Count to include", 0, 1000, 0)

        if st.button("Generate Citation Map"):

            show_citation_map(
                df_citation=df_citation,
                root_openalex_id=root_openalex_id,
                max_hop=max_hop,
                min_citation_count=min_citations
            )

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
    

