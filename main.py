import streamlit as st
import pandas as pd
import numpy as np
import pickle
from modules.preprocessing import preprocess_text
from modules.embeddings import load_model
from modules.search import search_hadiths
from time import time

# Load preprocessed hadiths data with embeddings
@st.cache_resource
def load_hadith_data():
    with open('data/hadiths_with_embeddings.pkl', 'rb') as file:
        df = pickle.load(file)
    return df

# Load model
@st.cache_resource
def load_embedding_model():
    return load_model()

# Streamlit app
def main():
    st.title("Hadith Semantic Search")

    # Load data and model
    df = load_hadith_data()
    model = load_embedding_model()
    # index = build_index(df)

    # Interface for searching hadiths
    st.header("Search Hadiths")
    query = st.text_input("Enter your search query:")
    top_k = st.number_input("Number of results to display", min_value=1, max_value=100, value=5)  # Default to 10 results

    if query:
        with st.spinner('Searching for Hadiths...'):
            
            # time
            start = time()
            results = search_hadiths(query, df, model, top_k)
            final = time() - start
            st.write(f"Search time: {final:.2f} seconds")
            st.write(f"Top {top_k} results for: '{query}'")
            for idx, row in results.iterrows():
                st.write(f"**Hadith No**: {row['hadith_no']}")
                st.write(f"**Source**: {row['source']}")
                st.write(f"**Chapter**: {row['chapter']}")
                st.write(f"**Text**: {row['text_en']}")
                st.write(f"**Similarity**: {row['similarity']:.4f}")
                st.write("---")

if __name__ == "__main__":
    main()
