import streamlit as st
import pandas as pd
import pickle
from modules.preprocessing import preprocess_text
from modules.embeddings import load_model
from modules.search import search_hadiths
from modules.visualization import generate_word_cloud

# Load preprocessed hadiths data with embeddings
@st.cache(allow_output_mutation=True)
def load_hadith_data():
    with open('data/hadith_with_embeddings.pkl', 'rb') as file:
        df = pickle.load(file)
    return df

# Load model
@st.cache(allow_output_mutation=True)
def load_embedding_model():
    return load_model()

# Streamlit app
def main():
    st.title("Hadith Semantic Search")

    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose an option", ["Search Hadiths", "Generate Word Cloud"])

    df = load_hadith_data()
    model = load_embedding_model()

    if option == "Search Hadiths":
        st.header("Search Hadiths")
        query = st.text_input("Enter your search query:")
        if query:
            results = search_hadiths(query, df, model)
            st.write(f"Top {len(results)} results for: '{query}'")
            for index, row in results.iterrows():
                st.write(f"**Hadith No**: {row['hadith_no']}")
                st.write(f"**Source**: {row['source']}")
                st.write(f"**Chapter**: {row['chapter']}")
                st.write(f"**Text**: {row['text_en']}")
                st.write(f"**Similarity**: {row['similarity']:.4f}")
                st.write("---")

    elif option == "Generate Word Cloud":
        st.header("Generate Word Cloud")
        st.write("Generating word cloud from the hadith texts...")
        generate_word_cloud(df, 'cleaned_text')
        st.pyplot()

if __name__ == "__main__":
    main()
