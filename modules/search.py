import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def search_hadiths(query, dataframe, model, top_k=5):
    query_embedding = model.encode([query])[0]
    dataframe['similarity'] = dataframe['embeddings'].apply(lambda emb: cosine_similarity([query_embedding], [emb])[0,0])
    top_results = dataframe.sort_values(by='similarity', ascending=False).head(top_k)
    return top_results[['source', 'hadith_no', 'chapter', 'text_en', 'similarity']]
