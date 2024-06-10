import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Normalization
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)

    # Tokenize text
    tokens = word_tokenize(text)
    
    # Stop words setup
    base_stop_words = set(stopwords.words('english'))
    islamic_stopwords = [
        'Prophet', 'Muhammad', 'PBUH', 'Messenger', 'Allah', 'Apostle', 'Narrated', 'Reported',
        'Heard', 'Told', 'Peace', 'Be', 'Upon', 'Him', 'Verily', 'Indeed', 'Allahs', 'Say', 'Thy', 
        'Thee', 'Thou', 'Man', 'One', 'Came', 'Went', 'Day', 'Said', 'Asked', 'Saw'
    ]
    islamic_stopwords = set(word.lower() for word in islamic_stopwords)
    stop_words = base_stop_words.union(islamic_stopwords)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Reconstruct the cleaned text
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text
