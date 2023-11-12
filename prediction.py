#importing libraries
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

#reading the saved dataset which has cleaned and processed data
merged = pd.read_csv('final.csv')

#unpickling stemmer for user input processing and prediction
with open('stemmer.pkl', 'rb') as stemmer_file:
    stemmer = pickle.load(stemmer_file)  
#unpickling lemmatizer for user input processing and prediction
with open('lemmatizer.pkl', 'rb') as lemmatizer_file:
    lemmatizer = pickle.load(lemmatizer_file)
#unpickling tfidf matrix for user input processing and prediction
with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf_matrix = pickle.load(tfidf_file)
    
    
# Function to clean and preprocess text
def clean_text(text):
    #substutuing the '-' with " "
    text = re.sub(r'-', ' ', str(text))
    # Removing any special characters and digits from text
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    # Tokenize the text
    tokens = word_tokenize(text)
    # Lemmatize each word before lowering
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in ENGLISH_STOP_WORDS]
    # Stem each word 
    tokens = [stemmer.stem(word) for word in tokens]
    # Convert to lowercase and making them as string
    text = ' '.join(tokens).lower()
    #returning text
    return text

# Function to search for offers based on user input
def search_offers(user_input, df, vectorizer, tfidf_matrix):
    # Preprocess user input with same lemmatizer and stemmer
    processed_input = clean_text(user_input)
    # Transform user input using the same vectorizer
    input_vector = vectorizer.transform([processed_input])
    # Calculate cosine similarity between user input and offers
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    # Add similarity scores to the DataFrame
    df['SIMILARITY_SCORES'] = similarity_scores
    # Sort offers by similarity score
    result_df = df.sort_values(by='SIMILARITY_SCORES', ascending=False)
    print(result_df[['OFFER', 'RETAILER', 'BRAND','BRAND_BELONGS_TO_CATEGORY', 'SIMILARITY_SCORES']][:10])

if __name__ == "__main__":
    while True:
        user_input = input("Enter a search term ('q' to quit): ")
        if user_input.lower() == 'q':
            print("Exiting the program.")
            break
        try:
            int(user_input)
            print('enter a valid string')
            continue
        except ValueError:
            pass
        #unpickling vectorizer for user input processing and prediction    
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)
        search_offers(user_input, merged, loaded_vectorizer, tfidf_matrix)