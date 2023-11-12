#importing libraries
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

#reading xlsx files using pandas
offer_retailer = pd.read_excel(r"offer_retailer.xlsx")
categories = pd.read_excel(r"categories.xlsx")
brand_category = pd.read_excel(r"brand_category.xlsx")

#merging offer_retailer and  brand_category on = "BRAND"
merged_df = pd.merge(brand_category,offer_retailer, on = "BRAND")
#filling na values with unknown values
merged_df['RETAILER'].fillna(value='Unknown', inplace=True)
#dropping Duplicate values
merged_df.drop_duplicates()

# lemmatizer and stemmer later we save this to pickle files for prediction
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

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

# creating another column with clean text function applied to the "OFFER" column
merged_df['cleaned_offer'] = merged_df['OFFER'].apply(clean_text)
# creating another column with clean text function applied to the "RETAILER" column
merged_df['cleaned_retailer'] = merged_df['RETAILER'].apply(clean_text)
# creating another column with clean text function applied to the "BRAND" column
merged_df['cleaned_brand'] = merged_df['BRAND'].apply(clean_text)
# creating another column with clean text function applied to the "BRAND_BELONGS_TO_CATEGORY" column
merged_df['cleaned_brand_category'] = merged_df['BRAND_BELONGS_TO_CATEGORY'].apply(clean_text)
# creating another column with all cleaned text
merged_df['cleaned_data'] = merged_df['cleaned_offer'] + ' ' + merged_df['cleaned_retailer'] + ' ' + merged_df['cleaned_offer'] + ' '+ merged_df['cleaned_brand_category']

#saving the cleaned data to final.csv file
merged_df.to_csv('final.csv',index=False)

# Create a TF-IDF vectorizer to later save this to pickle file
vectorizer = TfidfVectorizer()
# Fit and transform the processed text
tfidf_matrix = vectorizer.fit_transform(merged_df['cleaned_data'])

#saving tfidf.pkl for prediction using docker
with open('tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf_matrix, tfidf_file)
#saving vectorizer.pkl for prediction using docker
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
#saving lemmatizer.pkl for prediction using docker
with open('lemmatizer.pkl', 'wb') as lemmatizer_file:
    pickle.dump(lemmatizer, lemmatizer_file)
#saving stemmer.pkl for prediction using docker
with open('stemmer.pkl', 'wb') as stemmer_file:
    pickle.dump(stemmer, stemmer_file)

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
            
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)
        search_offers(user_input, merged_df, loaded_vectorizer, tfidf_matrix)