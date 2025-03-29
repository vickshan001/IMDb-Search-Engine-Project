from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load IMDb dataset
imdb_data = pd.read_csv("/home/vickshan001/mysite/imdb_top_1000.csv")

# Concatenating relevant columns to create a document for each movie
imdb_data['document'] = imdb_data['Series_Title'] + " " + imdb_data['Overview'] + " " + imdb_data['Genre'] + " " + imdb_data['Director'] + " " + imdb_data['Star1'] + " " + imdb_data['Star2'] + " " + imdb_data['Star3'] + " " + imdb_data['Star4']

# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

# Preprocess IMDb dataset
imdb_data['preprocessed_document'] = imdb_data['document'].apply(preprocess_text)

# Create an index for faster retrieval
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(imdb_data['preprocessed_document'])

# Initialize BM25 model
tokenized_corpus = [doc.split(" ") for doc in imdb_data['preprocessed_document']]
bm25 = BM25Okapi(tokenized_corpus)

# Initialize TF-IDF model
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(imdb_data['preprocessed_document'])

# Function to retrieve top N movies based on BM25 using the preprocessed data
def retrieve_top_movies_bm25_preprocessed(query, top_n=5, relevance_threshold=0.1):
    tokenized_query = preprocess_text(query).split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Filter results based on the relevance threshold
    relevant_results = [(idx, score) for idx, score in enumerate(doc_scores) if score > relevance_threshold]
    
    if not relevant_results:
        return []

    # Sort by score and select top N
    relevant_results.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in relevant_results[:top_n]]
    
    return format_movie_results(imdb_data.iloc[top_indices])

# Function to retrieve top N movies based on TF-IDF cosine similarity using the preprocessed data
def retrieve_top_movies_tfidf_preprocessed(query, top_n=5, relevance_threshold=0.1):
    # Preprocess the query to lower case, remove stopwords, and stem
    preprocessed_query = preprocess_text(query)
    
    # Transform the query to get its vector representation
    query_vector = tfidf_vectorizer.transform([preprocessed_query])
    
    # Calculate cosine similarity between the query vector and all documents
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Filter results based on the relevance threshold
    relevant_results = [(idx, score) for idx, score in enumerate(cosine_similarities) if score > relevance_threshold]
    
    if not relevant_results:
        return []

    # Sort by score and select top N
    relevant_results.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in relevant_results[:top_n]]
    
    # Fetch the movies based on the indices
    top_movies = imdb_data.iloc[top_indices]
    
    # Format and return the results
    return format_movie_results(top_movies)

# Function to format movie results in the desired structure
def format_movie_results(movies_df):
    formatted_results = []
    for _, movie in movies_df.iterrows():
        # Convert NaN values to None
        meta_score = movie["Meta_score"]
        if pd.isna(meta_score):
            meta_score = None

        # Convert other NaN values to None
        certificate = movie["Certificate"]
        if pd.isna(certificate):
            certificate = None

        overview = movie["Overview"]
        if pd.isna(overview):
            overview = None

        # You can continue this pattern for other columns as needed

        formatted_movie = {
            "Certificate": certificate,
            "Director": movie["Director"],
            "Genre": movie["Genre"],
            "IMDB_Rating": movie["IMDB_Rating"],
            "Meta_score": meta_score,
            "No_of_Votes": movie["No_of_Votes"],
            "Overview": overview,
            "Poster_Link": movie["Poster_Link"],
            "Released_Year": movie["Released_Year"],
            "Runtime": movie["Runtime"],
            "Series_Title": movie["Series_Title"],
            "Stars": [movie["Star1"], movie["Star2"], movie["Star3"], movie["Star4"]],
            "document": movie["document"],
            "preprocessed_document": movie["preprocessed_document"]
        }
        formatted_results.append(formatted_movie)
    return formatted_results

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter "query" is required'}), 400

    # Assuming the query is already preprocessed if needed
    top_movies_bm25 = retrieve_top_movies_bm25_preprocessed(query)
    top_movies_tfidf = retrieve_top_movies_tfidf_preprocessed(query)

    # Combine and deduplicate results
    combined_results = {movie['Series_Title']: movie for movie in top_movies_bm25 + top_movies_tfidf}.values()

    if not combined_results:
        return jsonify({'message': 'No results found. Please try a different query.'}), 404

    return jsonify({'search_results': list(combined_results)})

if __name__ == '__main__':
    app.run()
