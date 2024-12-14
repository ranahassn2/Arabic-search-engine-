import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize
import re
import base64

def get_image_as_base64(url):
    with open(url, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def add_custom_background(image_base64):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{image_base64}");
            background-size: cover;
            background-position: center center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

class ArabicTextSearchEngine:
    def __init__(self):
        self.stop_words = set(stopwords.words('arabic'))
        self.stemmer = ISRIStemmer()
        self.data = pd.read_csv('data.csv')
        self.setup_documents()

    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'[ى]', 'ي', text)
        text = re.sub(r'[ؤئ]', 'ء', text)
        return text

    def tokenize_text(self, text):
        return word_tokenize(text)

    def remove_stopwords_and_stem(self, words):
        return [self.stemmer.stem(word) for word in words if word not in self.stop_words and len(word) > 1]

    def process_text(self, text):
        text = self.normalize_text(text)
        words = self.tokenize_text(text)
        return ' '.join(self.remove_stopwords_and_stem(words))

    def setup_documents(self):
        self.doc_ids = self.data['docno'].tolist()
        self.titles = self.data['titles'].tolist()
        processed_docs = [self.process_text(doc) for doc in self.data['content']]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        st.write("Documents have been indexed and TF-IDF matrix is ready.")

    def search(self, query):
        processed_query = self.process_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        cos_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        sorted_results = sorted(zip(self.doc_ids, self.titles, cos_similarities), key=lambda x: x[2], reverse=True)[:5]
        st.write("Search Results:")
        for doc_id, title, score in sorted_results:
            st.write(f"Document ID: {doc_id}, Title: {title}, Similarity Score: {score:.3f}")
        return sorted_results

    def evaluate_search(self, predicted_docs, actual_docs):
        y_pred = [1 if doc in predicted_docs else 0 for doc in self.doc_ids]
        y_true = [1 if doc in actual_docs else 0 for doc in self.doc_ids]
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        st.write(f"Evaluation - Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1_score:.3f}")

st.title('Arabic Text Search Engine')

image_path = 'ali3.jpg'  
image_base64 = get_image_as_base64(image_path)
add_custom_background(image_base64)  

if 'engine' not in st.session_state:
    st.session_state.engine = ArabicTextSearchEngine()

query = st.text_area("Enter your search query here:")
if st.button('Search'):
    results = st.session_state.engine.search(query)
    actual_docs = [doc_id for doc_id, _, _ in results]
    predicted_docs = actual_docs
    st.session_state.engine.evaluate_search(predicted_docs, actual_docs)
