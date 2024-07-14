import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline
from elasticsearch import Elasticsearch, exceptions as es_exceptions
import urllib3
import spacy

# Suppress only the single warning from urllib3 needed.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Streamlit caching for model loading
@st.cache_resource
def load_models():
    intent_classifier = joblib.load('intent_classifier.pkl')
    print(f"Type of intent_classifier: {type(intent_classifier)}")
    if hasattr(intent_classifier, 'steps'):
        print(f"Steps in pipeline: {intent_classifier.steps}")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return intent_classifier, summarizer

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_sci_sm")

# Load data function (no caching)
def load_data():
    return pd.read_csv('cleaned_medical_papers.csv')

# Load pre-trained models and data
article_data = load_data()
intent_classifier, summarizer = load_models()
nlp = load_spacy_model()

# Initialize Elasticsearch
@st.cache_resource
def init_elasticsearch():
    try:
        es = Elasticsearch(
            ['https://localhost:9200'],
            http_auth=('elastic', 'waheed'),
            verify_certs=False,  # Only use this for testing. For production, use a proper SSL certificate
            ssl_show_warn=False  # This replaces use_ssl=True
        )
        # Check if the cluster is available
        if not es.ping():
            raise es_exceptions.ConnectionError("Could not connect to Elasticsearch")
        cluster_info = es.info()
        st.success(f"Connected to Elasticsearch cluster: {cluster_info['cluster_name']}")
        st.info(f"Elasticsearch version: {cluster_info['version']['number']}")
        return es
    except es_exceptions.ConnectionError as e:
        st.error(f"Failed to connect to Elasticsearch. Error: {e}")
        return None
    except es_exceptions.AuthenticationException:
        st.error("Authentication failed. Please check your username and password.")
        return None
    except es_exceptions.NotFoundError:
        st.error("Elasticsearch cluster not found. Please check your connection details.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

es = init_elasticsearch()
if es is None:
    st.stop()

def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.lower()

def predict_intent(classifier, query):
    if hasattr(classifier, 'predict'):
        # If it's a scikit-learn pipeline or model
        return classifier.predict([query])[0]
    elif callable(classifier):
        # If it's a callable function or Hugging Face pipeline
        result = classifier(query)
        if isinstance(result, list):
            return result[0]['label']
        return result
    else:
        st.error(f"Unsupported intent classifier type: {type(classifier)}")
        return None

def extract_entities(query):
    doc = nlp(query)
    return [(ent.text, ent.label_) for ent in doc.ents]

def search_papers(es, query):
    try:
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "abstract"]
                }
            }
        }
        res = es.search(index='medical_papers', body=body)
        return res['hits']['hits']
    except es_exceptions.RequestError as e:
        st.error(f"Error searching papers: {e}")
        return []

def summarize_text(text):
    try:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return ""

def index_data(es, data):
    progress_bar = st.progress(0)
    total_rows = len(data)
    for index, row in data.iterrows():
        try:
            es.index(index='medical_papers', id=row['uuid'], body=row.to_dict())
            progress_bar.progress((index + 1) / total_rows)
        except es_exceptions.RequestError as e:
            st.warning(f"Error indexing document {row['uuid']}: {e}")

def ensure_index_exists(es, index_name):
    if not es.indices.exists(index=index_name):
        st.warning(f"Index '{index_name}' does not exist. Creating it now...")
        es.indices.create(index=index_name, body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "abstract": {"type": "text"},
                    "uuid": {"type": "keyword"}
                }
            }
        })
        st.success(f"Index '{index_name}' created successfully.")
        return False
    return True

# Streamlit app
def main():
    st.title("Medical Papers NLP Search")

    # Check if the index exists, create if it doesn't
    index_exists = ensure_index_exists(es, 'medical_papers')

    if not index_exists:
        # If the index was just created, we need to index the data
        with st.spinner("Indexing data... This may take a while."):
            index_data(es, article_data)
        st.success("Data indexed successfully.")

    query = st.text_input("Enter your query:")
    if query:
        st.write(f"Query: {query}")

        try:
            # Predict intent
            intent = predict_intent(intent_classifier, query)
            if intent is not None:
                st.write(f"Detected Intent: {intent}")
            else:
                st.warning("Could not detect intent")

            # Extract entities
            entities = extract_entities(query)
            st.write(f"Extracted Entities: {entities}")

            # Search papers
            search_results = search_papers(es, query)
            st.write("Search Results:")
            for result in search_results:
                st.write(result['_source']['title'])

            if search_results:
                top_abstract = search_results[0]['_source']['abstract']
                summary = summarize_text(top_abstract)
                st.write("Summary of Top Result:")
                st.write(summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()