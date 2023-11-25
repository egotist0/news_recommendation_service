import swifter
import openai
from statistics import mean
from sentence_transformers import SentenceTransformer
import re
import pinecone
import pandas as pd
import json
import os
import warnings
from flask import Flask, render_template, request
import kornia
import evadb
import openai


app = Flask(__name__)

PINECONE_INDEX_NAME = "article-recommendation-service"
DATA_FILE = "articles.csv"


def initialize_pinecone():
    cursor = evadb.connect().cursor()
    warnings.filterwarnings("ignore")

    # Set api key
    api_key = ''
    os.environ["PINECONE_API_KEY"] = api_key

    openai.api_key = ""

    # Set environment
    environment = 'gcp-starter'
    os.environ["PINECONE_ENV"] = environment

    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    pinecone.init(api_key=api_key, environment=environment)


def delete_existing_pinecone_index():
    if PINECONE_INDEX_NAME in pinecone.list_indexes():
        pinecone.delete_index(PINECONE_INDEX_NAME)


def create_pinecone_index():
    pinecone.create_index(
        dimension=300, name=PINECONE_INDEX_NAME, metric="cosine", shards=1)
    pinecone_index = pinecone.Index(index_name=PINECONE_INDEX_NAME)

    return pinecone_index


def create_model():
    model = SentenceTransformer('average_word_embeddings_komninos')

    return model


def prepare_data(data):
    # rename id column and remove unnecessary columns
    data.rename(columns={"Unnamed: 0": "article_id"}, inplace=True)
    data.drop(columns=['date'], inplace=True)

    # extract only first few sentences of each article for quicker vector calculations
    data['content'] = data['content'].fillna('')
    data['content'] = data.content.swifter.apply(
        lambda x: ' '.join(re.split(r'(?<=[.:;])\s', x)[:4]))
    data['title_and_content'] = data['title'] + ' ' + data['content']

    # create a vector embedding based on title and article columns
    encoded_articles = model.encode(
        data['title_and_content'], show_progress_bar=True)
    data['article_vector'] = pd.Series(encoded_articles.tolist())

    return data


def upload_items(data):
    upsert_batch = []
    for i, row in data.iterrows():
        upsert_batch.append((str(row.id), row.article_vector))

        if len(upsert_batch) > 500:
            pinecone_index.upsert(upsert_batch)
            upsert_batch = []

    # Process any remaining data in upsert_batch
    if upsert_batch:
        pinecone_index.upsert(upsert_batch)


def process_file(filename):
    data = pd.read_csv(filename)
    data = prepare_data(data)
    upload_items(data)

    return data


def map_titles(data):
    return dict(zip(uploaded_data.id, uploaded_data.title))


def map_publications(data):
    return dict(zip(uploaded_data.id, uploaded_data.publication))


def map_content(data):
    return dict(zip(uploaded_data.id, uploaded_data.content))


def query_pinecone(reading_history_ids):
    reading_history_ids_list = list(map(int, reading_history_ids.split(',')))
    reading_history_articles = uploaded_data.loc[uploaded_data['id'].isin(
        reading_history_ids_list)]

    article_vectors = reading_history_articles['article_vector']
    reading_history_vector = [*map(mean, zip(*article_vectors))]

    query_results = pinecone_index.query(
        vector=[reading_history_vector], top_k=10)
    res = query_results['matches']

    results_list = []

    for idx, item in enumerate(res):
        results_list.append({
            "title": titles_mapped[int(item.id)],
            "publication": publications_mapped[int(item.id)],
            "score": item.score,
        })

    return json.dumps(results_list)


initialize_pinecone()
delete_existing_pinecone_index()
pinecone_index = create_pinecone_index()
model = create_model()
uploaded_data = process_file(filename=DATA_FILE)
titles_mapped = map_titles(uploaded_data)
publications_mapped = map_publications(uploaded_data)
content_mapped = map_content(uploaded_data)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        return query_pinecone(request.form.history)
    if request.method == "GET":
        return query_pinecone(request.args.get("history", ""))
    return "Only GET and POST methods are allowed for this endpoint"


if __name__ == '__main__':
    app.run()
