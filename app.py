#!/usr/bin/python3

from flask import Flask, request, jsonify, render_template, send_from_directory
from elasticsearch import Elasticsearch
import clip
import torch
from PIL import Image

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
client = Elasticsearch(hosts="https://127.0.0.1:9200", api_key="RWZSa3o0c0JWSElhQlhhOEtXT3k6SVc1RmFzcHdTZi02d1pqbHJKd05LZw==", verify_certs=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        request_body = request.form
        print(request_body['types'])
        types = []
        if(request_body['types']):
            types = request_body['types'].split(',')
        print(types)
        if request_body['type'] == 'text':
            return search_text(request_body['query'], types)
        else:
            file = request.files['query']
            file.save('img_queries/query.png')
            return search_image(types)

    return render_template('index.html')
    
def search_text(text_query, types):
    with torch.no_grad():
        text_tokens = clip.tokenize(text_query, truncate=True)
        text_features = model.encode_text(text_tokens.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings = text_features.cpu().detach().numpy().astype('float16')
    
    query_string = {
        "field": "text_embedding",
        "query_vector": text_embeddings[0],
        "k": 10,
        "num_candidates": 1000
    }
    results = client.search(index="sci-fig_embeddings_more", knn=query_string)
    response = results["hits"]['hits']
    # print(response)
    filtered_results = []
    for hit in response:
        if types:
            if(hit['_source']['label'] not in types):
                continue
        filtered_result = {}
        filtered_result['image_path'] = hit["_source"]['image_path']
        filtered_result['caption'] = hit['_source']['caption']
        filtered_result['label'] = hit['_source']['label']
        filtered_result['similarity'] = hit['_score']
        filtered_results.append(filtered_result)
    return jsonify(filtered_results)

def search_image(types):
    with torch.no_grad():
        preprocessed_image = preprocess(Image.open('img_queries/query.png')).unsqueeze(0).to(device)
        image_features = model.encode_image(preprocessed_image.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_embeddings = image_features.cpu().detach().numpy().astype('float16')
    
    query_string = {
        "field": "image_embedding",
        "query_vector": image_embeddings[0],
        "k": 10,
        "num_candidates": 1000
    }
    results = client.search(index="sci-fig_embeddings_more", knn=query_string)
    response = results["hits"]['hits']
    # print(response)
    filtered_results = []
    for hit in response:
        if types:
            if(hit['_source']['label'] not in types):
                continue
        filtered_result = {}
        filtered_result['image_path'] = hit["_source"]['image_path']
        filtered_result['caption'] = hit['_source']['caption']
        filtered_result['label'] = hit['_source']['label']
        filtered_result['similarity'] = hit['_score']
        filtered_results.append(filtered_result)
    return jsonify(filtered_results)

@app.route('/search_text_post', methods=['POST'])
def search_text_post():
    request_body = request.form
    my_search = request_body['query']
    with torch.no_grad():
        text_tokens = clip.tokenize(my_search, truncate=True)
        text_features = model.encode_text(text_tokens.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings = text_features.cpu().detach().numpy().astype('float16')
    
    query_string = {
        "field": "text_embedding",
        "query_vector": text_embeddings[0],
        "k": 10,
        "num_candidates": 1000
    }
    results = client.search(index="sci-fig_embeddings_more", knn=query_string)
    response = results["hits"]['hits']
    # print(response)
    filtered_results = []
    for hit in response:
        filtered_result = {}
        filtered_result['image_path'] = hit["_source"]['image_path']
        filtered_result['caption'] = hit['_source']['caption']
        filtered_result['label'] = hit['_source']['label']
        filtered_result['similarity'] = hit['_score']
        filtered_results.append(filtered_result)
    return jsonify(filtered_results)

@app.route('/search_image', methods=['POST'])
def search_images_post():
    request_body = request.files
    my_search = request_body['query']
    my_search.save('img_queries/query.png')
    
    with torch.no_grad():
        preprocessed_image = preprocess(Image.open('img_queries/query.png')).unsqueeze(0).to(device)
        image_features = model.encode_image(preprocessed_image.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_embeddings = image_features.cpu().detach().numpy().astype('float16')
    
    query_string = {
        "field": "image_embedding",
        "query_vector": image_embeddings[0],
        "k": 10,
        "num_candidates": 1000
    }
    results = client.search(index="sci-fig_embeddings_more", knn=query_string)
    response = results["hits"]['hits']
    # print(response)
    filtered_results = []
    for hit in response:
        filtered_result = {}
        filtered_result['image_path'] = hit["_source"]['image_path']
        filtered_result['caption'] = hit['_source']['caption']
        filtered_result['label'] = hit['_source']['label']
        filtered_result['similarity'] = hit['_score']
        filtered_results.append(filtered_result)
    return jsonify(filtered_results)

@app.route('/png/<path:filename>')
def serve_image(filename):
    return send_from_directory('png', filename)

if __name__ == "__main__":
    app.run(debug=True)