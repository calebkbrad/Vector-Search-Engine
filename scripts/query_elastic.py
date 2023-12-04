#!/usr/bin/python3

from elasticsearch import Elasticsearch
import clip
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    client = Elasticsearch(hosts="https://127.0.0.1:9200", api_key="RWZSa3o0c0JWSElhQlhhOEtXT3k6SVc1RmFzcHdTZi02d1pqbHJKd05LZw==", verify_certs=False)

    # API KEY for more: RWZSa3o0c0JWSElhQlhhOEtXT3k6SVc1RmFzcHdTZi02d1pqbHJKd05LZw==
    # API key for original: Vjl0OFNZc0JLek9pU2k4TzR2SGk6ajZWSHBaVHNTYzJYQnp5T0dwSEhpUQ==

    text = args[0]
    text_tokens = clip.tokenize([text], truncate=True)

    text_features = model.encode_text(text_tokens.to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_embeddings = text_features.cpu().detach().numpy().astype('float16')

    query_string = {
        "field": "text_embedding",
        "query_vector": text_embeddings[0],
        "k": 3,
        "num_candidates": 100
    }

    results = client.search(index="sci-fig_embeddings_more", knn=query_string)
    print(results)
    img_path = results["hits"]['hits'][0]['_source']['image_path']
    print(img_path)
    img = mpimg.imread(f'png/{img_path}')
    imgplot = plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])