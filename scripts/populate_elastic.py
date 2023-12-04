#!/usr/bin/python3

from elasticsearch import Elasticsearch
import numpy as np
import pyarrow.parquet as pq
import json
import csv
from elasticsearch.helpers import parallel_bulk

ES_HOST = "https://127.0.0.1:9200/"
ES_USER = "elastic"
ES_PASSWORD = "P*lnwAV*aX0tvcC-aPwk"
ES_TIMEOUT = 3600

DEST_INDEX = "sci-fig_embeddings"
DELETE_EXISTING = True
CHUNK_SIZE = 100

def main():
    embs = np.load('all_embeddings/img_emb/img_emb_0.npy')
    metadata = pq.read_table('all_embeddings/metadata/metadata_0.parquet')
    metadata = metadata.to_pandas()
    image_paths = metadata.values.tolist()

    to_be_indexed = []
    for embedding_index in range(len(embs)):

        to_add = {}
        to_add['image_embedding'] = embs[embedding_index]
        to_add['image_path'] = image_paths[embedding_index]
        with open(f'labels/{embedding_index}.txt', 'r') as input:
            to_add['label'] = input.readline().rstrip()
        to_be_indexed.append(to_add)

    es = Elasticsearch(hosts=[ES_HOST], verify_certs=False, basic_auth=(ES_USER, ES_PASSWORD))

    with open('mappings.json', 'r') as config_file:
        config = json.loads(config_file.read())
        if DELETE_EXISTING:
            if es.indices.exists(index=DEST_INDEX):
                es.indices.delete(index=DEST_INDEX, ignore=[400,404])
        es.indices.create(index=DEST_INDEX, mappings=config["mappings"], settings=config['settings'], ignore=[400,404], request_timeout=ES_TIMEOUT)
    count = 0
    for success, info in parallel_bulk(client=es, actions=to_be_indexed, thread_count=4, chunk_size=CHUNK_SIZE, timeout='%ss' % 120, index=DEST_INDEX):
        if success:
            count += 1
        else:
            print('Doc failed, ', info)
    print("done!")



if __name__ == "__main__":
    main()