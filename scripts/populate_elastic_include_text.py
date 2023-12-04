#!/usr/bin/python3

from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json
from elasticsearch.helpers import parallel_bulk

ES_HOST = "https://127.0.0.1:9200/"
ES_USER = "elastic"
ES_PASSWORD = "P*lnwAV*aX0tvcC-aPwk"
ES_TIMEOUT = 3600

DEST_INDEX = "sci-fig_embeddings_more"
DELETE_EXISTING = True
CHUNK_SIZE = 100

def main():
    embs = np.load('all_embeddings/img_emb/img_emb_0.npy')
    text_embs = np.load('text_embeddings.npy')
    metadata = pq.read_table('all_embeddings/metadata/metadata_0.parquet')
    metadata = metadata.to_pandas()
    image_paths = metadata.values.tolist()
    captions_data = pd.read_csv('SciFig/scientific_figures_large.csv')

    to_be_indexed = []
    for embedding_index in range(len(embs)):
        image_path = image_paths[embedding_index][0].split('png/')[1]
        caption = captions_data[captions_data['sci_fig'] == image_path].values[0][1]
        print(caption)

        to_add = {}
        with open(f'labels/{embedding_index}.txt', 'r') as input:
            to_add['label'] = input.readline().rstrip()
        if type(caption) == str:
            if 'table' in caption.lower():
                to_add['label'] = 'tabular data'

        to_add['image_embedding'] = embs[embedding_index]
        to_add['text_embedding'] = text_embs[embedding_index]
        to_add['caption'] = caption
        to_add['image_path'] = image_path

        # print(caption)

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