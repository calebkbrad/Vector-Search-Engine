#!/usr/bin/python3

from elasticsearch import Elasticsearch
import numpy as np
import pyarrow.parquet as pq
import json
import csv
from elasticsearch.helpers import parallel_bulk
import pandas as pd

ES_HOST = "https://127.0.0.1:9200/"
ES_USER = "elastic"
ES_PASSWORD = "P*lnwAV*aX0tvcC-aPwk"
ES_TIMEOUT = 3600

DEST_INDEX = "sci-fig_embeddings"
DELETE_EXISTING = True
CHUNK_SIZE = 100

def main():
    embs = np.load('text_embeddings.npy')
    img_embs = np.load('all_embeddings/img_emb/img_emb_0.npy')
    # metadata = pq.read_table('all_embeddings/metadata/metadata_0.parquet')
    # metadata = metadata.to_pandas()
    # image_paths = metadata.values.tolist()

    # for path in image_paths[0:10]:
    #     print(path)
    # df = pd.read_csv('SciFig/scientific_figures_large.csv')
    # print(df)
    # my_row = df.loc[df['sci_fig'] == '1994.amta-1.11.pdf-Figure1.png']

    # print(my_row.values[0][1])
    print(embs)
    print(img_embs)





if __name__ == "__main__":
    main()