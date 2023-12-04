#!/usr/bin/python3

import os
import clip
import torch
import numpy as np
import pyarrow.parquet as pq

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    embs = np.load('all_embeddings/img_emb/img_emb_0.npy')
    metadata = pq.read_table('all_embeddings/metadata/metadata_0.parquet')
    metadata = metadata.to_pandas()

    labels = ['algorithm', 
              'architecture diagram', 
              'bar chart', 
              'box and whisker plot', 
              'confusion matrix', 
              'graph', 
              'line chart', 
              'maps', 
              'natural image', 
              'neural network',
              'NLP text grammar',
              'pareto',
              'pie chart',
              'scatter plot',
              'screenshot',
              'tree data structure',
              'tabular data',
              'venn diagram',
              'word cloud diagram'
              ]
    
    label_tokens = torch.cat([clip.tokenize(f'an image of a {c}') for c in labels]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(label_tokens)

    
    text_features /= text_features.norm(dim=-1, keepdim=True)
    for embeddings_index in range(len(embs)):
        new_tensor = torch.from_numpy(embs[embeddings_index])
        new_tensor = new_tensor.type(torch.float32)

        similarity = (100.0 * new_tensor @ text_features.T).softmax(dim=-1)

        values, indices = similarity.topk(1)
        with open(f'labels/{embeddings_index}.txt', 'w') as output:
            for value, index in zip(values, indices):
                output.write(labels[index])

if __name__ == "__main__":
    main()