#!/usr/bin/python3

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import clip
import torch
import time

def main():
    metadata = pq.read_table('all_embeddings/metadata/metadata_0.parquet')
    metadata = metadata.to_pandas()
    image_paths = metadata.values.tolist()
    captions_data = pd.read_csv('SciFig/scientific_figures_large.csv')    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    start_time = time.time()

    all_embeddings = np.zeros(shape=(263952, 512), dtype=np.float16)

    print('beginning encoding')
    for index in range(len(image_paths)):
        if(index % 10000 == 0):
            print(index)
        correct_path = image_paths[index][0].split('png/')[1] 
        caption = captions_data[captions_data['sci_fig'] == correct_path].values[0][1]

        if type(caption) == str:
            with torch.no_grad():
                text_tokens = clip.tokenize(caption, truncate=True).to(device)
                text_embedding = model.encode_text(text_tokens).cpu().detach().to(torch.float16).numpy()[0]
                all_embeddings[index] = text_embedding

    end_time = time.time()
    print('done encoding!')
    time_elapsed = end_time - start_time
    with open('text_encoding_results.txt', 'w') as results:
        results.write(f'It took {time_elapsed} seconds to encode the text')
    np.save('text_embeddings', all_embeddings)

if __name__ == "__main__":
    main()