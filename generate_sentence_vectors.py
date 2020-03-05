from sentence_transformers import SentenceTransformer
import pandas as pd
import os


def get_all_data_set_names(file_path):
    data_set_names = os.listdir(file_path)
    return data_set_names


def generate_sentence_vectors_from_file(file_path, model, sample):
    df = pd.read_table(file_path, header=0, names=['sentence', 'Label'])

    if sample:
        df = df.smaple(1000)

    sentence_list = df.sentence.astype(str).tolist()
    sentence_embeddings = model.encode(sentence_list)
    df['embedded'] = sentence_embeddings

    return df


def main_generate_sentence_vectors(file_path, sample=False):
    model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
    data_sets = get_all_data_set_names(file_path)
    for data in data_sets:
        data_path = f'{file_path}/{data}'
        df_with_vectors = generate_sentence_vectors_from_file(data_path, model, sample)
        df_with_vectors.to_parquet(f'data_with_vectors/{data[:-4]}.parquet')


if __name__ == '__main__':
    main_generate_sentence_vectors(file_path='drive/My Drive/project_v0/data/sentiment labelled sentences')