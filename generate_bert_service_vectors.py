
from bert_serving.client import BertClient


import pandas as pd
import os


def get_all_data_set_names(file_path):
    data_set_names = []
    for file in  os.listdir(file_path):
        if file.endswith(".txt"):
            data_set_names.append(file)

    return data_set_names


def generate_sentence_vectors_from_file(file_path, model, sample):
    df = pd.read_table(file_path, header=0, names=['sentence', 'Label'])

    if sample:
        df = df.smaple(1000)

    sentence_list = df.sentence.astype(str).tolist()
    sentence_embeddings = model.encode(sentence_list)
    df['embedded'] = sentence_embeddings.tolist()

    return df


def main_generate_sentence_vectors(file_path, sample=False):
    model = BertClient()
    data_sets = get_all_data_set_names(file_path)
    for data in data_sets:
        print(data)
        data_path = f'{file_path}/{data}'
        df_with_vectors = generate_sentence_vectors_from_file(data_path, model, sample)
        df_with_vectors.to_parquet(f'data_with_vectors/{data[:-4]+"_bert_avg"}.parquet')


if __name__ == '__main__':
    main_generate_sentence_vectors(file_path='/Users/uri/nlp_active_learning/data/sentiment labelled sentences')


# bert-serving-start -max_seq_len NONE -model_dir '/Users/uri/Documents/Aquant/Research/NLP/utils/bert_uncased_L-12_H-768_A-12' -num_worker 4

