import re
from collections import Counter
import spacy
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
from torchnlp.word_to_vector import GloVe

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PAD = 0
UNK = 1


class ProcessDataset(Dataset):
    def __init__(self, df, word2idx=None, idx2word=None, max_vocab_size=50000):
        print('Processing Data')
        self._df = df
        print('Removing white space...')
        self._df.sentence = self._df.sentence
        self._nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
        if word2idx is None:
            print('Building Counter...')
            word_counter = self.build_counter()
            print('Building Vocab...')
            self._word2idx, self._idx2word = self.build_vocab(word_counter, max_vocab_size)
        else:
            self._word2idx, self._idx2word = word2idx, idx2word
        print('*' * 100)
        print('Dataset info:')
        print(f'Number of sent: {self._df.shape[0]}')
        print(f'Vocab Size: {len(self._word2idx)}')
        print('*' * 100)

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, idx):
        sent = self._df.sentence.iloc[idx]
        tokens = [w.text.lower() for w in self._nlp(self.process(sent))]
        vec = self.vectorize(tokens, self._word2idx)
        return vec, self._df.Label.iloc[idx]

    @staticmethod
    def process(text):
        text = re.sub(r'[\s]+', ' ', text)  # replace multiple white spaces with single space
        #         text = re.sub(r'@[A-Za-z0-9]+', ' ', text) # remove @ mentions
        text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # remove non alphanumeric character
        return text.strip()

    def build_counter(self):
        words_counter = Counter()
        for sent in tqdm(self._df.sentence.values):
            words_counter.update(w.text.lower() for w in self._nlp(self.process(sent)))
        return words_counter

    @staticmethod
    def build_vocab(words_counter, max_vocab_size):
        word2idx = {'<PAD>': PAD, '<UNK>': UNK}
        word2idx.update(
            {word: i + 2 for i, (word, count) in tqdm(enumerate(words_counter.most_common(max_vocab_size)))})
        idx2word = {idx: word for word, idx in tqdm(word2idx.items())}
        return word2idx, idx2word

    @staticmethod
    def vectorize(tokens, word2idx):
        vec = np.asarray([word2idx.get(token, UNK) for token in tokens], dtype=float)
        return vec


def pad_features(sent, seq_length=135):
    feature = sent
    if len(sent) < seq_length:
        feature = np.pad(sent, [seq_length - len(sent), 0], mode='constant')
    if len(sent) > seq_length:
        feature = sent[0:seq_length]
    return feature


def get_lstm_parsed_sentences_and_embeddings(data):
    processed_data = ProcessDataset(data, max_vocab_size=len(data))
    words_counter = processed_data.build_counter()
    vocab, index_to_vocab = processed_data.build_vocab(words_counter=words_counter, max_vocab_size=len(data))
    sentences = np.array([pad_features(processed_data.__getitem__(i)[0]) for i in range(len(data))])
    pretrained_embedding = GloVe(name='6B', dim=300, is_include=lambda w: w in vocab.keys())
    embedding_weights = torch.Tensor(len(vocab.keys()), pretrained_embedding.dim)
    for num, word in index_to_vocab.items():
        embedding_weights[num] = pretrained_embedding[index_to_vocab[num]]
    return sentences, embedding_weights