import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import rankdata
from model import Model
import nmslib


def group_cosine_distance_mean(x, n_sample,*args):
    distance = 1 - cosine_similarity(x, x)
    mean_distance_over_group = np.mean(distance, axis=1)
    ind = np.argpartition(-mean_distance_over_group, n_sample)[:n_sample]
    return ind


def cosine_distance_mean(x, train_sentences, n_sample):
    distance = 1 - cosine_similarity(x, train_sentences)
    mean_distance_over_train_samples = np.mean(distance, axis=1)
    ind = np.argpartition(-mean_distance_over_train_samples, n_sample)[:n_sample]

    return ind


def group_cosine_distance_sum(x, n_sample):
    distance = 1 - cosine_similarity(x, x)
    sum_distance_over_group = np.sum(distance, axis=1)
    ind = np.argpartition(-sum_distance_over_group, n_sample)[:n_sample]
    return ind


def representative(x):
    mean_sim_vector = np.mean(cosine_similarity(x, x), axis=1)
    return mean_sim_vector


def diversity(x, train_sentences):
    mean_diverse_vector = np.mean(1 - cosine_similarity(x, train_sentences), axis=1)
    return mean_diverse_vector


def mdr(x, train_sentences, n_sample):
    mdr_vector = diversity(x, train_sentences) * representative(x)
    ind = np.argpartition(-mdr_vector, n_sample)[:n_sample]
    return ind


def rank_mdr(x, train_sentences, n_sample):
    mdr_vector = (rankdata(-diversity(x, train_sentences)) + rankdata(-representative(x))) / 2
    ind = np.argpartition(mdr_vector, n_sample)[:n_sample]
    return ind


def representative_max(x, train_sentences, n_sample):
    representative_vector = representative(x)
    ind = np.argpartition(-representative_vector, n_sample)[:n_sample]
    return ind


def diversity_max(x, train_sentences, n_sample):
    diversity_vector = diversity(x, train_sentences)
    ind = np.argpartition(-diversity_vector, n_sample)[:n_sample]
    return ind


def cosine_distance_sum(x, train_sentences, n_sample):
    distance = 1 - cosine_similarity(x, train_sentences)
    sum_distance_over_train_samples = np.sum(distance, axis=1)
    ind = np.argpartition(-sum_distance_over_train_samples, n_sample)[:n_sample]
    return ind


def least_confidence(train_sentences, raw_x, raw_y):
    m = Model('RandomForest')
    m.fit(raw_x, raw_y)
    probs = m.proba(train_sentences)
    return 1 - np.nanmax(probs, axis=1)


def lc_representative(x, train_sentences, n_sample, raw_sent, raw_x, raw_y):
    represent_lc_vector = representative(x) * \
                               least_confidence(raw_sent, raw_x, raw_y)
    ind = np.argpartition(-represent_lc_vector, n_sample)[:n_sample]
    return ind


def entropy(train_sentences, raw_x, raw_y):
    m = Model('RandomForest')
    m.fit(raw_x, raw_y)
    probs = m.proba(train_sentences)
    probs_without_zero = np.where(probs == 0, 10**-10, probs)
    log_probs = np.log(probs_without_zero)
    return -np.sum(probs * log_probs, axis=1)


def entropy_representative(x, train_sentences, n_sample, raw_sent, raw_x, raw_y):
    represent_entropy_vector = representative(x) * entropy(raw_sent, raw_x, raw_y)
    ind = np.argpartition(-represent_entropy_vector, n_sample)[:n_sample]
    return ind


def random_sample(x, train_sentences, n_samples):
    ind = np.random.choice(len(x), n_samples)
    return ind


def random_sample_init(x,n_samples,random_init_sample):
    ind = random_init_sample
    return ind


def random(x, n_samples):
    ind = np.random.choice(len(x), n_samples)
    return ind


def knn_representative(x):
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(x)
    index.createIndex({'post': 2}, print_progress=True)
    neighbours = index.knnQueryBatch(x, k=len(x), num_threads=4)
    return neighbours


