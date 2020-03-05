import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def group_cosine_distance_mean(x, n_sample):
    distance = 1 - cosine_similarity(x, x)
    mean_distance_over_group = np.mean(distance, axis=1)
    ind = np.argpartition(-mean_distance_over_group, n_sample * 2)[:n_sample * 2]
    return ind


def cosine_distance_mean(x, train_sentences, n_sample):

    if len(x) > 2 * n_sample:
        x_ind = group_cosine_distance_mean(x, n_sample)
        x = x[x_ind]

    distance = 1 - cosine_similarity(x, train_sentences)
    mean_distance_over_train_samples = np.mean(distance, axis=1)
    ind = np.argpartition(-mean_distance_over_train_samples, n_sample)[:n_sample]

    return ind


def group_cosine_distance_sum(x, n_sample):
    distance = 1 - cosine_similarity(x, x)
    sum_distance_over_group = np.sum(distance, axis=1)
    ind = np.argpartition(-sum_distance_over_group, n_sample * 2)[:n_sample * 2]
    return ind


def cosine_distance_sum(x, train_sentences, n_sample):
    if len(x) > 2 * n_sample:
        x_ind = group_cosine_distance_sum(x, n_sample)
        x = x[x_ind]

    distance = 1 - cosine_similarity(x, train_sentences)
    sum_distance_over_train_samples = np.sum(distance, axis=1)
    ind = np.argpartition(-sum_distance_over_train_samples, n_sample)[:n_sample]

    return ind