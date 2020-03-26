import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import rankdata
from sklearn.preprocessing import normalize
from collections import Counter
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


def lc_diversity(x, train_sentences, n_sample, raw_sent, raw_x, raw_y):
    represent_lc_vector = diversity(x, train_sentences) * least_confidence(raw_sent, raw_x, raw_y)
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


def entropy_diversity(x, train_sentences, n_sample, raw_sent, raw_x, raw_y):
    represent_entropy_vector = diversity(x, train_sentences) * entropy(raw_sent, raw_x, raw_y)
    ind = np.argpartition(-represent_entropy_vector, n_sample)[:n_sample]
    return ind


def margin_uncertainty(train_sentences, raw_x, raw_y):
    m = Model('RandomForest')
    m.fit(raw_x, raw_y)
    probs = m.proba(train_sentences)
    sorted_probs = np.sort(probs, axis=1)
    difference = sorted_probs[:, -1] - sorted_probs[:, -2]
    return 1 - difference


def margin_representative(x, train_sentences, n_sample, raw_sent, raw_x, raw_y):
    represent_margin_vector = representative(x) * margin_uncertainty(raw_sent, raw_x, raw_y)
    ind = np.argpartition(-represent_margin_vector, n_sample)[:n_sample]
    return ind


def margin_k_means(x, train_sentences, n_sample, raw_sent, raw_x, raw_y):
    represent_margin_vector = k_means_min_dist_to_cluster_center(x, n_sample) * margin_uncertainty(raw_sent, raw_x, raw_y)
    ind = np.argpartition(-represent_margin_vector, n_sample)[:n_sample]
    return ind


def information_k_means(x, train_sentences, n_sample, raw_sent, raw_x, raw_y):
    most_informative = np.argsort(margin_uncertainty(raw_sent, raw_x, raw_y))[:max(int((len(x) / 2)), (n_sample-1))]
    ind = k_means_n_closet_to_cluster_center(x[most_informative], n_sample).ravel()
    return ind


def k_means_division_representative(x, train_sentences, n_sample):
    k_means = k_means_cluster(x, n_sample)
    representative_vec = representative(x)
    ind = np.array([np.argsort(-representative_vec)[k_means == cluster][:1] for cluster in np.unique(k_means)])
    if len(ind) < n_sample:
        ind = np.vstack((ind, np.argsort(-representative_vec)[:1]))
    return np.concatenate(ind)


def k_means_division_diversity(x, train_sentences, n_sample):
    k_means = k_means_cluster(x, n_sample)
    diversity_vec = diversity(x, train_sentences)
    ind = np.array([np.argsort(-diversity_vec)[k_means == cluster][:1] for cluster in np.unique(k_means)])
    if len(ind) < n_sample:
        ind = np.vstack((ind, np.argsort(-diversity_vec)[:1]))
    return np.concatenate(ind)


def k_means_division_uncertainty(x, train_sentences, n_sample, raw_sent, raw_x, raw_y):
    k_means = k_means_cluster(x, n_sample)
    lc_vector = least_confidence(raw_sent, raw_x, raw_y)
    ind = np.array([np.argsort(-lc_vector)[k_means == cluster][:1] for cluster in np.unique(k_means)])
    if len(ind) < n_sample:
        ind = np.vstack((ind, np.argsort(-lc_vector)[:1]))
    return np.concatenate(ind)


def dbscan_division_uncertainty(x, train_sentences, n_sample, raw_sent, raw_x, raw_y):
    clusters = dbscan_cluster(x)
    lc_vector = least_confidence(raw_sent, raw_x, raw_y)
    sample_dict = clusters_counts(clusters, n_sample)
    ind = np.array([np.argsort(-lc_vector)[clusters == cluster][:n] for cluster, n in sample_dict.items()])
    if len(ind) < n_sample:
        ind = np.append((ind, np.argsort(-lc_vector)[:1]))
    return np.concatenate(ind)[:n_sample]


def dbscan_division_representative(x, train_sentences, n_sample):
    clusters = dbscan_cluster(x)
    representative_vec = representative(x)
    sample_dict = clusters_counts(clusters, n_sample)
    ind = np.array([np.argsort(-representative_vec)[clusters == cluster][:n] for cluster, n in sample_dict.items()])
    if len(ind) < n_sample:
        ind = np.append((ind, np.argsort(-representative_vec)[:1]))
    return np.concatenate(ind)[:n_sample]


def clusters_counts(clusters, n_sample):
    sample_dict = {k: max(int(v / len(clusters) * n_sample), 1) for k, v in Counter(clusters).items()}
    return sample_dict


def random_sample(x, train_sentences, n_samples):
    ind = np.random.choice(len(x), n_samples)
    return ind


def k_means_n_closet_to_cluster_center(x, n_sample):
    x = normalize(x, axis=0)
    k_means = MiniBatchKMeans(n_clusters=n_sample, batch_size=min(100, len(x)), random_state=1).fit(x)
    return np.argsort(k_means.transform(x)[:, :])[:1]


def k_means_min_dist_to_cluster_center(x, n_sample):
    x = normalize(x, axis=0)
    k_means = MiniBatchKMeans(n_clusters=n_sample, batch_size=min(100, len(x)), random_state=1).fit(x)
    return np.min(k_means.transform(x)[:, :], axis=1)


def k_means_cluster(x, n_sample):
    x = normalize(x, axis=0)
    k_means = MiniBatchKMeans(n_clusters=n_sample, batch_size=min(100, len(x)), random_state=1).fit(x)
    return k_means.predict(x)


def dbscan_cluster(x):
    clusters = DBSCAN(metric='cosine', leaf_size=min(len(x), 30))
    return clusters.fit_predict(x)


def random_sample_init(x, n_samples, random_init_sample):
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


EXPERIMENT_METHODS = [
    mdr, lc_representative, entropy_representative, margin_representative,
    information_k_means,
    dbscan_division_uncertainty, mdr, k_means_division_uncertainty, random_sample]

ADDITION_SAMPLE_ARGS = [lc_representative, lc_diversity, entropy_representative, entropy_diversity,
                        margin_representative, margin_k_means, information_k_means, k_means_division_uncertainty,
                        dbscan_division_uncertainty]