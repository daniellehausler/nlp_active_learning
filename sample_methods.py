import numpy as np
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from model import Model
import nmslib
from math import e


class BaseSampler:
    def __init__(self, unlabelled_vectors: np.array, labelled_vectors: np.array, n_samples: int):
        self._unlabelled_vectors = unlabelled_vectors
        self._labelled_vectors = labelled_vectors
        self._n_samples = n_samples


class UncertaintySampler(BaseSampler):
    def __init__(
            self,
            unlabelled_vectors: np.array,
            labelled_vectors: np.array,
            n_samples: int,
            unlabelled_sentences: np.array,
            labelled_sentences: np.array,
            labels: np.array,
            model_type: str,
    ):
        super(UncertaintySampler, self).__init__(unlabelled_vectors, labelled_vectors, n_samples)
        self._unlabelled_sentences = unlabelled_sentences
        self._labelled_sentences = labelled_sentences
        self._labels = labels
        self._model_type = model_type

    @staticmethod
    def least_confidence(model_type, unlabelled_sentences, labelled_sentences, labels):
        m = Model(model_type)
        m.fit(labelled_sentences, labels)
        if model_type in ['SVC', 'SVM']:
            dist_from_decision_boundary = m._model.decision_function(unlabelled_sentences)
            if len(dist_from_decision_boundary.shape) > 1:
                least_confidence = -np.nanmin(np.abs(dist_from_decision_boundary), axis=1)
            else:
                least_confidence = -np.abs(dist_from_decision_boundary)
        else:
            probs = m.proba(unlabelled_sentences)
            least_confidence = 1 - np.nanmax(probs, axis=1)
        return least_confidence

    @staticmethod
    def entropy(model_type, unlabelled_sentences, labelled_sentences, labels):
        m = Model(model_type)
        m.fit(labelled_sentences, labels)
        probs = m.proba(unlabelled_sentences)
        probs_without_zero = np.where(probs == 0, 10 ** -10, probs)
        log_probs = np.log(probs_without_zero)
        return -np.sum(probs * log_probs, axis=1)

    @staticmethod
    def margin_uncertainty(model_type, unlabelled_sentences, labelled_sentences, labels):
        m = Model(model_type)
        m.fit(labelled_sentences, labels)
        probs = m.proba(unlabelled_sentences)
        sorted_probs = np.sort(probs, axis=1)
        difference = sorted_probs[:, -1] - sorted_probs[:, -2]
        return 1 - difference

    def get_uncertainty_vector(self, method: staticmethod):
        return method(self._model_type, self._unlabelled_sentences, self._labelled_sentences, self._labels)

    def uncertainty_sample(self, method: staticmethod):
        uncertainty_vector = self.get_uncertainty_vector(method)
        return np.argpartition(-uncertainty_vector, self._n_samples)[: self._n_samples]


class RepresentativeSampler(BaseSampler):

    @staticmethod
    def representative(unlabelled_vectors, labelled_vectors, n_sample):
        mean_sim_vector = np.mean(cosine_similarity(unlabelled_vectors, unlabelled_vectors), axis=1)
        return mean_sim_vector

    @staticmethod
    def diversity(unlabelled_vectors, labelled_vectors, n_sample):
        mean_diverse_vector = np.mean(1 - cosine_similarity(unlabelled_vectors, labelled_vectors), axis=1)
        return mean_diverse_vector

    @staticmethod
    def knn_density(unlabelled_vectors, labelled_vectors, n_sample, k=20):
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(unlabelled_vectors)
        index.createIndex({'post': 2}, print_progress=True)
        distances = np.array(index.knnQueryBatch(unlabelled_vectors, k=k, num_threads=4))[:, 1]
        ds = np.sum(distances, axis=1) / k
        return ds

    @staticmethod
    def k_means_n_closet_to_cluster_center(unlabelled_vectors, n_sample):
        unlabelled_vectors = normalize(unlabelled_vectors, axis=0)
        k_means = MiniBatchKMeans(n_clusters=n_sample, batch_size=min(100, len(unlabelled_vectors)),
                                  random_state=1).fit(unlabelled_vectors)
        return np.argsort(np.min(k_means.transform(unlabelled_vectors), axis=1))[:n_sample]

    @staticmethod
    def k_means_most_distance_from_cluster(unlabelled_vectors, n_sample):
        unlabelled_vectors = normalize(unlabelled_vectors, axis=0)
        k_means = MiniBatchKMeans(n_clusters=2, batch_size=min(100, len(unlabelled_vectors)),
                                  random_state=1).fit(unlabelled_vectors)
        return np.argsort(np.max(k_means.transform(unlabelled_vectors), axis=1))[-n_sample:]

    def get_representative_vector(self, method: staticmethod):
        return method(self._unlabelled_vectors, self._labelled_vectors, self._n_samples)

    def representative_sample(self, method):
        representative_vector = self.get_representative_vector(method)
        return np.argpartition(-representative_vector, self._n_samples)[: self._n_samples]

    def representative_diversity_sample(self):
        representative_vector = self.get_representative_vector(self.representative)
        diversity_vector = self.get_representative_vector(self.diversity)
        result = representative_vector * diversity_vector
        return np.argpartition(-result, self._n_samples)[: self._n_samples]


class QBCSampler(BaseSampler):
    def __init__(
            self,
            unlabelled_vectors: np.array,
            labelled_vectors: np.array,
            n_samples: int,
            unlabelled_sentences: np.array,
            labelled_sentences: np.array,
            labels: np.array,
            model_type: str,
    ):
        super(QBCSampler, self).__init__(unlabelled_vectors, labelled_vectors, n_samples)
        self._unlabelled_sentences = unlabelled_sentences
        self._labelled_sentences = labelled_sentences
        self._labels = labels
        self._model_type = model_type

    @staticmethod
    def qbc_vote_entropy(model_type, unlabelled_sentences, labelled_sentences, labels):
        n_labels = len(np.unique(labels))
        model = Model(model_type)._model
        bagging_clf = BaggingClassifier(model)
        bagging_clf.fit(labelled_sentences, labels)
        predictions = np.array([estimator.predict(unlabelled_sentences) for estimator in bagging_clf.estimators_])
        counts = [np.array(np.unique(predict, return_counts=True)[1]) for predict in predictions.T]
        counts = np.array(
            [np.append(arr, np.zeros(n_labels - len(arr))) if len(arr) < n_labels else arr for arr in counts])
        voting_entropy = entropy(counts.T, base=e)
        return voting_entropy

    def get_qbc_vector(self, method: staticmethod):
        return method(self._model_type, self._unlabelled_sentences, self._labelled_sentences, self._labels)

    def qbc_sample(self, method: staticmethod):
        qbc_vector = self.get_qbc_vector(method)
        return np.argpartition(-qbc_vector, self._n_samples)[: self._n_samples]


class UncertaintyRepresentativeSampler(UncertaintySampler, RepresentativeSampler):
    def __init__(
            self,
            unlabelled_vectors,
            labelled_vectors,
            n_samples,
            unlabelled_sentences,
            labelled_sentences,
            labels,
            model_type,
    ):
        super(UncertaintyRepresentativeSampler, self).__init__(unlabelled_vectors, labelled_vectors, n_samples,
                                                               unlabelled_sentences, labelled_sentences,
                                                               labels, model_type)

    def uncertainty_representative_sample(self, uncertainty_method: staticmethod, representative_method: staticmethod):
        uncertainty_vector = self.get_uncertainty_vector(uncertainty_method)
        representative_vector = self.get_representative_vector(representative_method)
        result_vector = uncertainty_vector * representative_vector
        return np.argpartition(-result_vector, self._n_samples)[: self._n_samples]

    def uncertainty_mdr_sample(self, uncertainty_method: staticmethod, representative_method: staticmethod,
                               diversity_method: staticmethod):
        uncertainty_vector = self.get_uncertainty_vector(uncertainty_method)
        representative_vector = self.get_representative_vector(representative_method)
        diversity_vector = self.get_representative_vector(diversity_method)
        result_vector = uncertainty_vector * representative_vector * diversity_vector
        return np.argpartition(-result_vector, self._n_samples)[: self._n_samples]

    def uncertainty_k_means_sample(self, uncertainty_method: staticmethod, kmens_method: staticmethod):
        uncertainty_vector = self.get_uncertainty_vector(uncertainty_method)
        most_uncertain = np.argsort(-uncertainty_vector)[:min(self._n_samples * 5, len(self._unlabelled_sentences))]
        ind = kmens_method(self._unlabelled_vectors[most_uncertain], self._n_samples)
        return most_uncertain[ind]


class QBCRepresentativeSampler(QBCSampler, RepresentativeSampler):
    def __init__(
            self,
            unlabelled_vectors,
            labelled_vectors,
            n_samples,
            unlabelled_sentences,
            labelled_sentences,
            labels,
            model_type,
    ):
        super(QBCRepresentativeSampler, self).__init__(unlabelled_vectors, labelled_vectors, n_samples,
                                                       unlabelled_sentences, labelled_sentences,
                                                       labels, model_type)

    def qbc_representative_sample(self, qbc_method: staticmethod, representative_method: staticmethod):
        qbc_vector = self.get_qbc_vector(qbc_method)
        representative_vector = self.get_representative_vector(representative_method)
        result_vector = qbc_vector * representative_vector
        return np.argpartition(-result_vector, self._n_samples)[: self._n_samples]

    def qbc_dr_sample(self, qbc_method: staticmethod, representative_method: staticmethod,
                      diversity_method: staticmethod):
        qbc_vector = self.get_qbc_vector(qbc_method)
        representative_vector = self.get_representative_vector(representative_method)
        diversity_vector = self.get_representative_vector(diversity_method)
        result_vector = qbc_vector * representative_vector * diversity_vector
        return np.argpartition(-result_vector, self._n_samples)[: self._n_samples]

    def qbc_k_means_sample(self, qbc_method: staticmethod, kmens_method: staticmethod):
        qbc_vector = self.get_qbc_vector(qbc_method)
        most_uncertain = np.argsort(-qbc_vector)[:min(self._n_samples * 5, len(self._unlabelled_sentences))]
        ind = kmens_method(self._unlabelled_vectors[most_uncertain], self._n_samples)
        return most_uncertain[ind]


def least_confidence_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = UncertaintySampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.uncertainty_sample(sampler.least_confidence)


def entropy_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = UncertaintySampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.uncertainty_sample(sampler.entropy)


def margin_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = UncertaintySampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.uncertainty_sample(sampler.margin_uncertainty)


def diversity_sample(unlabelled_vectors, labelled_vectors, n_samples):
    sampler = RepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples)
    return sampler.representative_sample(sampler.diversity)


def representative_sample(unlabelled_vectors, labelled_vectors, n_samples):
    sampler = RepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples)
    return sampler.representative_sample(sampler.representative)


def mdr_sample(unlabelled_vectors, labelled_vectors, n_samples):
    sampler = RepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples)
    return sampler.representative_diversity_sample()


def least_confidence_representative_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = UncertaintyRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.uncertainty_representative_sample(sampler.least_confidence, sampler.representative)


def entropy_representative_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = UncertaintyRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.uncertainty_representative_sample(sampler.entropy, sampler.representative)


def margin_representative_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = UncertaintyRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.uncertainty_representative_sample(sampler.margin_uncertainty, sampler.representative)


def least_confidence_k_means_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = UncertaintyRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.uncertainty_k_means_sample(sampler.least_confidence, sampler.k_means_n_closet_to_cluster_center)


def lc_most_distance_2_means(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = UncertaintyRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.uncertainty_k_means_sample(sampler.least_confidence, sampler.k_means_most_distance_from_cluster)


def least_confidence_knn_density_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = UncertaintyRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.uncertainty_representative_sample(sampler.least_confidence, sampler.knn_density)


def least_confidence_mdr_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = UncertaintyRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.uncertainty_mdr_sample(sampler.least_confidence, sampler.representative, sampler.diversity)


def qbc_representative_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = QBCRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.qbc_representative_sample(sampler.qbc_vote_entropy, sampler.representative)


def qbc_k_means_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = QBCRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.qbc_k_means_sample(sampler.qbc_vote_entropy, sampler.k_means_n_closet_to_cluster_center)


def qbc_knn_density_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = QBCRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.qbc_representative_sample(sampler.qbc_vote_entropy, sampler.knn_density)


def qbc_knn_diverse_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = QBCRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.qbc_dr_sample(sampler.qbc_vote_entropy, sampler.knn_density, sampler.diversity)


def qbc_knn_mdr_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = QBCRepresentativeSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.qbc_dr_sample(sampler.qbc_vote_entropy, sampler.representative, sampler.diversity)


def qbc_sample(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args):
    sampler = QBCSampler(unlabelled_vectors, labelled_vectors, n_samples, **sampling_args)
    return sampler.qbc_sample(sampler.qbc_vote_entropy)


def random_sample(unlabelled_vectors, labelled_vectors, n_samples):
    ind = np.random.choice(len(unlabelled_vectors), n_samples)
    return ind


def random_sample_init(unlabelled_vectors, n_samples, random_init_sample):
    ind = random_init_sample
    return ind


UNCERTAINTY_SAMPLES = [
    least_confidence_sample,
    entropy_sample,
    margin_sample,
]

UNCERTAINTY_REPRESENTATIVE_SAMPLES = [
    least_confidence_representative_sample,
    entropy_representative_sample,
    margin_representative_sample,
    least_confidence_k_means_sample,
    lc_most_distance_2_means,
    least_confidence_knn_density_sample,
    least_confidence_mdr_sample

]

REPRESENTATIVE_SAMPLES = [
    mdr_sample,
    representative_sample,
    diversity_sample
]

QBC_SAMPLES = [
    qbc_representative_sample,
    qbc_knn_density_sample,
    qbc_k_means_sample,
    qbc_knn_mdr_sample,
    qbc_knn_diverse_sample,
    qbc_sample
]

EXPERIMENT_METHODS = [
    lc_most_distance_2_means,
    least_confidence_sample,
    qbc_knn_density_sample,
    qbc_sample,
    mdr_sample,
    least_confidence_mdr_sample,
    least_confidence_k_means_sample,
    random_sample
]

ADDITION_SAMPLE_ARGS = UNCERTAINTY_SAMPLES + UNCERTAINTY_REPRESENTATIVE_SAMPLES + QBC_SAMPLES
