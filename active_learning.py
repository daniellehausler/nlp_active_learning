from typing import Callable
import numpy as np
from sample_methods import *


class ActiveLearner:
    """
    use example:
    >>> learner = ActiveLearner(initialization_method=group_cosine_distance_mean, n_samples=100)
    >>> sampled_index = learner.add_n_new_samples(x, y , sample_method=cosine_distance_mean)
    """

    def __init__(self, n_samples, initialization_method):
        self._initialization_method = initialization_method
        self._n_samples = n_samples
        self._train_sentences = None
        self._train_labels = None
        self._raw_sentences = None

    def initialize_learner(self, x: np.array, y: np.array, raw_sentences,random_init_sample, n_sample: int) -> np.array:
        """
        :param raw_sentences: array of strings
        :param n_sample: number of samples for initialization the learner
        :param x: embedded sentences
        :param y: labels
        """
        ind = self._initialization_method(x, n_sample,random_init_sample)
        self._train_sentences = x[ind]
        self._train_labels = y[ind]
        self._raw_sentences = raw_sentences[ind]
        return ind

    def add_n_new_samples(
            self,
            sample_method: Callable,
            x: np.array,
            y: np.array,
            raw_sentences: np.array,
            random_init_sample :np.array,
            **sampling_args
    ):
        """
        :param raw_sentences: array of strings
        :param sample_method: strategy of sampling new sentences from x
        :param sampling_args: additional args that the sample method should get
        :param x: embedded sentences
        :param y: labels
        """
        assert len(x) == len(y), "len of sentences doesn't match len of labels"
        assert len(x) >= self._n_samples, "there are not enough samples to add"
        if self._train_sentences is None:
            ind = self.initialize_learner(x, y, raw_sentences,random_init_sample, n_sample=int(self._n_samples))
        else:
            ind = sample_method(x, self._train_sentences, int(self._n_samples), **sampling_args)
            self._train_sentences = np.vstack((self._train_sentences, x[ind]))
            self._train_labels = np.vstack((self._train_labels, y[ind]))
            self._raw_sentences = np.vstack((self._raw_sentences, raw_sentences[ind]))
        return ind

    def get_raw_train_sent(self):
        return self._raw_sentences

    def get_y_train(self):
        return self._train_labels
