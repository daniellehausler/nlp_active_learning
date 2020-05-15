from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score, precision_score
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class Model:
    """
    >>> m = Model('RandomForest')
    >>> m.evaluate(X_train, X_test, y_train, y_test)
    >>> m.get_scores()
    """

    def __init__(self, clf_name, **kwargs):
        # TODO : Option for other type of solvers (lstm etc)
        self._clf = {
            'RandomForest': self.sklearn_pipeline(RandomForestClassifier(random_state=1)),
            'SVC': self.sklearn_pipeline(LinearSVC(random_state=1)),
            'LogisticRegression': self.sklearn_pipeline(LogisticRegression(random_state=1)),
            'RF': RandomForestClassifier(random_state=1),
            'SVM': LinearSVC(random_state=1),
            'LR': LogisticRegression(random_state=1),
        }
        self._model = self._clf[clf_name]
        self._scores = {'accuracy': [],
                        'f1': [],
                        'mcc': [],
                        'recall': [],
                        'precision': [],
                        'n_samples': []}
        self._kwargs = kwargs

    @staticmethod
    def sklearn_pipeline(clf):
        sklearn_pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                ("classifier", clf),
            ]
        )
        return sklearn_pipeline

    def fit(self, train_sentences, y_train):
        self._model.fit(train_sentences, y_train)

    def predict(self, test_sentences):
        return self._model.predict(test_sentences)

    def proba(self, test_sentences):
        return self._model.predict_proba(test_sentences)

    def log_proba(self, test_sentences):
        return self._model.predict_log_proba(test_sentences)

    def accuracy(self, y_test, y_pred):
        self._scores['accuracy'].append(accuracy_score(y_test, y_pred))

    def f1(self, y_test, y_pred):
        self._scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))

    def mcc(self, y_test, y_pred):
        self._scores['mcc'].append(matthews_corrcoef(y_test, y_pred))

    def recall(self,  y_test, y_pred):
        self._scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))

    def precision(self, y_test, y_pred):
        self._scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))

    def count_samples(self, x_train):
        self._scores['n_samples'].append(len(x_train))

    def evaluate(self, train_sentences, test_sentences, y_train, y_test):
        self.fit(train_sentences, y_train)
        y_pred = self.predict(test_sentences)
        self.count_samples(train_sentences)
        self.scores_calc(y_test, y_pred)

    def scores_calc(self, y_test, y_pred):
        self.f1(y_test.reshape(len(y_test),), y_pred)
        self.accuracy(y_test, y_pred)
        self.mcc(y_test, y_pred)
        self.recall(y_test, y_pred)
        self.precision(y_test, y_pred)

    def get_scores(self):
        return self._scores