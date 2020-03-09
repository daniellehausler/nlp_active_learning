from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


class Model:
    """
    >>> m = Model('RandomForest')
    >>> m.evaluate(X_train, X_test, y_train, y_test)
    >>> m.get_scores()
    """

    def __init__(self, clf_name):
        # TODO : Option for other type of solvers (lstm etc)
        self._clf = {
            'RandomForest': self.sklearn_pipeline(RandomForestClassifier()),
            'RBF_SVM': self.sklearn_pipeline(RandomForestClassifier(SVC(gamma=2, C=1))),
        }
        self._model = self._clf[clf_name]
        self._scores = {'accuracy': [],
                        'f1': [],'n_samples':[]}

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
        self._model.fit(train_sentences, y_train.ravel())

    def predict(self, test_sentences):
        return self._model.predict(test_sentences)

    def proba(self, test_sentences):
        return self._model.predict_proba(test_sentences)

    def log_proba(self, test_sentences):
        return self._model.predict_log_proba(test_sentences)

    def accuracy(self, y_test, y_pred):
        self._scores['accuracy'].append(accuracy_score(y_test, y_pred))

    def f1(self, y_test, y_pred):
        self._scores['f1'].append(f1_score(y_test, y_pred))

    def count_samples(self, x_train):
        self._scores['n_samples'].append(len(x_train))

    def evaluate(self, train_sentences, test_sentences, y_train, y_test):
        self.fit(train_sentences, y_train)
        y_pred = self.predict(test_sentences)
        self.count_samples(train_sentences)
        self.f1(y_test, y_pred)
        self.accuracy(y_test, y_pred)

    def get_scores(self):
        return self._scores

