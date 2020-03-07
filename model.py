from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd

df = pd.read_parquet('/Users/uri/nlp_active_learning/data_with_vectors/yelp_labelled.parquet')


class Model:

    def __init__(self):
        self.solver = 'sklearn' #TODO : Option for other type of solvers (lstm etc)


    def set_model(self,model_name):
        self.model= {'RandomForest' : RandomForestClassifier(),
                    'RBF_SVM' : SVC(gamma=2,C=1)}

        self.clf_algo = self.model[model_name]


    def sklearn_pipeline(self):
        sklearn_pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                ("classifier", self.clf_algo),
            ]
        )
        return sklearn_pipeline

    def evaluate(self,X_train,X_test,y_train,y_test,model_name):
        self.set_model(model_name)

        if self.solver == 'sklearn' :
            pipe = self.sklearn_pipeline()
            pipe.fit(X_train,y_train)
            self.y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test,self.y_pred)
            return acc



#Model().evaluate(X_train, X_test, y_train, y_test,'RandomForest')


