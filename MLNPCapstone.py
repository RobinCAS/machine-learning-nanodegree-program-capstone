from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, precision_score
from sys import stdout
from time import clock
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

class MLNPCapstone(object):

    # Construct a new MLNPCapstone object    
    def __init__(self, 
                 target_name,
                 positive_label,
                 negative_label,
                 missing_data_label='NaN',
                 pca_max_n=2):

        self._target_name = target_name                # column name of target values
        self._learner = None                           # the name of the algorithm
        self._clf = None                               # sklearn classifier object
        self._missing_data_label = missing_data_label  # ?, NaN, etc.
        self._use_pca = None                           # should we use PCA?
        self._df = None                                # pandas dataframe with the data
        self._train_data = (None, None)                # training data (X, y)
        self._test_data = (None, None)                 # testing data (X, y)
        self._positive_label = positive_label          # equals 1
        self._negative_label = negative_label          # equals 0
        self._pca_max_n = pca_max_n                    # max number of features to keep
        self._pca_n = None                             # exact num of features for PCA
        self._pca = None                               # Principal Component Analysis
        self._first_classifier = True                  # helps with print formatting

    # Setters
    def set_learner(self, learner): self._learner = learner
    def set_use_pca(self, use_pca): self._use_pca = use_pca
    def set_pca_n(self, n): self._pca_n = n

    # Load the data into a Pandas dataframe and do any necessary preprocessing on it,
    # including splitting the data into training and test sets
    def load_and_prepare_data(self, datafile):

        print " + Reading in the data...",; stdout.flush()
        self._df = pd.read_csv(datafile)
        print "done.\n + Preprocessing the data...",; stdout.flush()
        self.preprocess_data()
        print "done.\n + Splitting data into training and test sets...",; stdout.flush()
        self.split_dataframe()

    # Use PCA to analyze the data set - knowing the number of principal components
    # can help us choose a model
    def analyze_data(self):
        
        print "done.\n + Using PCA to analyze the data...",; stdout.flush()

        columns = self._df.columns.tolist()
        columns = [c for c in columns if c not in [self._target_name]]
        
        (X_train, _) = self._train_data
        if not self._pca:
            self._pca = RandomizedPCA(n_components=self._pca_max_n, whiten=True)
            self._pca.fit(X_train)

        # NOTE:  code for plot stolen from sklearn example: http://bit.ly/1X8ZsUw
        fig = plt.figure(1, figsize=(4,3))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(self._pca.explained_variance_ratio_)
        fig.suptitle('RandomizedPCA Analysis')
        plt.axis('tight')
        plt.xlabel('Component')
        plt.ylabel('Explained Variance Ratio')
        plt.show()

        # Reset the PCA object, since we will need to set the exact number
        # of components we want to use if and when we use it again
        self._pca = None

    # Train a classifier pipeline that may or may not use PCA or other
    # feature selection methods
    def train_classifier(self):

        params = dict() 
        pipeline_steps = []

        # If we're doing PCA, create the PCA object and add it to the pipeline
        # and add the parameters  
        if self._use_pca:
            if not self._pca:
                # If we haven't set the number of components for PCA, search for it
                if self._pca_n == None:
                    self._pca = RandomizedPCA()
                    params['pca__n_components'] = [i for i in range(2,self._pca_max_n+1)]
                    params['pca__whiten'] = [True, False]
                    params['pca__iterated_power'] = [2,3,4,5]
                else:
                    self._pca = RandomizedPCA(n_components = self._pca_n)

            pipeline_steps.append(('pca', self._pca))

        # Add the correct learner to the pipeline, along with its parameters 
        if self._learner == 'rfc':
            rfc = RandomForestClassifier()
            pipeline_steps.append(('rfc', rfc))
            params['rfc__n_estimators'] = [i for i in range(1,10)]
            params['rfc__max_depth'] = [i for i in range(1,10)]
            params['rfc__criterion'] = ['gini','entropy']
        elif self._learner == 'logistic':
            logistic = LogisticRegression()
            pipeline_steps.append(('logistic', logistic))
            params['logistic__C'] = [1.0, 10.0, 100.0, 1000.0]
            params['logistic__solver'] = ['newton-cg', 'lbfgs', 'sag', 'liblinear'] 
        elif self._learner == 'svc':
            svc = SVC()
            pipeline_steps.append(('svc', svc))
            params['svc__kernel'] = ['poly', 'rbf']
            params['svc__degree'] = [2, 3, 4, 5]
            params['svc__C'] = [1.0, 10.0, 100.0, 1000.0]
        elif self._learner == 'knc':
            knc = KNeighborsClassifier()
            pipeline_steps.append(('knc', knc))
            params['knc__n_neighbors'] = [5, 10, 15, 20]
            params['knc__weights'] = ['uniform', 'distance']
            params['knc__algorithm'] = ['ball_tree', 'kd_tree', 'auto']
            params['knc__leaf_size'] = [10, 20, 30] 
        else:
            raise Exception('Undefined learner!')
 
        pipe = Pipeline(steps=pipeline_steps)

        # Perform a grid search to find the best learning parameters
        (X, y) = self._train_data
        self._clf = GridSearchCV(pipe, params, scoring=make_scorer(precision_score))
        start = clock()
        self._clf.fit(X, y)
        end = clock()

        self._first_classifier = False

        # Return the time it took to train the classifier
        return (end - start)

    # Do any needed preprocessing of the data (normalization, addressing missing
    # values, centering, etc)
    def preprocess_data(self):

        # Use interpolation to handle missing data
        self._df.interpolate()

        # Normalize and center the data
        cols = self._df.columns.tolist()
        cols = [c for c in cols if c not in [self._target_name]]
        self._df[cols] = self._df[cols].apply(lambda x: (x-x.mean())/(x.max()-x.min()))

        # Set the positive label to 1 and the negative label to 0
        self._df.replace(to_replace=self._positive_label, value=1, inplace=True)
        self._df.replace(to_replace=self._negative_label, value=0, inplace=True)

    # Split the dataframe into X (the data) and y (the target), for both training
    # and testing
    def split_dataframe(self):

        cols = self._df.columns.tolist()
        cols = [c for c in cols if c not in [self._target_name]]
        X, y = self._df[cols].values, self._df[self._target_name].values

        X_train, X_test, y_train, y_test = train_test_split(
                                                X, y, 
                                                test_size=0.2)

        self._train_data = (X_train, y_train)
        self._test_data = (X_test, y_test)

    # Run all of the steps to produce a classifier from the given data
    def run(self, verbose=False):

        if self._first_classifier:
            print "done.\n",
        print " + Training classifier (learner = %s, use_pca = %r)..." \
            % (self._learner, self._use_pca),; stdout.flush()
        training_time = self.train_classifier()
        
        if verbose:
            print "done\n   * Best parameters:" 
            for (key, value) in self._clf.best_params_.iteritems():
                print "     - " + key[key.find("__")+2:] + " = " + str(value)

            # Print out our precision (swallow the warning about changed behavior
            # of the scoring function)
            (X_test, y_test) = self._test_data
            start = clock()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = self._clf.score(X_test, y_test)
            end = clock()
            testing_time = (end - start)

            print
            print "   * Precision (learner = %s): %f" % (self._learner, score)
            print "   * Training time: %f seconds" % training_time   
            print "   * Testing time: %f seconds" % testing_time 
            print
 
