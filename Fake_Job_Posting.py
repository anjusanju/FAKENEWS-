import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Other visualization packages
import seaborn as sns

#Importing NLP plugins

import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
stop_words = stopwords.words('english')
from nltk.stem import WordNetLemmatizer 
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Importing our Sklearn Plugins
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#importing our models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#Model Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


class Fake_Job_Posting_Predction:

    def __init__(self):
        pass

    def load_dataset(self):
        self.df = pd.read_csv("datasets/fake_job_postings.csv")

    def tokenizer(self, text):
        
        #All characters in this string will be converted to lowercase
        text = text.lower()
        
        #Removing sentence punctuations
        for punctuation_mark in string.punctuation:
            text = text.replace(punctuation_mark,'')
        
        #Creating our list of tokens
        list_of_tokens = text.split(' ')
        #Creating our cleaned tokens list 
        cleaned_tokens = []
        #Intatiating our Lemmatizer
        lemmatizer = WordNetLemmatizer()
        #Removing Stop Words in our list of tokens and any tokens that happens to be empty strings
        for token in list_of_tokens:
            if (not token in stop_words) and (token != ''):
                #lemmatizing our token
                token_lemmatized = lemmatizer.lemmatize(token)
                #appending our finalized cleaned token
                cleaned_tokens.append(token_lemmatized)
        
        return cleaned_tokens

    def preprocess(self):

        #Check percentage of data missing for each feature/column
        self.df.isna().sum()/len(self.df)
        df_2 = self.df.copy()
        df_2 = df_2.drop(labels = ['job_id','salary_range',
                            'department','benefits',
                            'company_profile'], axis = 1) #axis = 1 to refer droping columns
        df_2['employment_type'] = df_2['employment_type'].bfill(axis=0)
        df_2['required_experience'] = df_2['required_experience'].bfill(axis = 0)
        df_2['required_education'] = df_2['required_education'].bfill(axis = 0)
        df_2['industry'] = df_2['industry'].bfill(axis = 0)
        df_2['function'] = df_2['function'].bfill(axis = 0)
        # Make Dataframe copy
        df_3 = df_2.copy()
        # Keeping non NaN rows in my new dataframe
        df_3 = df_3[df_3['description'].notna()]
        df_3 = df_3.dropna(axis = 0, how = 'any')
        df_3 = df_3.drop_duplicates(keep = 'first')
        df_3.isna().sum()/len(self.df)
        # Make copy
        df_4 = df_3.copy()
        #concatenating our description and requirments columns
        df_4['description'] = df_4['description'] + ' ' + df_4['requirements']
        del df_4['requirements']
        df_clean = df_4.copy()
        df_5 = df_clean.copy()
        # One Hot Encoding using Pandas get dummies function
        columns_to_1_hot = ['employment_type','required_experience','required_education',
                        'industry', 'function']
        for column in columns_to_1_hot:
            encoded = pd.get_dummies(df_5[column])
            df_5 = pd.concat([df_5, encoded], axis = 1)
        columns_to_1_hot += ['title', 'location']
        #droping the original columns that we just one hot encoded from
        df_5 = df_5.drop(columns_to_1_hot, axis = 1)
        df_6 = df_5.copy()
        #Instatiating our tfidf vectorizer
        tfidf = TfidfVectorizer(tokenizer = self.tokenizer, min_df = 0.05, ngram_range=(1,3))
        print (tfidf)
        print (df_6['description'])
        #Fit_transform our description 
        tfidf_features = tfidf.fit_transform(df_6['description']) #this will create a sparse matrix
        #I want to append this sparse matrix to the original pandas Dataframe
        tfidf_vect_df = pd.DataFrame(tfidf_features.todense(), columns = tfidf.get_feature_names())

        df_tfidf = pd.concat([df_6, tfidf_vect_df], axis = 1)

        #Minor Cleaning steps after appending our tfidf results to our Dataframe, we will need to drop the description column. 
        df_tfidf = df_tfidf.drop(['description'], axis = 1)
        df_tfidf = df_tfidf.dropna()
        self.df_tfidf = df_tfidf
        #Instatiating our CountVectorizer
        # count_vect = CountVectorizer(tokenizer = self.tokenizer, min_df = 0.05, ngram_range=(1,3))
        # #Fit_transform our description 
        # count_vect_features = count_vect.fit_transform(df_6['description']) #this will create a sparse matrix

        # count_vect_df = pd.DataFrame(count_vect_features.todense(), columns = count_vect.get_feature_names())

        # df_count_vect = pd.concat([df_6, count_vect_df], axis = 1)
        # df_count_vect = df_count_vect.drop(['description'], axis = 1)
        # df_count_vect = df_count_vect.dropna()

    def train_test_split(self):
        target = self.df_tfidf.fraudulent
        features = self.df_tfidf.drop(['fraudulent'], axis = 1)

        #Spliting our Data into train and holdout sets to test our models
        print (features.head(5))
        self.X_train, self.X_hold, self.y_train, self.y_hold = train_test_split(features, target, test_size = 0.1,
                                                            stratify = target, random_state = 42)
    def prepare_logistic_regression(self):

        #Intatiating our Logistic Regression Model
        log_reg = LogisticRegression()
        #I want to optimze the C-Value and penalty
        c_values = [.00001, .0001, .001, .1, 1, 10, 100, 1000, 10000]
        penalty_options = ['l1','l2']

        param_grid = dict(C = c_values, penalty = penalty_options)

        grid_tfidf = GridSearchCV(log_reg, param_grid= param_grid, cv = 10, scoring = 'roc_auc', n_jobs = -1)

        grid_tfidf.fit(self.X_train, self.y_train)

        GridSearchCV(cv=10, estimator=LogisticRegression(), n_jobs=-1,
                    param_grid={'C': [1e-05, 0.0001, 0.001, 0.1, 1, 10, 100, 1000,
                                    10000],
                                'penalty': ['l1', 'l2']},
                    scoring='roc_auc')

        print(grid_tfidf.best_score_)
        print(grid_tfidf.best_params_)

        log_reg_tfidf_pred = grid_tfidf.predict(self.X_hold)
        print(roc_auc_score(self.y_hold, log_reg_tfidf_pred))
        print(classification_report(self.y_hold, log_reg_tfidf_pred))
    def prepare_KNN(self):
        # Model - KNearestNeighbors
        knn = KNeighborsClassifier()

        #The parameters we would like to optimize for
        k_range = list(np.arange(2,23,2))
        param_grid_knn = dict(n_neighbors=k_range)
        print(param_grid_knn)

        {'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]}

        #Intatiate our knn gridsearch
        grid_knn = GridSearchCV(knn, param_grid_knn, cv=10, scoring='roc_auc',
                                n_jobs = -1)

        #Fit our grid_knn
        grid_knn.fit(self.X_train, self.y_train)
        print(grid_knn.best_score_)
        print(grid_knn.best_params_)

        0.857757918485041
        {'n_neighbors': 20}

        #predicting on our holdout data
        knn_pred = grid_knn.predict(self.X_hold)
        #Printing out our evaluation metrics
        print(roc_auc_score(self.y_hold, knn_pred))
        print(classification_report(self.y_hold, knn_pred))




predictor = Fake_Job_Posting_Predction()
predictor.load_dataset()
predictor.preprocess()
predictor.train_test_split()
# predictor.prepare_logistic_regression()
# predictor.prepare_KNN()




