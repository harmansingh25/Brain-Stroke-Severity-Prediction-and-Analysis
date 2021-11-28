# Brain Stroke Severity Prediction and Analysis

# Introduction
Stroke is a disease that affects the arteries leading to and within the brain. A stroke occurs when a blood vessel that carries oxygen and nutrients to the brain is either blocked by a clot or ruptures. According to the WHO, stroke is the 2nd leading cause of death worldwide. Globally, 3% of the population are affected by subarachnoid hemorrhage, 10% with intracerebral hemorrhage, and the majority of 87% with ischemic stroke. 80% of the time these strokes can be prevented, so putting in place proper education on the signs of stroke is very important. The existing research is limited in predicting risk factors pertained to various types of strokes. This research work proposes an early prediction of stroke diseases by using different machine learning approaches with the occurrence of hypertension, body mass index level, average glucose level, smoking status, previous stroke and age. Machine Learning techniques including Logistic Regression, Random Forest, Decision Trees, Naive Bayes, SVM, MLP etc. are used to predict the severity of future stroke occurrence on a scale of 0 to 3. The study not only predicts the fututre risk of a getting a stroke for a certain individual who has never had a stroke, but also the future risk of occurrence of a more dangerous variant of stroke for those who have already had a stroke.

# Dataset
https://data.mendeley.com/datasets/jpb5tds9f6/1

### Installed dependencies
    python 3.7
    jupyter notebook.
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from math import sqrt
    from google.colab import drive
    import sklearn
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import MinMaxScaler
    from imblearn.combine import SMOTETomek
    import seaborn as sns
    from collections import Counter
    import warnings
    from numpy import mean
    from numpy import std
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import validation_curve
    from numpy import arange
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import ComplementNB
    from sklearn.metrics import cohen_kappa_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    import pydotplus
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    from sklearn.manifold import TSNE
    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import PCA
    





# Contributors:
Harman Singh (2019042) \
Amisha Aggarwal (2019016) \
Yash Tanwar (2019130) \
Meenal Gurbaxani (2019434)
