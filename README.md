
# Contributors:
Harman Singh (harman19042@iiitd.ac.in) \
Amisha Aggarwal (amisha19016@iiitd.ac.in) \
Yash Tanwar (yash19130@iiitd.ac.in) \
Meenal Gurbaxani (meenal19434@iiitd.ac.in)


# Brain Stroke Severity Prediction and Analysis
Stroke is a disease that affects the arteries leading to and within the brain. A stroke occurs when a blood vessel that carries oxygen and nutrients to the brain is either blocked by a clot or ruptures. According to the WHO, stroke is the 2nd leading cause of death worldwide. Globally, 3% of the population are affected by subarachnoid hemorrhage, 10% with intracerebral hemorrhage, and the majority of 87% with ischemic stroke. 80% of the time these strokes can be prevented, so putting in place proper education on the signs of stroke is very important. The existing research is limited in predicting risk factors pertained to various types of strokes. This research work proposes an early prediction of stroke diseases by using different machine learning approaches with the occurrence of hypertension, body mass index level, average glucose level, smoking status, previous stroke and age. Machine Learning techniques including Logistic Regression, Random Forest, Decision Trees, Naive Bayes, SVM, MLP etc. are used to predict the severity of future stroke occurrence on a scale of 0 to 3. The study not only predicts the fututre risk of a getting a stroke for a certain individual who has never had a stroke, but also the future risk of occurrence of a more dangerous variant of stroke for those who have already had a stroke.

# Introduction #
In 2018, 1 in every six deaths from cardiovascular disease was due to stroke. Someone in the United States has a stroke every 40 seconds. Every 4 minutes, someone dies of a stroke. Every year, more than 795,000 people in the United States have a stroke. About 610,000 of these are first or new strokes. Stroke is a leading cause of serious long-term disability. Stroke reduces mobility in more than half of stroke survivors age 65 and over.
### Brain Stroke ###
According to the definition proposed by the World Health Organization in 1970, “stroke is rapidly developing clinical signs of focal (or global) disturbance of cerebral function, with symptoms lasting 24 hours or longer, or leading to death, with no apparent cause other than of vascular origin”. A stroke occurs when blood flow to different areas of the brain gets disrupted and the cells in those regions do not get nutrients and oxygen, and as a result, start to die. Stroke is a medical emergency requiring immediate care. Early detection can help minimize further damage to the affected areas of the brain and avoid other complications in the body. Strokes are broadly of two types. If the flow of blood among the blood tissues decreases, it is a case of ischemic stroke. On the other hand, internal bleeding among the brain tissues results in a hemorrhagic stroke.
### Machine Learning for Stroke Prediction ###
Machine learning techniques can be used to predict the occurrence and risk of stroke in a human being.The existing research is limited in predicting whether a stroke will occur or not.
Our work attempts to predict the risk of stroke-based upon a ranking scale determined with the following criteria: 0:Low risk, 1: Moderate Risk, 2: High Risk, 3: Severe risk. This is a multiclass classification in contrast to the binary classification done by most authors earlier.
We have used features including hypertension, body mass index level, average glucose level, smoking status, previous stroke severity(Nihss score), age and gender to predict the risk of stroke for an individual. Machine Learning techniques including Logistic Regression, Random Forest, Decision Trees and Naive Bayes have been used for prediction.Our work also determines the importance of the characteristics available and determined by the dataset.Our contribution can help predict early signs and prevention of this deadly disease.

# Dataset #
* The source of the dataset is : https://data.mendeley.com/datasets/jpb5tds9f6/1 
* Dataset can also be found in this repository with the path ./Stroke_analysis1 - Stroke_analysis1.csv
* The dataset description is as follows: \
  The dataset consists of 4798 records of patients out of which 3122 are males and 1676 are females. There are 12 primary features describing the dataset with one   feature being the target variable. The description about the primary features is given in the following table. \
  ![alt text](https://github.com/harmansingh25/ML_Project_2021/blob/main/Plots/Dataset%20Description.png?raw=true)


# Steps to run models #
The models are saved via pickle in the /Weights directory by their name. Eg LogisticRegression
  * To load the models
 
        file = open("modelName", "rb")
        model = pickle.load(file)
        file.close()    
    
  * To test the models 
  
        X_test : Preprocessed testing data
        Y_pred = model.predict(X_test) : Predicted array
  * To run the entire file on Google Colab
        
        Click on Runtime -> Run all 
 
      

# Installed dependencies #
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
    





