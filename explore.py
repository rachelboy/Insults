# ALL the Scikit-Learn functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model as lm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import naive_bayes as nb

# Workhorse libraries for vector ops and loading
import numpy as np
import pandas as pd

# Visualization
from matplotlib import pyplot as plt

# Miscellaneous
import string
import os
from scipy.sparse import hstack, csr_matrix


# Used for counting punctuation present in text
exclude = set(string.punctuation)

'''
An example of custom feature engineering -

Given a string of text, it returns a set of features that hopefully
describe some aspect of the text meaningfully for the classifier to use.
Since we're trying to figure out if the text is an insult we've picked
features like the caps ratio which would make sense.

Features are:

Number of words in the text
Ratio of capitalized characters to non-capitalized
Average word length
Whether or not there is capitalization
The ratio of punctuation to number of words
'''
def text_features(text):
	n_words = float(len(text.split(' ')))
	n_alphanumeric = float(len([ch for ch in text if ch not in exclude]))
	n_caps = float(len([ch for ch in text if ch.isupper()]))
	word_len = n_alphanumeric/n_words
	cap_ratio = n_caps/n_alphanumeric
	caps_present = cap_ratio != 0
	punc_ratio = len([ch for ch in text if ch in exclude])/n_words
	return [n_words,cap_ratio,punc_ratio,word_len,caps_present]

'''
Uses pandas to load the csv.

Generates text_features for each example and does some distribution scaling to
make them more Gaussian like via sqrts and log transforms.
Shows more complex numpy slicing as well.
(Assigning and slicing multiple columns using an index list)
'''
def load(f):

	# Load data with pandas
	data = pd.read_csv(f)

	# Nice pandas data selecting using the csv header column names
	text = data['Comment']
	labels = data['Insult']

	# Get text features for every text example using a list comprehension
	# which is like a condensed for-loop.
	text_feats = np.array([text_features(t) for t in text])

	# Feature scaling via log and sqrt transforms to force their distributions
	# into more Guassian (normal bell curve) shapes
	text_feats[:,[0,2,3]] = np.log(text_feats[:,[0,2,3]])
	text_feats[:,1] = np.sqrt(text_feats[:,1])

	return text,text_feats,labels

# Going to have to change to wherever your data is stored
data_dir = '/home/rboy/KaggleKaggleKaggle/Insults/'

# Loading in raw training and test data
train_text,train_text_feats,train_labels = load(os.path.join(data_dir,'train.csv'))
test_text,test_text_feats,test_labels = load(os.path.join(data_dir,'impermium_verification_labels.csv'))

# Example of using a TfidfVectorizer to convert text to ML friendly format
# Uses bigrams and a minimum document frequency to improve quality of text vector

'''vect1 = TfidfVectorizer(min_df=0.001,ngram_range=(1, 2),analyzer='word')
vect1.fit(np.hstack((train_text,test_text)))
train_text1 = vect1.transform(train_text)
test_text1 = vect1.transform(test_text)
'''
vect2 = TfidfVectorizer(min_df=0.0005,max_df=0.6,ngram_range=(3, 6),analyzer='char')
vect2.fit(np.hstack((train_text,test_text)))
train_text2 = vect2.transform(train_text)
test_text2 = vect2.transform(test_text)


# Scale the generated text features to be friendly to linear models
scaler = StandardScaler()
scaler.fit(np.vstack((train_text_feats,test_text_feats)))
train_text_feats = scaler.transform(train_text_feats)
test_text_feats = scaler.transform(test_text_feats)

# Adding the extracted text features into the word vectors to have a single 
# feature matrix to train the model on. hstack stacks columns horizontally 
# which is what we want here (adding more features to the rows of examples)
train_text = hstack((csr_matrix(train_text_feats),train_text2))
test_text = hstack((csr_matrix(test_text_feats),test_text2))

# Standard sklearn interface example
# Using Ridge Regression which is a form of Ordinary Least Squares Line fitting
# Kind of un-orthodox for a classification problem but since the scoring metric
# AUC is rank based it doesn't matter.
model = lm.Ridge()
model.fit(train_text,train_labels)
preds = model.predict(test_text)

print 'AUC score:',round(metrics.roc_auc_score(test_labels,preds),5)
