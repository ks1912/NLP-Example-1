# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts

# Cleaning a single line
"""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus  import stopwords
from nltk.stem.porter import PorterStemmer
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review = review.lower()
corpus = []
# Execute the 3 lines together otherwise some output error may occur
    review = review.split()
    ps = PorterStemmer() # Loved will change to love
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Stemming -->  Taking the root of word eg--> Dataset[Review][0]--> loved
    review = ' '.join(review)
"""
#---------------------------------------
# Cleaning all the dataset text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# -----------------------------------------------
# bag of words model
# We take all the 1000 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
# Definition : CountVectorizer(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern=r"(?u)
# tokenizer is used for cleaning the text.
# lowercase is used for converting Upper case to lower case.
# stopwords is used for removing un necessary words from the text.
# Sparse_matrix is being created here variable name X and toarray is used to create matrix here
# We can use max_feature which will keep more frequent words in review 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values
# Bag of Model is created here
#--------------------------------------------------------------------
"""
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
"""
# ------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(55+91)/200