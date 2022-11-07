import numpy as np
import random
import re
from nltk.util import ngrams
import itertools
import pandas as pd
from sklearn import svm
from joblib import dump, load
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split





alphanum = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','0','1','2','3','4','5','6','7','8','9']
permutations = itertools.product(alphanum, repeat=3)

featuresDict = {}
counter = 0
for perm in permutations:
    featuresDict[(''.join(perm))] = counter
    counter = counter + 1


def generate_ngram(sentence):
    s = sentence.lower()
    s = ''.join(e for e in s if e.isalnum()) #replace spaces and slashes
    processedList = []
    for tup in list(ngrams(s,3)):
        processedList.append((''.join(tup)))
    return processedList

def preprocess_sentences():
    df = pd.read_csv("raw_data\\phishing_site_urls.csv")
    for index,row in df.iterrows():
        url = row['URL'].strip().replace("https://","")
        url = row['URL'].strip().replace("http://","")
        url = url.replace("http://","")
        url = re.sub(r'\.[A-Za-z0-9]+/*','',url)
    
        for gram in generate_ngram(url):

            try:
                X[index][featuresDict[gram]] = X[index][featuresDict[gram]] + 1
            except:
                print(gram,"doesn't exist")
        # y[index] = int(row['label'])
        # return (X,y)
preprocess_sentences()