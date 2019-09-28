# from models import Note
# from db import session
from datetime import datetime
from flask_restful import reqparse
from flask_restful import abort
from flask_restful import Resource
from flask_restful import fields
from flask_restful import marshal_with


parser = reqparse.RequestParser()
parser.add_argument('list')

import sys
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 

import gensim

import re
import codecs
import matplotlib.pyplot as plt
from subprocess import check_output
import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/IPython/.*')

from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer 

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df
def tok(DataDic):
    tokenizer = RegexpTokenizer(r'\w+')
    DataDic['Text'] = DataDic['Text'].astype('str') 
    DataDic.dtypes
    DataDic['Text'] = DataDic['Text'].str.lower()
    DataDic["tokens"] = DataDic["Text"].apply(tokenizer.tokenize)
    #Filtrer Punctuation
    stop_words = set(nltk.corpus.stopwords.words('french'))
    arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
    stop_words.update(arb_stopwords)

    DataDic["tokens"] = DataDic["tokens"].apply(lambda vec: [word for word in vec if word not in stop_words])
    ponctuation = [",",".","?","[","]","(",")","{","}"]
    DataDic["tokens"] = DataDic["tokens"].apply(lambda vec: [word for word in vec if word not in ponctuation ])
    return DataDic
def stem(DataDic):
    for i in range (len(DataDic)):
        ps = PorterStemmer() 
        for j in range (len(DataDic.tokens[i])):

            DataDic.tokens[i][j]=DataDic.tokens[i][j].replace('il vous plait','')
            DataDic.tokens[i][j]=DataDic.tokens[i][j].replace('svp','')
            DataDic.tokens[i][j]=DataDic.tokens[i][j].replace('plait','')
            DataDic.tokens[i][j]=DataDic.tokens[i][j].replace('brabi','')
            DataDic.tokens[i][j]=DataDic.tokens[i][j].replace('aman','')
            DataDic.tokens[i][j]=DataDic.tokens[i][j].replace('soume','prix')
            DataDic.tokens[i][j]=ps.stem(DataDic.tokens[i][j])
    return DataDic
def NewInput(DataDic):
    DataDic=DataDic[DataDic["Text"]!=np.nan]
    DataDic=DataDic.reset_index(drop=True)
    DataDic["NewInput"]=np.nan
    for i in range(len(DataDic)):
        DataDic["NewInput"][i]=" ".join(DataDic["tokens"][i])
    return DataDic
def vector(DataDic):
    vectorizer = CountVectorizer(stop_words = set(nltk.corpus.stopwords.words('french')))
    documents = list(DataDic.NewInput)

    x=vectorizer.fit_transform(documents)
    x=x.toarray()
    return x
def cahclass(x):
    model = AgglomerativeClustering(n_clusters=50, affinity='euclidean', linkage='ward')
    model.fit(x)
    return model.labels_
def amelior(DataDic):
    cons=DataDic[DataDic.Classe!='notdef']  
    pred=DataDic[DataDic.Classe=='notdef'] 
    cons.index= range(len(cons))
    pred.index=range(len(pred))
    for i in range(len(cons)):
        for j in range (len(pred)):
                if cons.newpred[i]== pred.newpred[j]:
                    pred.Classe[j]=cons.Classe[i]
    return pred

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class RequestList(Resource):
    def get(self, compI):
        return {'hello': 'world'}

    def post(self, compI):
        parsed_args = parser.parse_args()
        hr = parsed_args.list
        
        print(hr)
        k = {}
        
        
        DataDic=pd.DataFrame()
        DataDic['Text']=hr
        DataDic['Classe']=compI
        DataDic=DataDic[DataDic["Text"]!=np.nan]
        DataDic=DataDic.reset_index(drop=True)
        DataDic=tok(DataDic)
        DataDic=stem(DataDic)
        DataDic=NewInput(DataDic)
        x=vector(DataDic)
        DataDic['newpred']=cahclass(x)
        pred=amelior(DataDic)
        k['list']=pred['Text']
        k['Classe']=pred['Classe']
        
        
        
        return k
        