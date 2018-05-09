#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 23:05:53 2018

@author: sasha
"""


# coding: utf-8

# =============================================================================
# PART 1
# Crash Prediction
# =============================================================================

import pandas as pd
import numpy as np
import string

# Text topic imports
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
# class for decision tree
from Class_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from collections import defaultdict

# =============================================================================
# Get statistics from confusion matrix
# =============================================================================
def getClassificationMetrics(tn, fp, fn, tp):
       
    #sensitivity
    Recall=tp/(tp+fn);
    print("sensitivity/recall/TPR:", Recall)
    #specificity
    Specificity= tn/(tn+fp)
    print("specificity: ", Specificity)
    #accuracy
    print("accuracy: ", (tp+tn)/(tp+fn+tn+fp))
    #precision
    Precision=tp/(tp+fp);
    print("precision: ", Precision)
    #f1 score
    print("f1_score: ",  (2*Recall*Precision)/(Recall + Precision))
    #misclassification
    print("misc: ", (fp+fn)/(tp+fn+tn+fp))
    #False Positive Rate
    print("FPR: ", 1-Specificity)
        
    return


# =============================================================================
# This pre processing is used for finding synonyms
# =============================================================================
def DoPreProcessing(s):
    
    # Replace special characters with spaces
    s = s.replace('-', ' ')
    s = s.replace('_', ' ')
    s = s.replace(',', '. ')
    # Replace not contraction with not
    s = s.replace("'nt", " not")
    s = s.replace("n't", " not")
    # Tokenize 
    tokens = word_tokenize(s)
    #tokens = [word.replace(',','') for word in tokens ]
    tokens = [word for word in tokens if ('*' not in word) and               ("''" != word) and ("``" != word) and               (word!='description') and (word !='dtype')               and (word != 'object') and (word!="'s")]
       
            
    # Remove stop words
    punctuation = list(string.punctuation)+['..', '...']
    pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
    #Top frequency words ar usually stop words. Top 150 words by frequency are
    #listed and then manually below words were added as stop words
    others   = ["'m","us","v","ws","w","eld","would", "told","tc","sr",
                "cls","could", "took", "said", "get", "since",
                  "came", "went", "called", "go", "going","'d", "co","gm",
                  "ed", "put", "say", "get", "can", "become",
                "los", "sta", "la", "use", "iii", "else", "could", "also",
                "even", "really", "one", "would", "get", "getting", "go", "going",
                "place", "want", "get","take", "end","next", "though","non", "seem"
               ]

    stop = stopwords.words('english') + punctuation + pronouns + others
    filtered_terms = [word for word in tokens if (word not in stop) and                   (len(word)>1) and (not word.replace('.','',1).isnumeric())                   and (not word.replace("'",'',2).isnumeric())]
    
    # Lemmatization & Stemming - Stemming with WordNet POS
    # Since lemmatization requires POS need to set POS
    tagged_words = pos_tag(filtered_terms, lang='eng')
    # Stemming with for terms without WordNet POS
    stemmer = SnowballStemmer("english")
    wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
    wnl = WordNetLemmatizer()
    stemmed_tokens = []
    for tagged_token in tagged_words:
        term = tagged_token[0]
        pos  = tagged_token[1]
        pos  = pos[0]
        try:
            pos   = wn_tags[pos]
            stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
        except:
            stemmed_tokens.append(stemmer.stem(term))
    #print(stemmed_tokens)        
    return stemmed_tokens
   
    
# =============================================================================
# Get synonyms that need to be replaced
# =============================================================================
def get_synonyms(totalList):
           
    #this dictionary contains words and their synonyms
    #Some of the main words are not in the corpus, but their synonyms are
    d = defaultdict(list)
    
    for item in totalList:
        syn = wn.synsets(item)
        if len(syn)>0:
                    if syn[0].lemma_names()[0]!=item:
                            d[syn[0].lemma_names()[0]].append(item)
                        
    len(d)
    
    #This list contains main words and their synonyms
    # if no main word is present its first synonym becomes the main words
    #and the subsequent synonyms become its synonyms
    synonyms = defaultdict(str)
    
    for item in d:
        if totalList.count(item)==0:
            if len(d[item])>1:
                # the flag is there to make the first synonym in the list 
                # as main word
                flag =0
                for a in d[item]:
                    if flag==0:
                        flag = flag+1
                    else:
                        synonyms[a] = d[item][0]
    return synonyms
    
# =============================================================================
# Used for finding the Term/Document matrix
# =============================================================================
def my_analyzer(s):
    # Synonym List
# =============================================================================
#     syns = { "n't":'not',  'wont':'would not', 'cant':'can not', 'cannot':'can not', 
#               'couldnt':'could not', 'shouldnt':'should not', 
#               'wouldnt':'would not',  }
# =============================================================================
   
    syns = synonymsDict
    # Preprocess String s
    s = s.lower()
    # Replace special characters with spaces
    s = s.replace('-', ' ')
    s = s.replace('_', ' ')
    s = s.replace(',', '. ')
    # Replace not contraction with not
    s = s.replace("'nt", " not")
    s = s.replace("n't", " not")
    # Tokenize 
    tokens = word_tokenize(s)
    #tokens = [word.replace(',','') for word in tokens ]
    tokens = [word for word in tokens if ('*' not in word) and               ("''" != word) and ("``" != word) and               (word!='description') and (word !='dtype')               and (word != 'object') and (word!="'s")]
    
    
            
    # Remove stop words
    punctuation = list(string.punctuation)+['..', '...']
    pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
    #Top frequency words ar usually stop words. Top 150 words by frequency are
    #listed and then manually below words were added as stop words
    others   = ["us","v","ws","w","eld","would", "told","tc","sr",
                "cls","could", "took", "said", "get", "since",
                  "came", "went", "called", "go", "going","'d", "co","gm",
                  "ed", "put", "say", "get", "can", "become",
                "los", "sta", "la", "use", "iii", "else", "could", "also",
                "even", "really", "one", "would", "get", "getting", "go", "going",
                "place", "want", "get","take", "end","next", "though","non", "seem"
               ]

    stop = stopwords.words('english') + punctuation + pronouns + others
    filtered_terms = [word for word in tokens if (word not in stop) and (len(word)>1) and (not word.replace('.','',1).isnumeric())                   and (not word.replace("'",'',2).isnumeric())]
    
    # Lemmatization & Stemming - Stemming with WordNet POS
    # Since lemmatization requires POS need to set POS
    tagged_words = pos_tag(filtered_terms, lang='eng')
    # Stemming with for terms without WordNet POS
    stemmer = SnowballStemmer("english")
    wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
    wnl = WordNetLemmatizer()
    stemmed_tokens = []
    for tagged_token in tagged_words:
        term = tagged_token[0]
        pos  = tagged_token[1]
        pos  = pos[0]
        try:
            pos   = wn_tags[pos]
            stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
        except:
            stemmed_tokens.append(stemmer.stem(term))
     
        
    for i in range(len(stemmed_tokens)):
        if stemmed_tokens[i] in syns:
            stemmed_tokens[i] = syns[stemmed_tokens[i]]   
    #print(stemmed_tokens)
    
    return stemmed_tokens


# =============================================================================
# Used for sentiment analysis
# =============================================================================
def my_preprocessor(s):
   # Preprocess String s
    s = s.lower()
   # Replace special characters with spaces
    s = s.replace('-', ' ')
    s = s.replace('_', ' ')
    s = s.replace(',', '. ')
   # Replace not contraction with not
    s = s.replace("'nt", " not")
    s = s.replace("n't", " not")
    return s



def display_topics(lda, terms, n_terms=15):
    for topic_idx, topic in enumerate(lda):
        message  = "Topic #%d: " %(topic_idx+1)
        print(message)
        abs_topic = abs(topic)
        topic_terms_sorted =                 [[terms[i], topic[i]]                      for i in abs_topic.argsort()[:-n_terms - 1:-1]]
        k = 5
        n = int(n_terms/k)
        m = n_terms - k*n
        for j in range(n):
            l = k*j
            message = ''
            for i in range(k):
                if topic_terms_sorted[i+l][1]>0:
                    word = "+"+topic_terms_sorted[i+l][0]
                else:
                    word = "-"+topic_terms_sorted[i+l][0]
                message += '{:<15s}'.format(word)
            print(message)
        if m> 0:
            l = k*n
            message = ''
            for i in range(m):
                if topic_terms_sorted[i+l][1]>0:
                    word = "+"+topic_terms_sorted[i+l][0]
                else:
                    word = "-"+topic_terms_sorted[i+l][0]
                message += '{:<15s}'.format(word)
            print(message)
        print("")
    return


#Set Seed
seed = 12345

# topic analysis 
pd.set_option("max_colwidth", 32000)
file_path = "/Users/sasha/Library/Mobile Documents/com~apple~CloudDocs/STAT 656/Final Exam/"
df = pd.read_excel(file_path + "HondaComplaints.xlsx")

df["description"] = df["description"].str.lower()


#Create complaints corpus 
description=""
for item in df["description"]:
      description = description+item
     

#To find stop words in first 150 words with highest frequency
stop = stopwords.words('english')
example = df['description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
example.head()
     
series=pd.Series(' '.join(example).split())
pd.Series(' '.join(example).split()).value_counts()[:50]
pd.Series(' '.join(example).split()).value_counts()[50:100]
pd.Series(' '.join(example).split()).value_counts()[100:150]
     
#Get unique words
StopWordsSeries= set(series)
StopWordslist = list(StopWordsSeries)

#create synonyms dictionary   
totallist= DoPreProcessing(description)
# used to remove duplicate items
totalSet= set(totallist)
totallist = list(totalSet)
  
synonymsDict=get_synonyms(totallist)
 


# Setup program constants
n_comments  = len(df['description']) # Number of wine reviews
m_features = None                   # Number of SVD Vectors
s_words    = 'english'               # Stop Word Dictionary
comments = df['description']         # place all text reviews in reviews
n_topics =  7                        # number of topic clusters to extract
max_iter = 10                        # maximum number of itertions  
learning_offset = 10.                 # learning offset for LDA
learning_method = 'online'            # learning method for LDA



# Create Word Frequency by Review Matrix using Custom Analyzer
cv = CountVectorizer(max_df=0.7, min_df=4, max_features=m_features,analyzer=my_analyzer, ngram_range=(1,2))
tf    = cv.fit_transform(comments)
terms = cv.get_feature_names()
term_sums = tf.sum(axis=0)
term_counts = []
for i in range(len(terms)):
    term_counts.append([terms[i], term_sums[0,i]])
def sortSecond(e):
    return e[1]
term_counts.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest Frequency:")
for i in range(10):
    print('{:<15s}{:>5d}'.format(term_counts[i][0], term_counts[i][1]))
print("")  




# Modify tf, term frequencies, to TF/IDF matrix from the data
print("Conducting Term/Frequency Matrix using TF-IDF")
tfidf_vect = TfidfTransformer(norm=None, use_idf=True) #set norm=None
tf         = tfidf_vect.fit_transform(tf)

term_idf_sums = tf.sum(axis=0)
term_idf_scores = []
for i in range(len(terms)):
    term_idf_scores.append([terms[i], term_idf_sums[0,i]])
print("The Term/Frequency matrix has", tf.shape[0], " rows, and",            tf.shape[1], " columns.")
print("The Term list has", len(terms), " terms.")
term_idf_scores.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest TF-IDF Scores:")
for i in range(10):
    print('{:<15s}{:>8.2f}'.format(term_idf_scores[i][0],  term_idf_scores[i][1]))




# In sklearn, LDA is synonymous with SVD (according to their doc)
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,learning_method=learning_method, learning_offset=learning_offset, random_state=seed)
lda.fit_transform(tf)
print('{:.<22s}{:>6d}'.format("Number of Reviews", n_comments))
print('{:.<22s}{:>6d}'.format("Number of Terms", len(terms)))
print("\nTopics Identified using LDA with TF_IDF")
display_topics(lda.components_, terms, n_terms=20)




# Review Scores
# Normalize LDA Weights to probabilities
lda_norm = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
# ***** SCORE REVIEWS *****
rev_scores = [[0]*(n_topics+1)] * n_comments
# Last topic count is number of reviews without any topic words
topic_counts = [0] * (n_topics+1)
for r in range(n_comments):
    idx = n_topics
    max_score = 0
    # Calculate Review Score
    j0 = tf[r].nonzero()
    nwords = len(j0[1])
    rev_score = [0]*(n_topics+1)
    # get scores for rth doc, ith topic
    for i in range(n_topics):
        score = 0
        for j in range(nwords):
            j1 = j0[1][j]
            if tf[r,j1] != 0:
                score += lda_norm[i][j1] * tf[r,j1]
        rev_score [i+1] = score
        if score>max_score:
            max_score = score
            idx = i
# Save review's highest scores
    rev_score[0] = idx
    rev_scores [r] = rev_score
    topic_counts[idx] += 1
    
# Augment Dataframe with topic group information
cols = ["topic"]
for i in range(n_topics):
    s = "T"+str(i+1)
    cols.append(s)
df_topics = pd.DataFrame.from_records(rev_scores, columns=cols)
df        = df.join(df_topics)



print("\n**** Sentiment Analysis ****")
sw = pd.read_excel(file_path + "/Afinn_sentiment_words.xlsx")

# setup sentiment dictionary
sentiment_dic = {}
for i in range(len(sw)):
    sentiment_dic[sw.iloc[i][0]] = sw.iloc[i][1]


# Create Word Frequency by Review Matrix using Custom Analyzer
# max_df is a stop limit for terms that have more than this
# proportion of documents with the term (max_df - don't ignore any terms)
cv = CountVectorizer(max_df=1.0, min_df=1, max_features=None, preprocessor=my_preprocessor, ngram_range=(1,2))
tf = cv.fit_transform(df['description'])
terms = cv.get_feature_names()
n_terms = tf.shape[1]
print('{:.<22s}{:>6d}'.format("Number of Reviews", n_comments))
print('{:.<22s}{:>6d}'.format("Number of Terms", n_terms))



# calculate average sentiment for every review save in sentiment_score[]
min_sentiment = +5
max_sentiment = -5
avg_sentiment, min, max = 0,0,0
min_list, max_list = [],[]
sentiment_score = [0]*n_comments
for i in range(n_comments):
    # iterate over the terms with nonzero scores
    n_sw = 0
    term_list = tf[i].nonzero()[1]
    if len(term_list) >0:
        for t in np.nditer(term_list):
            score = sentiment_dic.get(terms[t])
            if score !=None:
                sentiment_score[i] += score * tf[i, t]
                n_sw += tf[i, t]
    if n_sw >0:
        sentiment_score[i] = sentiment_score[i]/n_sw
    if sentiment_score[i]==max_sentiment and n_sw >3:
        max_list.append(i)
    if sentiment_score[i]>max_sentiment and n_sw>3:
        max_sentiment=sentiment_score[i]
        max=i
        max_list=[i]
    if sentiment_score[i]==min_sentiment and n_sw >3:
        min_list.append(i)
    if sentiment_score[i]<min_sentiment and n_sw>3:
        min_sentiment=sentiment_score[i]
        min=i
        min_list=[i]
    avg_sentiment += sentiment_score[i]
avg_sentiment = avg_sentiment/n_comments
print ("\nCorpus Average Sentiment: ", avg_sentiment)
print ("\nMost Negative Reviews with 4 or more Sentiment Words:")
for i in range(len(min_list)):
    print("{:<s}{:<d}{:<s}{:<5.2f}".format("   Review ", min_list[i],                                           " sentiment is ", min_sentiment))
print("\nMost Positive Reviews with 4 or more Sentiment Words:")
for i in range(len(max_list)):
     print("{:<s}{:<d}{:<s}{:<5.2f}".format("   Review ", max_list[i],                                           " sentiment is ", max_sentiment))
        
# Augment Dataframe with  sentiment score information
cols = ["sentiment"]
df_score = pd.DataFrame(sentiment_score, columns=cols)
df        = df.join(df_score)


#Average Sentiment by topic 
df.groupby(['topic'])['sentiment'].mean()

#Average Sentiment by make
df.groupby(['Make'])['sentiment'].mean()

#Average Sentiment by model
df.groupby(['Model'])['sentiment'].mean()

#Average Sentiment by make, topic and model
df.groupby(['Make','topic','Model'])['sentiment'].mean()


print('***Topic by Complaints count***')
df.groupby('topic').topic.count()




print("\n**** Decision tree Analysis ****")
# create attribute map
# Attribute Map:  the key is the name in the DataFrame
# The first number of 0=Interval, 1=binary and 2=nomial
# The 1st tuple for interval attributes is their lower and upper bounds
# The 1st tuple for categorical attributes is their allowed categories
# The 2nd tuple contains the number missing and number of outliers
attribute_map = {
    "NhtsaID":[3, (560001,10891880), [0,0]],
    "Make": [1, ("HONDA", "ACURA"), [0,0]],
    "Model":[2, ("TL", "ODYSSEY", "CR-V", "CL", "CIVIC", "ACCORD"),[0,0]],
    "Year":[2, (2001, 2002, 2003), [0,0]],
    "State": [3, (""), [0,0]],
    "abs": [1, ("Y", "N"), [0,0]],
    "cruise":[1, ("Y", "N"), [0,0]],
    "crash": [1, ("Y","N"), [0,0]],
    "mph": [0, (0,80), [0,0]],
    "mileage": [0,(0,200000), [0,0]],
    'topic':[2,(0,1,2,3,4,5,6),[0,0]],     
    'T1':[0,(-1e+8,1e+8),[0,0]],     
    'T2':[0,(-1e+8,1e+8),[0,0]],     
    'T3':[0,(-1e+8,1e+8),[0,0]],     
    'T4':[0,(-1e+8,1e+8),[0,0]],     
    'T5':[0,(-1e+8,1e+8),[0,0]],     
    'T6':[0,(-1e+8,1e+8),[0,0]],     
    'T7':[0,(-1e+8,1e+8),[0,0]],
    "sentiment":[0,(-1e+8,1e+8),[0,0]]
   
    
}


# drop=False - used for Decision tree
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', interval_scale = 'std',drop = False, display=True)
encoded_df = rie.fit_transform(df)
#create X and y
varlist = ["crash", 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'] 
y = encoded_df["crash"]
X = encoded_df.drop(varlist, axis=1)
np_y = np.ravel(y)
col  = rie.col 
for i in range(len(varlist)):     
    col.remove(varlist[i]) 




# Cross Validation for decision tree: 
# best model: Maximum Tree Depth:  15 Min_samples_leaf 3 Min_samples_split 5
depth_list = [5,6,8,10, 12, 15, 20, 25, 50]
minSamplesLeaf= [3,5,7]
minSamplesSplit=[3]

recall_best = 0
recall_best_model = ''
f1score_best = 0
f1score_best_model = ''
accuracy_best = 0
accuracy_best_model = ''
precision_best = 0
precision_best_model = ''

score_list = ['accuracy', 'recall', 'precision', 'f1']
for d in depth_list:
    for l in minSamplesLeaf:
        for s in minSamplesSplit:
            print("\nMaximum Tree Depth: ", d, "Min_samples_leaf", l, "Min_samples_split", s)
            dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l,  min_samples_split=s, random_state=seed)
            dtc = dtc.fit(X,np_y)
            scores = cross_validate(dtc, X, np_y, scoring=score_list, return_train_score=False, cv=10)
    
            print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
            for sl in score_list:
                var = "test_"+sl
                mean = scores[var].mean()
                std  = scores[var].std()
                if sl=='recall':
                    if recall_best<mean:
                        recall_best = mean
                        recall_best_model ="d:"+str(d)+" l:"+str(l)+" s:"+str(s)     
                if sl=='precision':
                    if precision_best<mean:
                        precision_best = mean
                        precision_best_model ="d:"+str(d)+" l:"+str(l)+" s:"+str(s)
                if sl=='f1':
                    if f1score_best<mean:
                        f1score_best = mean
                        f1score_best_model ="d:"+str(d)+" l:"+str(l)+" s:"+str(s)
                if sl=='accuracy':
                    if accuracy_best<mean:
                        accuracy_best = mean
                        accuracy_best_model ="d:"+str(d)+" l:"+str(l)+" s:"+str(s)
                    
                print("{:.<13s}{:>7.4f}{:>10.4f}".format(sl, mean, std))

# =============================================================================
# d: depth; l: leaf size; s: splits
# recall_best
# 0.53917120387174822
# 
# recall_best_model
# 'd:15 l:5 s:3'
# 
# f1score_best
# 0.59287652022030457
# 
# f1score_best_model
# 'd:15 l:5 s:3'
# 
# accuracy_best
# 0.92382945659149662
# 
# accuracy_best_model
# 'd:8 l:5 s:3'
# 
# precision_best
# 0.76163029737558041
# 
# precision_best_model
# 'd:6 l:5 s:3'
# =============================================================================
                
#15 ,3,5
# 70/30 split
X_train, X_validate, y_train, y_validate =  train_test_split(X, np_y,test_size = 0.3, random_state=seed)

# Decison Tree
dtc = DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, min_samples_split=3, random_state=seed)
dtc = dtc.fit(X_train,y_train)
DecisionTree.display_binary_split_metrics(dtc, X_train, y_train,X_validate, y_validate)

DecisionTree.display_importance(dtc, col)

# =============================================================================
# Training
# Confusion Matrix  Class 0   Class 1  
# Class 0.....      3292        50
# Class 1.....       118       271
# 
# 
# Validation
# Confusion Matrix  Class 0   Class 1  
# Class 0.....      1373        44
# Class 1.....        84        98
# =============================================================================

print('***** Train set ******')
getClassificationMetrics(3292, 50, 118, 271)


print('\n')
print('***** Validation set ******')
getClassificationMetrics(1373, 44, 84, 98)

# =============================================================================
# ***** Train set ******
# sensitivity/recall/TPR: 0.6966580976863753
# specificity:  0.9850388988629563
# accuracy:  0.9549718574108818
# precision:  0.8442367601246106
# f1_score:  0.7633802816901407
# misc:  0.0450281425891182
# FPR:  0.014961101137043742
# 
# 
# ***** Validation set ******
# sensitivity/recall/TPR: 0.5384615384615384
# specificity:  0.9689484827099506
# accuracy:  0.9199499687304565
# precision:  0.6901408450704225
# f1_score:  0.6049382716049383
# misc:  0.08005003126954346
# FPR:  0.031051517290049402
# 
# =============================================================================




# =============================================================================
# PART 2 
# Web Scrapping - Search Word 'Takata'
# API Key: 444171d89d544b2da002bb61fe78833a
# =============================================================================

import re
import requests
import newspaper
from newspaper import Article
from newsapi import NewsApiClient # Needed for using API Feed
from time import time 

# News Agencies used by API
agency_urls = {
'huffington': 'http://huffingtonpost.com',
'reuters': 'http://www.reuters.com',
'cbs-news': 'http://www.cbsnews.com',
'usa-today': 'http://usatoday.com',
'cnn': 'http://cnn.com',
'npr': 'http://www.npr.org',
'wsj': 'http://wsj.com',
'fox': 'http://www.foxnews.com',
'abc': 'http://abc.com',
'abc-news': 'http://abcnews.com',
'abcgonews': 'http://abcnews.go.com',
'nyt': 'http://nytimes.com',
'washington-post': 'http://washingtonpost.com',
'us-news': 'http://www.usnews.com',
'msn': 'http://msn.com',
'pbs': 'http://www.pbs.org',
'nbc-news': 'http://www.nbcnews.com',
'enquirer': 'http://www.nationalenquirer.com',
'la-times': 'http://www.latimes.com'
}

# =============================================================================
# Clean the donloaded content to remove HTML, CSS, and Javascript code.
# =============================================================================
def clean_html(html):
    # First we remove inline JavaScript/CSS:
    pg = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    pg = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", pg)
    # Next we can remove the remaining tags:
    pg = re.sub(r"(?s)<.*?>", " ", pg)
    # Finally, we deal with whitespace
    pg = re.sub(r"&nbsp;", " ", pg)
    pg = re.sub(r"&rsquo;", "'", pg)
    pg = re.sub(r"&ldquo;", '"', pg)
    pg = re.sub(r"&rdquo;", '"', pg)
    pg = re.sub(r"\n", " ", pg)
    pg = re.sub(r"\t", " ", pg)
    pg = re.sub(r" ", " ", pg)
    pg = re.sub(r" ", " ", pg)
    pg = re.sub(r" ", " ", pg)   
    return pg.strip()

# =============================================================================
# Get news URLS    
# =============================================================================
def newsapi_get_urls(search_words, agency_urls):
    if len(search_words)==0 or agency_urls==None:
        return None
    print("Searching agencies for pages containing:", search_words)
    # This is my API key, each user must request their own
    # API key from https://newsapi.org/account
    api = NewsApiClient(api_key='444171d89d544b2da002bb61fe78833a')
    api_urls = []
    # Iterate over agencies and search words to pull more url's
    # Limited to 1,000 requests/day - Likely to be exceeded
    for agency in agency_urls:
        domain = agency_urls[agency].replace("http://", "")
        print(agency, domain)
        for word in search_words:
    # Get articles with q= in them, Limits to 20 URLs
            try:
                articles = api.get_everything(q=word, language='en',\
                sources=agency, domains=domain)
            except:
                print("--->Unable to pull news from:", agency, "for", word)
                continue
    # Pull the URL from these articles (limited to 20)
            d = articles['articles']
            for i in range(len(d)):
                url = d[i]['url']
                api_urls.append([agency, word, url])
    df_urls = pd.DataFrame(api_urls, columns=['agency', 'word', 'url'])
    n_total = len(df_urls)
    # Remove duplicates
    df_urls = df_urls.drop_duplicates('url')
    n_unique = len(df_urls)
    print("\nFound a total of", n_total, " URLs, of which", n_unique,\
    " were unique.")
    return df_urls

# =============================================================================
# Get Downloaded Content from URLs obtained
# =============================================================================
def request_pages(df_urls):
    web_pages = []
    for i in range(len(df_urls)):
        u = df_urls.iloc[i]
        url = u[2]
        short_url = url[0:50]
        short_url = short_url.replace("https//", "")
        short_url = short_url.replace("http//", "")
        n = 0
        # Allow for a maximum of 5 download failures
        stop_sec=3 # Initial max wait time in seconds
        while n<3:
            try:
                r = requests.get(url, timeout=(stop_sec))
                if r.status_code == 408:
                    print("-->HTML ERROR 408", short_url)
                    raise ValueError()
                if r.status_code == 200:
                    print("Obtained: "+short_url)
                else:
                    print("-->Web page: "+short_url+" status code:", \
                r.status_code)
                n=99
                continue # Skip this page
            except:
                n += 1
                # Timeout waiting for download
                t0 = time()
                tlapse = 0
                print("Waiting", stop_sec, "sec")
                while tlapse<stop_sec:
                    tlapse = time()-t0
        if n != 99:
        # download failed skip this page
            continue
        # Page obtained successfully
        
        html_page = r.text
        page_text = clean_html(html_page)
        #print(page_text)
        web_pages.append([url, page_text])
    df_www = pd.DataFrame(web_pages, columns=['url', 'text'])
    n_total = len(df_urls)
    # Remove duplicates
    df_www = df_www.drop_duplicates('url')
    n_unique = len(df_urls)
    print("Found a total of", n_total, " web pages, of which", n_unique,\
    " were unique.")
    return df_www

#Search word
search_words = ['Takata']
df_urls = newsapi_get_urls(search_words, agency_urls)
print("Total Articles:", df_urls.shape[0])


print("Agency:", df_urls.iloc[0]['agency'])
print("Search Word:", df_urls.iloc[0]['word'])
print("URL:", df_urls.iloc[0]['url'])


# Download Discovered Pages
df_www = request_pages(df_urls)
# Store in Excel File
df_www.to_excel('/Users/sasha/Desktop/df_www.xlsx')


for i in range(df_www.shape[0]):
    short_url = df_www.iloc[i]['url']
    short_url = short_url.replace("https://", "")
    short_url = short_url.replace("http://", "")
    short_url = short_url[0:60]
    page_char = len(df_www.iloc[i]['text'])
    print("{:<60s}{:>10d} Characters".format(short_url, page_char))
    


