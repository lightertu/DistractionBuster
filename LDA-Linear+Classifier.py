
# coding: utf-8

# In[ ]:

# imports:
import time;
import urllib.request
import csv
import os
import itertools
import logging
import json
import numpy as np
import gensim
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
from xml.etree.cElementTree import iterparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO
model_name="LDA-Linear Classifier"
id2word_wiki=gensim.corpora.Dictionary()

# In[ ]:

# List Categories:
#category_list = ["Mathematics","Technology","Music"]
category_list = ["Mathematics","Technology","Music","History","Geography","Arts","Health","Nature","Religion","Literature"]
test_folder =  "./test/"
root_folder = "./Simplex/"
list.sort(category_list)
folder_name=''.join([x[0]+x[1]+x[2] for x in category_list])
root_folder='./'+folder_name+'/'

print("Checking folder : "+folder_name)
if not os.path.exists(root_folder):
    os.mkdir(root_folder)
if not os.path.exists(root_folder+model_name):
    os.mkdir(root_folder+model_name)

    

wiki_bow_path = root_folder+'wiki_bow.mm'
wiki_dict_path = root_folder+'wiki_dict.dict'


# In[ ]:

# Download Page Ids:
#https://petscan.wmflabs.org/?language=en&project=wikipedia&depth=1&format=csv&categories=mathematics&doit=Do it!
print ("Scanning CSV")
for cat in category_list:
    if not os.path.exists(root_folder+cat+".csv"):
        url="https://petscan.wmflabs.org/?language=en&project=wikipedia&depth=1&format=csv&doit=Do%20it!&categories="+cat
        urllib.request.urlretrieve(url, root_folder+cat+".csv")
        print("Downloading ",cat+".csv")
    


# In[ ]:

# CSV to XML Download data:

def get_data(ids,outputFile):
    url="https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=xml&pageids="+ids
    req = urllib.request.urlopen(url)
    if req.getcode() == 200:
        soup = BeautifulSoup(req.read(), 'html.parser')
        s = soup.find_all('page')
        for si in s:
            outputFile.write(str(si))

def batch_train(file):
    outputFile = open(file.replace(".csv",".xml"), 'a', encoding="utf8")
    outputFile.write("<pages>")
                    
    csvReader = csv.reader(open(file,'r'))
    totalRecords = sum(1 for row in csv.reader(open(file,'r',encoding="UTF-8")) )
    print (file.replace(".csv",".xml"),totalRecords)
    start = 0
    end = start + 50
   
    while (start <= totalRecords):
        pageIds = ""
        for row in itertools.islice(csv.reader(open(file,'r',encoding="UTF-8")),start,end):
            pageIds = pageIds + row[2] + "|"
        
        get_data(pageIds,outputFile)
        start = end + 1
        end = start + 50
        if end> totalRecords:
            end=totalRecords
            
    outputFile.write("</pages>")
    

for path, subdirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith('.csv'):
            if not os.path.exists(root_folder+file.replace(".csv",".xml")):
                print("Downloading ",root_folder+file.replace(".csv",".xml"))
                batch_train(path + file)

print ("Done downloading")



# In[ ]:

# Train Model
def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))

def my_extract_pages(f):
    elems = (elem for _, elem in iterparse(f, events=("end",)))
    page_tag = "rev"
    for elem in elems:
        if elem.tag == page_tag and elem.text != None:
            text = elem.text
            yield text
            elem.clear()

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for text in my_extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50:
            continue  # ignore short articles and various meta-articles
        yield tokens
        


# In[ ]:

print("Training Model")
class WikiCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).
        
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs
    
    def __iter__(self):
        for tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs

if not (os.path.exists(wiki_bow_path) and  os.path.exists(wiki_dict_path)):
    for path, subdirs, files in os.walk(root_folder):
        del subdirs[:]
        for file in files:
            if file.endswith('.xml'):
                doc_path = path + file
                stream = iter_wiki(doc_path)
                doc_stream = (tokens for tokens in iter_wiki(doc_path))
                id2word_wiki.merge_with(gensim.corpora.Dictionary(doc_stream))

    id2word_wiki.filter_extremes(no_below=10, no_above=0.1)
    
    # create a stream of bag-of-words vectors
    wiki_corpus = WikiCorpus(doc_path, id2word_wiki)
    
    id2word_wiki.save(wiki_dict_path) 

    gensim.corpora.MmCorpus.serialize(wiki_bow_path, wiki_corpus)

id2word_wiki = gensim.corpora.Dictionary.load(wiki_dict_path)
mm_corpus = gensim.corpora.MmCorpus(wiki_bow_path)
clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000) 
lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=len(category_list), id2word=id2word_wiki, passes=10, alpha='asymmetric')
print("done")


# In[ ]:

y=[]
X=[]

def get_values(l):
    g=[0]*len(category_list)
    for i in l:
        g[i[0]]=i[1]        
    return g


for path, subdirs, files in os.walk(root_folder):
    del subdirs[:]
    for file in files:
        if file.endswith('.xml'):
            doc_path = path + file
            topic = file.replace(".xml","")
            
            test_doc = [tokens for tokens in iter_wiki(doc_path)]
            X+=[get_values(lda_model[id2word_wiki.doc2bow(tokens)]) for tokens in test_doc]
            y+=[category_list.index(topic)]*len(test_doc)

print("Feeding data to SVC")
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)


# In[ ]:

# Test 
results=""
def getPart(testFile):
    test_doc = [tokens for tokens in iter_wiki(testFile)]
    part = [get_values(lda_model[id2word_wiki.doc2bow(tokens)]) for tokens in test_doc]
    return part

print("Testing results")
for path, subdirs, files in os.walk(test_folder):
    for file in files:
        if file.endswith('.xml'):
            doc_path = test_folder + file
            results+=file.replace(".xml","")+":  "
            
            part=getPart(doc_path)
            for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
                 #print(category_list[clf.predict(part)[0]])
                 results+=category_list[clf.predict(part)[0]]
                 results += "\n\n"
                
outputFile = open(root_folder+model_name+"/"+model_name+str(len(category_list))+".txt", 'w', encoding="utf8")
outputFile.write(results)
print(" Done ")


# In[ ]:



