
# coding: utf-8

# In[1]:

# import and setup modules we'll be using in this notebook
import logging
import itertools

import numpy as np
import gensim

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))


# In[50]:

from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
from xml.etree.cElementTree import iterparse

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


# In[56]:

doc_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\AllTopics.xml'
stream = iter_wiki(doc_path)
doc_stream = (tokens for tokens in stream)
get_ipython().magic('time id2word_wiki = gensim.corpora.Dictionary(doc_stream)')
print(id2word_wiki)


# In[57]:

# ignore words that appear in less than 20 documents or more than 10% documents
id2word_wiki.filter_extremes(no_below=20, no_above=0.1)
print(id2word_wiki)


# In[58]:


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

# create a stream of bag-of-words vectors
wiki_corpus = WikiCorpus(doc_path, id2word_wiki)
vector = next(iter(wiki_corpus))
print(vector)  # print the first vector in the stream


# In[59]:

# what is the most common word in that first article?
most_index, most_count = max(vector, key=lambda x: x[0])
print(id2word_wiki[most_index], most_count)


# In[60]:

wiki_bow_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\wiki_bow.mm'
get_ipython().magic('time gensim.corpora.MmCorpus.serialize(wiki_bow_path, wiki_corpus)')


# In[61]:

mm_corpus = gensim.corpora.MmCorpus(wiki_bow_path)
print(mm_corpus)


# In[62]:

clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)  # use fewer documents during training, LDA is slow
# ClippedCorpus new in gensim 0.10.1
# copy&paste it from https://github.com/piskvorky/gensim/blob/0.10.1/gensim/utils.py#L467 if necessary (or upgrade your gensim)
get_ipython().magic('time lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=3, id2word=id2word_wiki, passes=4)')


# In[63]:

_ = lda_model.print_topics(-1)  # print a few most important words for each LDA topic


# In[64]:

get_ipython().magic('time tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=id2word_wiki)')


# In[65]:

get_ipython().magic('time lsi_model = gensim.models.LsiModel(tfidf_model[mm_corpus], id2word=id2word_wiki, num_topics=200)')


# In[66]:

# cache the transformed corpora to disk, for use in later notebooks
wiki_tfidf_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\wiki_tfidf.mm'
wiki_lsa_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\wiki_lsa.mm'
get_ipython().magic('time gensim.corpora.MmCorpus.serialize(wiki_tfidf_path, tfidf_model[mm_corpus])')
get_ipython().magic('time gensim.corpora.MmCorpus.serialize(wiki_lsa_path, lsi_model[tfidf_model[mm_corpus]])')
# gensim.corpora.MmCorpus.serialize('./data/wiki_lda.mm', lda_model[mm_corpus])


# In[ ]:



