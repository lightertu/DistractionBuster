
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


# In[2]:

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


# In[3]:

doc_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\AllTopics.xml'
stream = iter_wiki(doc_path)

for tokens in itertools.islice(iter_wiki(doc_path), 8):
    print (tokens[:10])
doc_stream = (tokens for tokens in iter_wiki(doc_path))
get_ipython().magic('time id2word_wiki = gensim.corpora.Dictionary(doc_stream)')
print(id2word_wiki)


# In[4]:

# ignore words that appear in less than 20 documents or more than 10% documents
id2word_wiki.filter_extremes(no_below=10, no_above=0.1)
print(id2word_wiki)


# In[5]:


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
# print(vector)  # print the first vector in the stream


# In[6]:

# what is the most common word in that first article?
most_index, most_count = max(vector, key=lambda x: x[1])
print(id2word_wiki[most_index], most_count)

print(id2word_wiki[68])

import heapq
print(heapq.nlargest(3, vector, key=lambda x: x[1]))


# In[7]:

wiki_bow_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\wiki_bow.mm'
get_ipython().magic('time gensim.corpora.MmCorpus.serialize(wiki_bow_path, wiki_corpus)')


# In[8]:

mm_corpus = gensim.corpora.MmCorpus(wiki_bow_path)
print(mm_corpus)

print(len([ x for x in iter(mm_corpus)]))


# In[9]:

clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)  # use fewer documents during training, LDA is slow
# ClippedCorpus new in gensim 0.10.1
# copy&paste it from https://github.com/piskvorky/gensim/blob/0.10.1/gensim/utils.py#L467 if necessary (or upgrade your gensim)
get_ipython().magic('time lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=3, id2word=id2word_wiki, passes=6)')


# In[10]:

_ = lda_model.print_topics()  # print a few most important words for each LDA topic


# In[26]:

# evaluate on 1k documents **not** used in LDA training
test1_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\CompSci.xml'
test2_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\Literature.xml'
test3_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\Mathematics.xml'

# doc_stream = [tokens for tokens in iter_wiki(test1_path))  # generator
test_doc_1 = [tokens for tokens in iter_wiki(test1_path)]
part1 = [lda_model[id2word_wiki.doc2bow(tokens)] for tokens in test_doc_1]


topic_dic = {0:0, 1:0, 2:0}

for doc in part1:
    for p in doc:
        topic_dic[p[0]] += p[1]

print(topic_dic)

num_docs = len(part1)

print("Centroid : (", topic_dic[0]/num_docs, ", ", topic_dic[1]/num_docs, ", ", topic_dic[2]/num_docs, ")")

centroid_1 = [(x, topic_dic[x]/num_docs) for x in range(3)]

topic_dic = {0:0, 1:0, 2:0}

for doc in part2:
    for p in doc:
        topic_dic[p[0]] += p[1]

print(topic_dic)

num_docs = len(part2)

print("Centroid : (", topic_dic[0]/num_docs, ", ", topic_dic[1]/num_docs, ", ", topic_dic[2]/num_docs, ")")

centroid_2 = [(x, topic_dic[x]/num_docs) for x in range(3)]

topic_dic = {0:0, 1:0, 2:0}

for doc in part3:
    for p in doc:
        topic_dic[p[0]] += p[1]

print(topic_dic)

num_docs = len(part3)

print("Centroid : (", topic_dic[0]/num_docs, ", ", topic_dic[1]/num_docs, ", ", topic_dic[2]/num_docs, ")")

centroid_3 = [(x, topic_dic[x]/num_docs) for x in range(3)]

print(part1[0])
print(part1[1])

# doc_stream = [tokens for tokens in iter_wiki(test1_path))  # generator
test_doc_2 = [tokens for tokens in iter_wiki(test2_path)]
part2 = [lda_model[id2word_wiki.doc2bow(tokens)] for tokens in test_doc_2]
print(part2[0])
print(part2[1])

# doc_stream = [tokens for tokens in iter_wiki(test1_path))  # generator
test_doc_3 = [tokens for tokens in iter_wiki(test3_path)]
part3 = [lda_model[id2word_wiki.doc2bow(tokens)] for tokens in test_doc_3]
print(part3[0])
print(part3[1])


# In[ ]:




# In[24]:

all_tokens = set()
for tokens in test_doc_1:
    all_tokens |= set(tokens)

print(lda_model[id2word_wiki.doc2bow(list(all_tokens))])


# In[17]:

print(type(part1))


# In[16]:

print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part3)]))
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part2, part3)]))


# In[27]:

# evaluate on 1k documents **not** used in LDA training
test1_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\Test_doc_Ethics.xml'
test2_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\Test_doc_Allegory.xml'
test3_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\Test_doc_AI.xml'



# doc_stream = [tokens for tokens in iter_wiki(test1_path))  # generator
test_doc_1 = [tokens for tokens in iter_wiki(test1_path)]
part1 = [lda_model[id2word_wiki.doc2bow(tokens)] for tokens in test_doc_1]
#print(part1)

# doc_stream = [tokens for tokens in iter_wiki(test1_path))  # generator
test_doc_2 = [tokens for tokens in iter_wiki(test2_path)]
part2 = [lda_model[id2word_wiki.doc2bow(tokens)] for tokens in test_doc_2]

# doc_stream = [tokens for tokens in iter_wiki(test1_path))  # generator
test_doc_3 = [tokens for tokens in iter_wiki(test3_path)]
part3 = [lda_model[id2word_wiki.doc2bow(tokens)] for tokens in test_doc_3]


# In[33]:

print(centroid_1)

print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip([centroid_3], part1)]))
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip([centroid_3], part2)]))
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip([centroid_3], part3)]))
print()
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip([centroid_1], part1)]))
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip([centroid_1], part2)]))
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip([centroid_1], part3)]))
print()
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip([centroid_2], part1)]))
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip([centroid_2], part2)]))
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip([centroid_2], part3)]))


# In[ ]:

get_ipython().magic('time tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=id2word_wiki)')


# In[ ]:

get_ipython().magic('time lsi_model = gensim.models.LsiModel(tfidf_model[mm_corpus], id2word=id2word_wiki, num_topics=3)')


# In[ ]:

# cache the transformed corpora to disk, for use in later notebooks
wiki_tfidf_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\wiki_tfidf.mm'
wiki_lsa_path = 'D:\\Study\Winter-2017\Machine Learning\Project\DistractionBuster\Simplex\Dump\wiki_lsa.mm'
get_ipython().magic('time gensim.corpora.MmCorpus.serialize(wiki_tfidf_path, tfidf_model[mm_corpus])')
get_ipython().magic('time gensim.corpora.MmCorpus.serialize(wiki_lsa_path, lsi_model[tfidf_model[mm_corpus]])')
# gensim.corpora.MmCorpus.serialize('./data/wiki_lda.mm', lda_model[mm_corpus])


# In[ ]:

part1 = [lsi_model[id2word_wiki.doc2bow(tokens)] for tokens in test_doc_1]

part2 = [lsi_model[id2word_wiki.doc2bow(tokens)] for tokens in test_doc_2]

part3 = [lsi_model[id2word_wiki.doc2bow(tokens)] for tokens in test_doc_3]

print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part3)]))
print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part2, part3)]))

