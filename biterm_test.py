'''
WORK IN PROGRESS (DON'T BE A TWIT, FINISH THIS SHIT)
Created by Miles Latham, last update: 30 October 2019.
Email: mbl2161@columbia.edu
Phone: 18022997994
Permissions: this code is the property of Miles Latham.
No other user or develoepr may use/update it without his express
permission.
'''

import csv
import numpy as np
import pyLDAvis
import past
from biterm.utility import vec_to_biterms, topic_summuary
from biterm.btm import oBTM
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

startTime = datetime.now()

def read_file(csv):
    texts = open(csv).read.splitlines()
    return texts

def split_list(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def vectorizer()

if __name__ == "__main__":

    texts = open('quora_challenge.csv').read().splitlines()[:100000]
    print('1. open CSV')
    print(len(texts))
    # vectorize texts
    vec = CountVectorizer(stop_words='english')
    print('2. vectorize texts')
    X = vec.fit_transform(texts).toarray()
    print(len(X))
    # get vocabulary
    vocab = np.array(vec.get_feature_names())
    print('3. get vocabulary')
    print(len(vocab))
    # get biterms
    biterms = vec_to_biterms(X)
    print(len(biterms))
    print('4. get biterms')

    # create btm
    btm = oBTM(num_topics=30, V=vocab)
    print('5. create btm')

    print("\n\n Training Online BTM ..")
    for i in range(0, len(biterms), 50000): # process chunk of 200 texts
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=50)
    topics = btm.transform(biterms)

    print("\n\n Visualize Topics ..")
    #vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
    #pyLDAvis.save_html(vis, 'pyLDA_result.html')

    print("\n\n Topic coherence ..")
    summ_dict = topic_summuary(btm.phi_wz.T, X, vocab, 10)
    with open('topic_coherence.txt', 'w') as f:
        print(summ_dict, file=f)
        f.close()
    with open('questions_by_topic.csv', 'w') as f:
        for i in range(len(texts)):
            f.write("{} (topic: {}), ".format(texts[i], topics[i].argmax()))
        f.close()

print(datetime.now() - startTime)