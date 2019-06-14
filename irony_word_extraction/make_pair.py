import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from tqdm import tqdm
import random
import nltk
# python3 xx.py corpus1 corpus2 attr1 attr2 output
# corpus1 non-irony
# corpus2 irony


def common_token(str1, str2):
    word1 = set(str1.split())
    word2 = set(str2.split())
    intersection = word1.intersection(word2)
    return len(intersection) * 2 / (len(word1) + len(word2))



# corpus1 = []
with open(sys.argv[1], 'r') as f:
    corpus1 = [s.strip() for s in f.read().split('\n')]


with open(sys.argv[2], 'r') as f:
    corpus2 = [s.strip() for s in f.read().split('\n')]

with open(sys.argv[3], 'r') as f:
    attr1 = [tok.strip() for tok in f]

with open(sys.argv[4], 'r') as f:
    attr2 = [tok.strip() for tok in f]

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_mat = tfidf_vectorizer.fit_transform(corpus1).toarray()

sent_lst = []
candidate_lst = []

random.shuffle(corpus1)
for sent in tqdm(corpus1[:2000]):
    # if len(sent_lst) > 10:
    #     break
    best = ''
    best_score = 0.0
    for candidate in corpus2:
        temp_score = common_token(sent, candidate)
        # print(temp_score)
        if temp_score > best_score:
            best = candidate
            best_score = temp_score
    sent_lst.append(sent)
    candidate_lst.append(best)
    # print('{}###$###{}'.format(sent, best))

count = 0
with open(sys.argv[5], 'w') as f:
    for sent, candidate in zip(sent_lst, candidate_lst):
        opposite_attr = ''
        tags = nltk.pos_tag(candidate.split())
        for tok, tag in tags:
            if tok in attr2 and tok != 'not' and tok != 'no' and ('JJ' in tag):
                opposite_attr = tok
                break
        # assert opposite_attr != ''
        if opposite_attr != '':
            count += 1
            f.write('{}###$###{}'.format(sent, opposite_attr))
            f.write('\n')
            # print('{}###$###{}'.format(sent, opposite_attr))
    # print(count)
