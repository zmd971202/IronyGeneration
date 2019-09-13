from multiprocessing import Pool
import nltk
from langdetect import detect
import os
import random

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import string
from collections import Counter

import sys

if len(sys.argv) == 3:
    src = sys.argv[1]
    tgt = sys.argv[2]
else:
    exit(0)

lines = open(src, 'r').read().split('\n')

print('extracting sents...')
print('read {} lines'.format(len(lines)))


length = len(lines)
num_proc = 10
part = length // num_proc


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    #unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

text_processor1 = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

remove_words = ['<url>', '<hashtag>', '</hashtag>', '<allcaps>', '<elongated>', '<repeated>',
                '<emphasis>', '<censored>', '</allcaps>']


p1 = ['.', '?', '!']
p2 = [',', ';']
p3 = ['\'']

replace_lines = open('data/replace_words.txt', 'r').read().split('\n')
replace_words = {}
for i, l in enumerate(replace_lines):
    parts = l.split('\t', 1)
    w = parts[0].lower()
    r = ' '.join(SocialTokenizer(lower=True).tokenize(parts[1].strip()))
    replace_words[w] = r.lower()

    
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def f(x):
    if x == num_proc - 1:
        ls = lines[x * part:]
    else:
        ls = lines[x * part : x * part + part]
    
    local_new_lines = []
    
    for i, l in enumerate(ls):
        if i % 5000 == 0:
            print('prepared {} lines'.format(i), flush=True)
        
        l = deEmojify(l)
        if len(l) == 0:
            continue
        
        # url
        ind = l.find('http')
        if ind != -1:
            ind1 = l.find('…', ind)
            l = l[:ind] + l[ind:ind1].replace(' ', '')

        words = [w for w in l.split() if 'pic.twitter.com' not in w]
        l = ' '.join(words)
        words = text_processor.pre_process_doc(l)
        words = [w for w in words if w not in remove_words]

        # jumo hashtag
        while len(words) > 0 and '#' in words[-1]:
            words = words[:-1]
        if len(words) == 0:
            continue

        # juzhong hashtag
        words = text_processor1.pre_process_doc(' '.join(words))
        words = [w for w in words if w not in remove_words]

        #
        new_words = []
        if words[0] in p3 or words[0] not in string.punctuation:
            new_words.append(words[0])
        length = len(words)
        for j in range(1, length):
            if (words[j] == words[j - 1] and words[j] in string.punctuation) or words[j] == '…':
                continue
            if words[j] not in string.punctuation:
                new_words.append(words[j])
                continue
            if words[j] in p1:
                new_words.append('.')
            elif words[j] in p2:
                new_words.append(',')
            elif words[j] in p3:
                new_words.append(words[j])

        if len(new_words) == 0:
            continue

        words = new_words
        new_words = []

        length = len(words)   
        if words[length - 1] != 'not':
            new_words.append(words[length - 1])
        for j in range(1, length):
            if words[length - j - 1] == 'not' and words[length - j] == '.':
                continue
            new_words.append(words[length - j - 1])

        # replace
        words = new_words[::-1]
        words = [replace_words[w] if w in replace_words else w for w in words]

        length = len(words)
        if length < 10 or length > 40:
            continue
            

        new_l = ' '.join(words)
        new_l = new_l.replace('<money>', '<number>')
        new_l = new_l.replace('<time>', '<number>')
        local_new_lines.append(new_l)
    
    return local_new_lines
        
if __name__ == '__main__':
    with Pool(num_proc) as p:
        all_returns = p.map(f, [i for i in range(num_proc)])
    
    new_lines = []
    
    for r in all_returns:
        new_lines += r

    open(tgt, 'w', encoding='utf-8').write('\n'.join(new_lines))