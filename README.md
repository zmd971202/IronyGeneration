# IronyGeneration
## irony_word_extraction
  1. make_vocab.py  
    python make_vocab.py courpus K > vocab  
    k: prints K most frequent vocab items
    
  2. make_attribute_vocab.py  
    python make_attribute_vocab.py [vocab] [corpus1] [corpus2] r  
    corpus1: irony  
    corpus2: non-irony  
    r: threshold ( # in corpus_a  / # in corpus_b )  
  
  3. make_pair.py  
    python3 make_pair.py corpus1 corpus2 attr1 attr2 output  
    corpus1: non-irony  
    corpus2: irony  
    attr1: attr of non-irony (will not be used)  
    attr2: attr of irony  
