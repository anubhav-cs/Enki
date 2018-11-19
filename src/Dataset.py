'''
Created on 28 Jun. 2018

@author: Anubhav Singh
'''

import enchant
import os.path
import json
import nltk
from multiprocessing import Pool
from multiprocessing import cpu_count

#===============================================================================
# 
#===============================================================================
class Dataset(object):
    '''
    classdocs
    '''

    d = enchant.Dict("en_US")

    def __init__(self, corenlp, fname):
        '''
        Constructor
        '''
        
        
        self.corenlp = corenlp
        with open(os.path.dirname(__file__) + '/../dataset/' + fname) as f:
            self.dataset_json = json.load(f)

#===============================================================================
# 
#===============================================================================
    def generate_taggged_corpus_nltk(self):
        p = Pool(cpu_count())
        data = self.dataset_json

        args_list = []
        for ques in data:
            args_list.append((ques, nltk, 'nltk'))
        self.dataset_json = p.map(self.tokenize_n_tag, args_list)

#===============================================================================
# 
#===============================================================================
    def generate_tagged_corpus_stanfordnlp(self):
        p = Pool(cpu_count())
        data = self.dataset_json
        args_list = []
        for ques in data:
            args_list.append((ques, self.corenlp, 'stanfordnlp'))
        self.dataset_json = p.map(self.tokenize_n_tag, args_list)

#===============================================================================
# 
#===============================================================================
    def tokenize_n_tag(self, ques, lib, libname):
        d = self.d
        #-------------------------------------------------------------- Tokenize
        text = nltk.word_tokenize(ques['question'])
        sent_spellc = []
        #-------------------------------------------------------- Assign POS tag
        tagged_text= lib.pos_tag(text)
        for tagged_word, word in zip(tagged_text,text):
            #-------------------------- Check if the word's spelling is correct'
            if (d.check(word)==False):
                for suggest in d.suggest(word):
                    # Replace the word with word suggested by enchant if, 
                    # POS tag remains same
                    if tagged_word[1] == nltk.pos_tag([suggest])[0][1]:
                        word = suggest
                        break
            sent_spellc.append(word)
        ques['tokenized_'+ libname] = sent_spellc
        tagged_text= nltk.pos_tag(sent_spellc)
        ques['tagged_' + libname] = tagged_text
        return ques
