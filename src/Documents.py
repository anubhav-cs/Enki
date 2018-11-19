'''
Created on 26 Jun. 2018

@author: Anubhav Singh
'''
import json
import os.path
from multiprocessing import Pool
from functools import partial
from multiprocessing import cpu_count
import copyreg
import types

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

#===============================================================================
# 
#===============================================================================
class Documents(object):
    '''
    classdocs
    '''

    def __init__(self,corenlp=None):
        '''
        Constructor
        '''

        self.corenlp = corenlp
        #------------------------------------------------ Load the dataset files
        with open(os.path.dirname(__file__) + '/../dataset/documents.json') as f:
            self.documents_json = json.load(f)

#===============================================================================
# 
#===============================================================================
    def generate_tagged_corpus_stanfordnlp(self):
        '''
        Pre-process the documents and returns tokenized and 
        POS tagged sentence tokens
        '''
        #-------------------------------- StanfordCoreNLP properties for parsing
        print(cpu_count())
        #p = Pool(cpu_count())
        #self.documents_json = p.map(self.tokenize_n_tag_stanfordnlp, [1,2,3])

#===============================================================================
# 
#===============================================================================
    def tokenize_n_tag_stanfordnlp(self, doc):
        props={'ner.useSUTime':'false', 'annotators': 'pos,ner',
               'pipelineLanguage':'en','outputFormat':'json'}
        print(doc)
        '''
        doc['tokenized_stanford'] = []
        doc['tagged_stanford'] = []
        for para in doc['text']:
            para_tokenized = []
            para_tagged = []
            #------------------------ Annotate using Stanford corenlp method
            parse = json.loads(self.corenlp.annotate(para, properties=props))
            for sent in parse['sentences']:
                sent_tokenized = []
                sent_tagged = []
                for token in sent['tokens']:
                    sent_tokenized.append(token['word'])
                    sent_tagged.append((token['word'], token['pos']))
                para_tokenized.append(sent_tokenized)
                para_tagged.append(sent_tagged)
            doc['tokenized_stanford'].append(para_tokenized)
            doc['tagged_stanford'].append(para_tagged)
            '''
#===============================================================================
# 
#===============================================================================
    def dump_into_file(self):
        '''
        Dump_into_file as backup
        '''
        
        with open('documents_up.json',"w") as f:
            json.dump(self.documents_json, f)

#===============================================================================
# 
#===============================================================================
    def load_from_backupfile(self):
        '''
        Load from backup file
        '''
        
        with open('documents_up.json') as f:
            self.documents_json = json.load(f)




