'''
Created on 26 Jun. 2018

@author: Anubhav Singh
'''

import Documents
import Dataset
import stanfordcorenlp
import nltk

#===============================================================================
# initialize the Document instance
#===============================================================================
def initialize_document(corenlp, document_obj):
    if document_obj == None:
        document_obj = Documents.Documents(corenlp)
    return document_obj

#===============================================================================
# 
#===============================================================================
def initialize_dataset(corenlp, dataset, fname):
    if dataset == None:
        dataset = Dataset.Dataset(corenlp, fname)
    return dataset

#===============================================================================
# 
#===============================================================================
if __name__ == '__main__':
    #------------------------------------------------- Run and load Stanford NLP
    corenlp = stanfordcorenlp.StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27')
    #------------------------------------- load nltk method to segment sentences
    sent_segmenter = nltk.data.load('tokenizers/punkt/english.pickle')
    #--------------------------------------------------------- Load the datasets
    document_obj = initialize_document(corenlp, None)
    training_dataset_obj = initialize_dataset(corenlp, None, "training.json")
    devel_dataset_obj = initialize_dataset(corenlp, None, "devel.json")
    test_dataset_obj = initialize_dataset(corenlp, None, "testing.json")
    document_obj.generate_tagged_corpus_stanfordnlp()
    print(document_obj.documents_json[0])




#---



