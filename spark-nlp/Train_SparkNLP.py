#!/usr/bin/env python
# coding: utf-8

# In[9]:


from pyspark.ml import Pipeline

import sparknlp

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from sparknlp.training import CoNLL
from seqeval.metrics import accuracy_score, f1_score, classification_report

import pandas as pd
import os

# to use GPU 
spark = sparknlp.start()

print("Spark NLP version: ", sparknlp.version())
print("Apache Spark version: ", spark.version)


# In[10]:


bert = BertEmbeddings.pretrained('bert_base_cased', 'en').setInputCols(["sentence",'token']).setOutputCol("bert").setCaseSensitive(True)#.setMaxSentenceLength(512)


# In[11]:


#Takes a regular bio file, converts it to spark-conll format. 
def convert_format(inputpath, outputpath):
    # create the training file
    with open(inputpath) as fp:
        text = fp.readlines()
    text = "".join(text[1:]).split("\n\n") 
    df = pd.DataFrame([x.split('\t') for x in text[1].split('\n')], 
                      columns=["Token","Pos","Pos_special","Entity_label"])
    
    # creating the training data
    conll_lines = "-DOCSTART- -X- -X- -O-\n\n"
    for t in range(len(text)):    
        df = pd.DataFrame([x.split('\t') for x in text[t].split('\n') if len(x.split('\t')) == 4], columns=["Token","Pos","Pos_special","Entity_label"])
        tokens = df.Token.tolist()
        pos_labels = df.Pos.tolist()
        entity_labels = df.Entity_label.tolist()
        for token, pos, label in zip(tokens,pos_labels,entity_labels):
            conll_lines += "{} {} {} {}\n".format(token, pos, pos, label)
        conll_lines += "\n"
        
    with open(outputpath,"w") as fp:
        for line in conll_lines:
            fp.write(line)


# In[14]:


#Takes a regular bio file, converts it to spark-conll, converts it to bert rep, and stores it as a parquet file
def make_ner_ready(inputpath):
    filename = os.path.basename(inputpath)
    tempfolder = "tmp/"
    outputpath = os.path.join(tempfolder,filename+"_tmp")
    convert_format(inputpath,outputpath)
    conllformat = CoNLL().readDataset(spark, outputpath)
    readyData = bert.transform(conllformat)
    readyData.write.mode("Overwrite").parquet(os.path.join(tempfolder,filename+"_pq"))
    print("Wrote to pq")
    

def get_results(myNERModel, readyTestData):
    results = myNerModel.transform(readyTestData).select("sentence","token","label","ner").collect()    
    #test_data.show()
    
    # to find exceptions where no. of labels does not match no. of ners detected
    count = 0
    indices = []
    for i,row in enumerate(results):
        if len(row['label']) != len(row['ner']):
            count += 1
            indices.append(i)

    print(count)
    print(indices)

    exclusion_list = [results[t] for t in indices]
    results = [results[i] for i in range(len(results)) if i not in indices]
    
    tokens = []
    labels = []
    ners = []

    for row in results:
        tokens.append([t['result'] for t in row['token']])
        labels.append([t['result'] for t in row['label']])
        ners.append([t['result'] for t in row['ner']])

    #print(accuracy_score(labels,ners))
    #print(f1_score(labels,ners))

    print(classification_report(labels,ners, zero_division=1,digits=6))

# initialize NER tagger
nerTagger = NerDLApproach().setInputCols(["sentence", "token", "bert"]).setLabelColumn("label").setOutputCol("ner").setMaxEpochs(10).setBatchSize(4).setEnableMemoryOptimizer(True).setRandomSeed(0).setVerbose(1).setValidationSplit(0.2).setEvaluationLogExtended(True).setEnableOutputLogs(True).setIncludeConfidence(True)

make_ner_ready("fold10_dev.ner")
make_ner_ready("fold10_train.ner")
make_ner_ready("fold10_test.ner")
readyTrainingData = spark.read.parquet('tmp/fold10_train.ner_pq')
readyDevData = spark.read.parquet('tmp/fold10_dev.ner_pq')
readyTestData = spark.read.parquet('tmp/fold10_test.ner_pq')

# train the model
myNerModel = nerTagger.fit(readyTrainingData)
get_results(readyTestData)




