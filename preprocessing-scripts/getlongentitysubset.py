#prints stats about long entities in the entire test set.
def get_ent_len_stats(mypath):
    fh = open(mypath, encoding="utf-8")
    mydict = {}  # keys are <1--N> values are number of entities with length 1-N.
    fh.readline()
    tempents = []
    temptoks = []
    entname = ""
    numsents = 0
    for line in fh:
        if line.strip() is not "":
            splits = line.strip().split("\t")
            tok = splits[0]
            tag = splits[3]
            if "B-" in tag:
                if tempents:
                    mydict[len(tempents)] = mydict.get(len(tempents), 0) + 1
                tempents = []  # new entity is starting
                tempents.append(tag)
                temptoks.append(tok)
            elif tempents and "I-" in tag:  # old entity is continuing
                tempents.append(tag)
                temptoks.append(tok)
            else:  # tag has to be O if both thes conditions are not true?
                if tempents:  # so close tempents
                    mydict[len(tempents)] = mydict.get(len(tempents), 0) + 1
                """
                if len(tempents) > 20:
                    print(" ".join(temptoks))
                    print(" ".join(tempents))
                """
                tempents = []
                temptoks = []
            # temptoks.append(word)
        else:
            # if len(tempents) == 0:
            # print(" ".join(temptoks))
            # print(" ".join(tempents))
            numsents += 1
            tempents = []
            temptoks = []
    print(dict(sorted(mydict.items())))

import spacy, stanza
from spacy.tokenizer import Tokenizer
import spacy_stanza
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import sys
import itertools

"""
From a list of entity tags (for a given sentence), gets the length of the longest entity.
"""
def get_max_len(list_of_ent_tags):
    temp = [item for item in list_of_ent_tags if item != 'O']
    maxval = 0
    if temp:
        z = [(x[0], len(list(x[1]))) for x in itertools.groupby(temp)]
        maxval = max(z, key=lambda x:x[1])[1]
    return maxval

"""
Reads an ontonotes test file, and stores it as two lists of lists: one each for sentences (as tokens) and their NER tags.
TODO: Should think about a better data structure here.
"""
def read_file(filepath):
    fh = open(filepath)
    sentences = []
    netags = []
    tempsen = []
    tempnet = []
    for line in fh:
       if line.strip() == "":
          if tempsen and tempnet:
              sentences.append(tempsen)
              netags.append(tempnet)
              tempsen = []
              tempnet = []
       else:
          splits = line.strip().split("\t")
          tempsen.append(splits[0])
          tempnet.append(splits[3]) #change here for 3 col vs 4 col conll format.
    fh.close()
    print("Num sentences in: ", filepath, ":", len(sentences))
    return sentences, netags

mypath = "C:/Users/vajjalas/Downloads/NERProject_Materials/bio/bio-everything/onto.test.ner"
maxlen_const = 2#Choose the entity length here. 2--21

# setting up spacy and stanza
nlp = spacy.load("en_core_web_lg")  # can try with other models later
nlp.tokenizer = Tokenizer(nlp.vocab)

snlp = spacy_stanza.load_pipeline(name="en", tokenize_pretokenized=True)
spacy_netags = []
stanza_netags = []
gold_netags = []

#get_ent_len_stats()
sentences,netags = read_file(mypath)
for i in range(0,len(netags)):
    maxlen = get_max_len(netags[i])
    if maxlen > maxlen_const: #change to > if we need "all entites longer than N etc"
        #print(netags[i])
        #print(sentences[i])
        #add spacy/stanza eval code here along with a comparison. output can be a spreadsheet too, to manually go through
        actual_sen = " ".join(sentences[i])
        doc_spacy = nlp(actual_sen)
        doc_stanza = snlp(actual_sen)

        temp_tags_spacy = []
        temp_tags_stanza = []

        for token in doc_spacy:
            if token.ent_iob_ and token.ent_type_:
                tag = token.ent_iob_ + "-" + token.ent_type_
            else:
                tag = token.ent_iob_
            temp_tags_spacy.append(tag)

        for token in doc_stanza:
            if token.ent_iob_ and token.ent_type_:
                tag = token.ent_iob_ + "-" + token.ent_type_
            else:
                tag = "O"

            temp_tags_stanza.append(tag)

        spacy_netags.append(temp_tags_spacy)
        stanza_netags.append(temp_tags_stanza)
        gold_netags.append(netags[i])


print("Classification report for Spacy NER: ")
print(classification_report(gold_netags, spacy_netags, digits=4))

print("Classification report for Stanza NER: ")
print(classification_report(gold_netags, stanza_netags, digits=4))