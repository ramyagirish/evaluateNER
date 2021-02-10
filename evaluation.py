import spacy, stanza
from spacy_stanza import StanzaLanguage
from spacy.tokenizer import Tokenizer

from sklearn.metrics import make_scorer,confusion_matrix
from sklearn.metrics import f1_score,classification_report

import sys

#TODO: Clean this up and wrap into functions. 


#setting up spacy and stanza 
nlp = spacy.load("en_core_web_sm") #can try with other models later
nlp.tokenizer = Tokenizer(nlp.vocab)

stan = stanza.Pipeline(lang="en", tokenize_pretokenized=True)
snlp = StanzaLanguage(stan)

print("Models loaded, and they assume whitespace tokenized text")

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
          tempnet.append(splits[2])
    fh.close()
    print("Num sentences in: ", filepath, ":", len(sentences))
    return sentences, netags


#source for this function: https://gist.github.com/zachguo/10296432
"""pretty print for confusion matrixes"""
def print_cm(cm, labels):
    print("\n")
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        sum = 0
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            sum =  sum + int(cell)
            print(cell, end=" ")
        print(sum) #Prints the total number of instances per cat at the end.



gold_sen, gold_ner = read_file(sys.argv[1])
#"bio/test/onto.bn.ner"

matches = 0
mis_matches = 0

#Making a flat list of NE tags, for evaluations later. 
flat_ne_tags = [ne for gold_ner_sen in gold_ner for ne in gold_ner_sen]

label_list = list(set(flat_ne_tags)) #for printing confusion matrix
spacy_netags = [] #will contain spacy preds
stanza_netags = [] #will contain stanza preds

for sen in gold_sen:
   actual_sen = " ".join(sen)
   doc_spacy = nlp(actual_sen)
   doc_stanza = snlp(actual_sen)

   for token in doc_spacy:
      if token.ent_iob_ and token.ent_type_:
        tag = token.ent_iob_+"-"+token.ent_type_
      else:
        tag = token.ent_iob_
      spacy_netags.append(tag)

   for token in doc_stanza:
      if token.ent_iob_ and token.ent_type_:
        tag = token.ent_iob_+"-"+token.ent_type_
      else:
        tag = "O"

      stanza_netags.append(tag)

   if spacy_netags == stanza_netags:
      matches = matches+1
   else:
      mis_matches = mis_matches+1

print(len(flat_ne_tags), len(spacy_netags), len(stanza_netags))
   
print("***Basic stats: ****")
print("Num sentences: ", len(gold_sen), "in this genre: ", sys.argv[1].split("onto.")[1])
print("Num. predictions where stanza and spacy match exactly: ", matches)
print("Num. predictions where there is a difference between stanza and spacy: ", mis_matches)

print("Classification report for Spacy NER: ")
print(classification_report(flat_ne_tags, spacy_netags))

print("Classification report for Stanza NER: ")
print(classification_report(flat_ne_tags, stanza_netags))

print("Confusion matrix for Stanza NER: ")
print_cm(confusion_matrix(flat_ne_tags, stanza_netags), labels=label_list)

print("Confusion matrix for Spacy NER: ")
print_cm(confusion_matrix(flat_ne_tags, spacy_netags), labels=label_list)
