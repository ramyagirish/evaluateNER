"""
Evaluation of NER performance by sentence length.
What I plan to do: build a sheet with columns:
senid, senlen, num entities, num correct_spacy, num correct_stanza
- this can later be used to plot the effect of sentence length on entity prediction performance.
"""

from evaluation import read_file
import spacy, stanza, spacy_stanza
from spacy.tokenizer import Tokenizer
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import sys
import itertools
from seqeval.metrics.sequence_labeling import get_entities

"""
https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py
>>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
"""

mypath = "C:/Users/vajjalas/Downloads/NERProject_Materials/bio/bio-everything/onto.test.ner"
gold_sen, gold_ner = read_file(mypath)
print("BIO formatted test file is read")

# setting up spacy and stanza
nlp = spacy.load("en_core_web_trf")  # can try with other models later
nlp.tokenizer = Tokenizer(nlp.vocab)
snlp = spacy_stanza.load_pipeline(name="en", tokenize_pretokenized=True)
print("Models loaded, and they assume whitespace tokenized text")

#output file
fh = open("results/senlen-stats.csv", "w", encoding="utf-8")
fh.write("sentid, sentlen, numents, percentcorrect_spacy, percentcorrect_stanza")
fh.write("\n")

for index in range(0,len(gold_sen)):
    actual_sen = gold_sen[index]
    actual_nerseq = gold_ner[index]
    doc_spacy = nlp(" ".join(actual_sen))
    doc_stanza = snlp(" ".join(actual_sen))

    #spacy NER tagged output as a list
    temp_tags_spacy = []
    for token in doc_spacy:
        if token.ent_iob_ and token.ent_type_:
            tag = token.ent_iob_ + "-" + token.ent_type_
        else:
            tag = token.ent_iob_
        temp_tags_spacy.append(tag)

    #stanza NER tagged output as a list
    temp_tags_stanza = []
    for token in doc_stanza:
        if token.ent_iob_ and token.ent_type_:
            tag = token.ent_iob_ + "-" + token.ent_type_
        else:
            tag = "O"
        temp_tags_stanza.append(tag)

    #use seqeval's get_entities function to compare entities by sentence
    gold = get_entities(gold_ner[index])
    spacyents = get_entities(temp_tags_spacy)
    stanzaents = get_entities(temp_tags_stanza)

    #write to output
    if len(gold) > 0:
        temp = [index,len(actual_sen),len(gold),len(set(gold).intersection(set(spacyents)))/len(gold),
             len(set(gold).intersection(set(stanzaents)))/len(gold)]
    else:
        temp = [index, len(actual_sen), len(gold), 0, 0]
    fh.write(','.join(map(str, temp)))
    fh.write("\n")
    #print(temp)

fh.close()
print("Done")