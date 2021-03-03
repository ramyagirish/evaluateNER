# training NER models with spacy CLI, for the 4 genres of text in OntoNotes
echo "Train on news, test on news"
python3 -m spacy evaluate train-news/model-best/ spacy-format/test/onto.news.ner.json
echo "Train on news, test on bc"
python3 -m spacy evaluate train-news/model-best/ spacy-format/test/onto.bc.ner.json
echo "Train on news, test on tc"
python3 -m spacy evaluate train-news/model-best/ spacy-format/test/onto.tc.ner.json
echo "Train on news, test on wb"
python3 -m spacy evaluate train-news/model-best/ spacy-format/test/onto.wb.ner.json
