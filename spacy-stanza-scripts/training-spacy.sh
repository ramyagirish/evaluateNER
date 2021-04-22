# training NER models with spacy CLI, for the 4 genres of text in OntoNotes
python3 -m spacy train en train-news spacy-format/train/onto.news.ner.json spacy-format/development/onto.news.ner.json -G -p ner
python3 -m spacy train en train-bc spacy-format/train/onto.bc.ner.json spacy-format/development/onto.bc.ner.json -G -p ner
python3 -m spacy train en train-tc spacy-format/train/onto.tc.ner.json spacy-format/development/onto.tc.ner.json -G -p ner
python3 -m spacy train en train-wb  spacy-format/train/onto.wb.ner.json spacy-format/development/onto.wb.ner.json -G -p ner
