# Script to convert BIO format to Spacy's json, to train with spacy. 
# Using bio-combined, which collapses nw, mz, bn into "news"
home="bio-combined/"
output="spacy-format/"
listvar="train/ test/ development/"
for i in $listvar; do
  echo $home$i
  temp=`ls $home$i`
  for j in $temp; do
     echo $home$i$j
     echo $output$i$j
     python3 -m spacy convert $home$i$j -c ner > $output$i$j".json"
     #NOTE: This won't work with Spacy 3.0 anymore. Check the other spacy conversion script.
   done
   done
