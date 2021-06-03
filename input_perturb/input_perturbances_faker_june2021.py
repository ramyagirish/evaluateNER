import csv
import pandas as pd
from faker import Faker
import random

mypath = "C:/Users/vajjalas/Downloads/NERProject_Materials/bio/bio-everything/onto.test.ner"

#anything to edit
mycat = 'GPE' #PERSON, ORG, GPE
#GPE is countries, cities, states; ORG: companies, agencies, institutions;
fake = Faker('en_IE')
myoutput = "perturb_en-ie_gpe.ner"

fh = open(mypath)
fw = open(myoutput, "w", encoding="utf-8")

numcols = 4
for line in fh:
    splits = line.strip().split("\t")
    if len(splits) == 4:
        if splits[3] == 'B-'+mycat:
            #splits[0] = fake.name_female().split()[0]
            splits[0] = random.choice([fake.county(), fake.country(), fake.city()]).split()[0]
        elif splits[3] == 'I-'+mycat:
            #splits[0] = fake.last_name_female().split()[0]
            splits[0] = fake.city_suffix().split()[0]
        print("\t".join(splits))
        fw.write("\t".join(splits))
        fw.write("\n")
    else:
        fw.write("\n")

fh.close()
fw.close()
print("DONE")

#df.Word = words
#df.to_csv("perturb_in.ner",sep="\t",index=False,header=False)