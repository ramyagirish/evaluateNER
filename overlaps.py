"""
Calculates entity level (and entity-entity type) level overlaps
between train-dev and train-test sets, within and across genres.
"""
import os

"""
Reads the stats files created from biostats.py/get_entity_pair_counts()
and stores it in an appropriate DS
"""
def read_file(file_path):
    fh = open(file_path, encoding="utf-8")
    mytupslist = []
    for line in fh:
        mytup = eval(line.split("\t")[0])
        mytupslist.append(mytup)
    fh.close()
    return mytupslist

def get_overlap_percent(tuplst1, tuplst2):
    templst =list(set(tuplst1) & set(tuplst2))
    return 100*len(templst)/len(tuplst2)

dir = "basicstats/"
subdirs = ["train", "test", "development"]
filenames = ["onto.bc.ner", "onto.bn.ner", "onto.mz.ner", "onto.nw.ner", "onto.tc.ner", "onto.wb.ner"]
for filename in filenames:
    mytupstrain = read_file(dir+subdirs[0]+"-"+filename+"-dist.txt")
    mytupsdev = read_file(dir+subdirs[1]+"-"+filename+"-dist.txt")
    mytupstest = read_file(dir+subdirs[2]+"-"+filename+"-dist.txt")

    print("For the subset: ", filename, " overlap between train and dev is: ",
          get_overlap_percent(mytupstrain, mytupsdev), "percent of dev")
    print("For the subset: ", filename, " overlap between train and test is: ",
          get_overlap_percent(mytupstrain, mytupstest), "percent of test")

#overlap between genres - training data.
print("Printing inter genre overlap among training data")
for filename1 in filenames:
    for filename2 in filenames:
        mytups1 = read_file("basicstats/train-" + filename1 + "-dist.txt")
        mytups2 = read_file("basicstats/train-" + filename2 + "-dist.txt")
        print("The overlap between ", filename1, " and ", filename2, " is: ",
              get_overlap_percent(mytups1, mytups2), "percent of ", filename1)

