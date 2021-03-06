import os, json, pprint, collections

dir = "C:/Users/vajjalas/Downloads/NERProject_Materials/bio/bio/"
subdirs = ["train", "test", "development"]
filepaths = ["onto.bc.ner", "onto.bn.ner", "onto.mz.ner", "onto.nw.ner", "onto.tc.ner", "onto.wb.ner"]
maindir = "basicstats/"

"""
Get stats on number of entities per sentence.
input: bio format data in "dir"
output: dict showing num sentences with 1-N entities, for each partition.
"""
def get_sent_ent_counts():
    for subdir in subdirs:
        for afile in filepaths:
            fh = open(os.path.join(dir, subdir, afile), encoding="utf-8")
            mydict = {}  # keys are <1--N> values are number of sentences with 1-N entities.
            fh.readline()
            tempents = []
            temptoks = []
            numsents = 0
            for line in fh:
                if line.strip() is not "":
                    splits = line.strip().split("\t")
                    word =splits[0]
                    tag = splits[3]
                    if "B-" in tag:
                        tempents.append(tag)
                    temptoks.append(word)
                else:
                    #if len(tempents) == 0:
                        #print(" ".join(temptoks))
                        #print(" ".join(tempents))
                    numsents +=1
                    mydict[len(tempents)] = mydict.get(len(tempents),0)+1
                    tempents = []
                    temptoks =[]
            print("For ", subdir+"/"+afile+ " : num. sents: ", numsents, " and sent stats: ")
            print(dict(sorted(mydict.items())))

"""
Get stats about number of NEs for varying length (in tokens)
input: bio
output: {number:number of entities with that length}
"""
def get_ent_len_stats():
     for subdir in subdirs:
            for afile in filepaths:
                fh = open(os.path.join(dir, subdir, afile), encoding="utf-8")
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
                            tempents = [] #new entity is starting
                            tempents.append(tag)
                            temptoks.append(tok)
                        elif tempents and "I-" in tag: #old entity is continuing
                            tempents.append(tag)
                            temptoks.append(tok)
                        else: #tag has to be O if both thes conditions are not true?
                            if tempents: #so close tempents
                                mydict[len(tempents)] = mydict.get(len(tempents), 0) + 1
                            if len(tempents) > 20:
                                print(" ".join(temptoks))
                                print(" ".join(tempents))
                            tempents = []
                            temptoks = []
                        #temptoks.append(word)
                    else:
                        # if len(tempents) == 0:
                        # print(" ".join(temptoks))
                        # print(" ".join(tempents))
                        numsents += 1
                        tempents = []
                        temptoks = []
                print("For ", subdir + "/" + afile + " : num. sents: ", numsents, " and ent len stats: ")
                print(dict(sorted(mydict.items())))

"""
Gets Stats of the form "('Rumsfeld', 'PERSON')	57" for a given BIO formatted file. 
Why?: Sometimes, the same word may be tagged differently!
"""
def get_entity_pair_counts():
    for subdir in subdirs:
        for afile in filepaths:
            fh = open(os.path.join(dir, subdir, afile), encoding="utf-8")
            mydict = {}  # keys are <NE,tag> tuples, values are their counts.
            ents = []
            fh.readline()
            tempsen = []
            tempnet = []
            numsens = 0
            for line in fh:
                if line.strip() is not "":
                  splits = line.strip().split("\t")
                  mystr = splits[0]
                  tag = splits[3].replace("B-","").replace("I-","")
                  if tag is 'O':
                      if tempnet:
                        mytuple = (" ".join(tempsen), tempnet[0])
                        mydict[mytuple] = mydict.get(mytuple, 0) + 1
                        ents.append(" ".join(tempsen))
                      tempsen = []
                      tempnet = []
                  else:
                      if tempnet and tag == tempnet[-1]:
                        tempsen.append(mystr)
                        tempnet.append(tag)
                      else:
                        if tempnet:
                            mytuple = (" ".join(tempsen), tempnet[0])
                            mydict[mytuple] = mydict.get(mytuple, 0) + 1
                            ents.append(" ".join(tempsen))
                        tempsen = [mystr]
                        tempnet = [tag]
                else:
                    numsens += 1
                    if tempsen:
                        mytuple = (" ".join(tempsen), tempnet[0])
                        mydict[mytuple] = mydict.get(mytuple, 0) + 1
                        ents.append(" ".join(tempsen))
                        tempsen = []
                        tempnet = []

            fh.close()
            total_ents = len(ents)
            total_entcat_pairs = len(mydict)
            sorteddict = sorted(mydict.items(), key=lambda x: x[1], reverse=True)
            output_path = open("basicstats/"+subdir+"-"+afile+"-dist.txt", "w", encoding="utf-8")
            print("Total entities and unique entity-category pairs in ", subdir+ "-" + afile +":  ",
                  str(total_ents), "\t", str(total_entcat_pairs))

            for key,val in sorteddict:
                output_path.write(str(key) + "\t"+str(val))
                output_path.write("\n")
            output_path.close()


"""
Get stats of the form {Entity: {Tag: Count}} form, taking previous function's output as input.
Why?: I initially did not have a clear idea of what I wanted to see. I now want the stats in another format :-)
"""
def get_entity_level_stats():
    #initial form: ('Iran', 'PERSON')	1, ('Iran', 'GPE')	49 etc
    #final form: {'Iran': {'Person':1, 'GPE', '49'}}
    for subdir in subdirs:
        for afile in filepaths:
            fh = open(maindir+subdir+"-"+afile+"-dist.txt", encoding="utf-8")
            fhw = open(maindir+subdir+"-"+afile+"-dist-format2.txt", "w", encoding="utf-8")
            mydict = {}
            for line in fh:
                temp = line.strip().split("\t")
                (entity, tag) = eval(temp[0])
                entity_tag_count = temp[1]
                if not entity in mydict.keys():
                    mydict[entity] = {}
                mydict[entity][tag] = entity_tag_count


            fhw.write(json.dumps(mydict, indent=4))
            fhw.close()
            fh.close()

"""
Get NE category level stats for train/test/dev splits, for all partitions of the data (6)
"""
def get_cat_level_stats():
    master_dict = {} #{split: {genre: {tag: count}}}
    for subdir in subdirs:
        master_dict[subdir] = {}
        for afile in filepaths:
            fh = open(maindir + subdir + "-" + afile + "-dist.txt", encoding="utf-8")
            genre = afile.replace("onto.", "").replace(".ner", "")
            tags_dict = {}
            for line in fh:
                temp = line.strip().split("\t")
                (entity, tag) = eval(temp[0])
                entity_tag_count = temp[1]
                tags_dict[tag] = tags_dict.get(tag,0) +1
            fh.close()
            master_dict[subdir][genre] = tags_dict
    return master_dict

def print_ent_stats_table():
    master_dict = get_cat_level_stats()
    necats = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
              "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT",
              "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
    genres = ["bc", "bn", "mz", "nw", "tc", "wb"]
    fw = open("basicstats/necounts.csv", "w")
    fw.write("necats,"+ ",".join(genres)+"\n")

    for subdir in subdirs:
        fw.write(subdir.upper() + "\n\n")
        for cat in necats:
            templist = []
            for genre in genres:
                if cat in master_dict[subdir][genre]:
                    templist.append(str(master_dict[subdir][genre][cat]))
                else:
                    templist.append("0")
            fw.write(cat+","+",".join(templist)+"\n")
        fw.write("\n\n")


    fw.close()

def get_entity_length_stats():
    master_dict = {} #{split: {genre: {entity: {token_length:count}}}}}
    for subdir in subdirs:
        master_dict[subdir] = {}
        for afile in filepaths:
            fh = open(maindir + subdir + "-" + afile + "-dist.txt", encoding="utf-8")
            genre = afile.replace("onto.", "").replace(".ner", "")
            tags_dict = {}
            for line in fh:
                temp = line.strip().split("\t")
                (entity, tag) = eval(temp[0])
                entity_tag_count = temp[1]
                tags_dict[tag] = tags_dict.get(tag,0) +1
            fh.close()
            master_dict[subdir][genre] = tags_dict
    return master_dict

"""
Return the number of unique entities of a given category, for a given file
"""
def get_all_ents_list(filepath, cat):
    allents = []
    fh = open(filepath, encoding="utf-8")
    for line in fh:
      if "\t" in line:
        splits = line.strip().split("\t")
        entity = splits[0]
        tag = splits[3]
        if cat in tag:
            allents.append(entity)
    fh.close()
    return allents


#get_entity_pair_counts()
#get_entity_level_stats()
#get_cat_level_stats()
#print_ent_stats_table()

#get_sent_ent_counts()
#get_ent_len_stats()

filename_train = "C:/Users/vajjalas/Downloads/NERProject_Materials/bio/bio-everything/onto.train.ner"
gpestrain = get_all_ents_list(filename_train, "GPE")
perstrain = get_all_ents_list(filename_train, "PERSON")
uniqents_GPE_train = set(gpestrain)
uniqents_PER_train = set(perstrain)

filename_test = "C:/Users/vajjalas/Downloads/NERProject_Materials/bio/bio-everything/onto.test.ner"

gpestest = get_all_ents_list(filename_test, "GPE")
perstest = get_all_ents_list(filename_test, "PERSON")
uniqents_GPE_test = set(gpestest)
uniqents_PER_test = set(perstest)

print("Everything is token level stats!")

print("Total entity tokens for GPE in train and test: ", len(gpestrain), len(gpestest))
print("Total unique entity tokens for GPE in train and test: ",len(uniqents_GPE_train), len(uniqents_GPE_test))

print("Total entity tokens for PER in train and test: ", len(perstrain), len(perstest))
print("Total unique entity tokens for PER in train and test: ",len(uniqents_PER_train), len(uniqents_PER_test))

print("Percentage of test PER tokens that overlaps with train PER: ",
       len(uniqents_PER_test.intersection(uniqents_PER_train))/len(uniqents_PER_test))

print("Percentage of test GPE tokens that overlaps with train GPE: ",
       len(uniqents_GPE_test.intersection(uniqents_GPE_train))/len(uniqents_GPE_test))
