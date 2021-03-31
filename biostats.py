import os, json
dir = "C:/Users/vajjalas/Downloads/NERProject_Materials/bio/bio/"
subdirs = ["train", "test", "development"]
filepaths = ["onto.bc.ner", "onto.bn.ner", "onto.mz.ner", "onto.nw.ner", "onto.tc.ner", "onto.wb.ner"]

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
    maindir = "basicstats/"
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
    maindir = "basicstats/"
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


import pprint

#get_entity_pair_counts()
#get_entity_level_stats()

get_cat_level_stats()
print_ent_stats_table()