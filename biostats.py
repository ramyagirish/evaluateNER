import os
dir = "C:/Users/vajjalas/Downloads/NERProject_Materials/bio/bio/"
subdirs = ["train", "test", "development"]
filepaths = ["onto.bc.ner", "onto.bn.ner", "onto.mz.ner", "onto.nw.ner", "onto.tc.ner", "onto.wb.ner"]

mydict = {} #keys are <NE,tag> tuples, values are their counts.
for subdir in subdirs:
    for afile in filepaths:
        fh = open(os.path.join(dir, subdir, afile), encoding="utf-8")
        fh.readline()
        tempsen = []
        tempnet = []
        for line in fh:
            if line.strip() is not "":
              splits = line.strip().split("\t")
              mystr = splits[0]
              tag = splits[3].replace("B-","").replace("I-","")
              if tag is 'O':
                  if tempnet:
                    mytuple = (" ".join(tempsen), tempnet[0])
                    mydict[mytuple] = mydict.get(mytuple, 0) + 1
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
                    tempsen = [mystr]
                    tempnet = [tag]
            else:
                if tempsen:
                    mytuple = (" ".join(tempsen), tempnet[0])
                    mydict[mytuple] = mydict.get(mytuple, 0) + 1
                    tempsen = []
                    tempnet = []

        fh.close()

        sorteddict = sorted(mydict.items(), key=lambda x: x[1], reverse=True)
        output_path = open(subdir+"-"+afile+"-dist.txt", "w", encoding="utf-8")
        for key,val in sorteddict:
            output_path.write(str(key) + "\t"+str(val))
            output_path.write("\n")
        output_path.close()
