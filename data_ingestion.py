import time
import pathlib
import os
import pandas as pd
import math
import csv

# function to list files in a folder
def file_list(file_dir,ext):
	'''
	input - 
	file_dir is path of directory
	ext is extension of files we are interested in
	output -  
	list of files in the directory with that extension
	'''
	path = pathlib.Path(file_dir)
	return [str(f) for f in path.rglob("*." + ext)]

# function  to read from .ner files and write  txt files that could be used for computation
def read_write_file(file_dir,folder_list,dest_folder_name,ext):
	'''
	input -
	file_dir is path of directory
	folder_list is used for listing training files, testing files and validation files
	dest_folder_name is the name of destination folder name where the txt files needs to reside
	ext is extension of files we are interested in

	'''
	for folder in folder_list:
        # create file-path for source file and final text file
        source_file_path = os.path.join(file_dir,folder)
        dest_file_path = os.path.join(file_dir,dest_folder_name,folder)

        # create destination folder if it does not exist
        if not(pathlib.Path(dest_file_path).exists()):
            pathlib.Path(dest_file_path).mkdir()

        # list of files that need to be read
        req_files = file_list(source_file_path,ext)
        for f in req_files:
        	# handle that reads file
            fread = open(f,"r")
            # create destination text file
            filename = "onto_" + pathlib.Path(f).name.split(".")[1] + "_ner.txt"
            if not(pathlib.Path(os.path.join(dest_file_path,filename)).exists()):
                pathlib.Path(os.path.join(dest_file_path,filename)).touch()
            # handle that writes file
            fwrite = open(os.path.join(dest_file_path,filename),"+w")
            count = 0
            # tab-limited dataframe
            for line in fread.readlines():
                if line != "\n":
                    if '\n' in line:
                        line = line.replace('\n',"")
                    line = line + '\t' + str(count) + "\n"
                    fwrite.writelines(line)
                else:
                    count += 1


def create_df(filepath):

	# read text tab-delimited file
	df = pd.read_csv(filepath,sep="\t",header=None,names=["Word","POS","DEREP","TYPE","SENT_NO"])
	sentences = []
	entities = []
	entities_type = []
	print("there are {} sentences".format(len(df.groupby("SENT_NO").groups.items())))
	count = 1
	start = time.time()
	# create groups based on sentence number
	for _,v in df.groupby("SENT_NO").groups.items():
	    temp1 = []
	    temp2 = []
	    temp3 = []
	    if count%1000 == 0:
	        print("Done for {} in {} seconds".format(count,time.time()-start))
	    for i,t in enumerate(df.iloc[v,:].Word.tolist()):
	        if i < (len(df.iloc[v,:].Word.tolist())-1):
                if isinstance(df.iloc[v,:].Word.tolist()[i],float) or isinstance(df.iloc[v,:].Word.tolist()[i+1],float):
                    continue
                elif df.iloc[v,:].Word.tolist()[i][0].isalnum() and not(df.iloc[v,:].Word.tolist()[i+1][0].isalnum()):
                    temp1.append(t + df.iloc[v,:].Word.tolist()[i+1])
                    temp2.append(df.iloc[v,:].TYPE.tolist()[i])
                    temp2.append(df.iloc[v,:].TYPE.tolist()[i+1])
                    temp3.append(df.iloc[v,:].Word.tolist()[i])
                    temp3.append(df.iloc[v,:].Word.tolist()[i+1])
                elif not(df.iloc[v,:].Word.tolist()[i][0].isalnum()):
                    continue
                else:
                    temp1.append(t)
                    temp2.append(df.iloc[v,:].TYPE.tolist()[i])
                    temp3.append(df.iloc[v,:].Word.tolist()[i])
            elif i == (len(df.iloc[v,:].Word.tolist())-1):
                if isinstance(df.iloc[v,:].Word.tolist()[i],float):
                    continue
                elif t[0].isalnum():
                    temp1.append(t)
                    temp2.append(df.iloc[v,:].TYPE.tolist()[i])
                    temp3.append(df.iloc[v,:].Word.tolist()[i])
	    # reconstruct the sentence, entity list and corresponding entity type list
	    sentences.append(" ".join(temp1))
	    entities_type.append(temp2)
	    entities.append(temp3)
	    count += 1

	return (sentences, entities_type, entities)


def create_conll_brat(file_dir,folder_list):
	'''
	input -
	file_dir is path of directory
	folder_list is used for listing training files, testing files and validation files
	output -
	a folder bio_brat with the required files
	'''
	for folder in folder_list:
		# create file-path for source file and final text file
        source_file_path = os.path.join(file_dir,folder)
        dest_file_path = os.path.join(file_dir,"bio_brat",folder)

        # create destination folder if it does not exist
        if not(pathlib.Path(dest_file_path).exists()):
            pathlib.Path(dest_file_path).mkdir()

        # list of files that need to be read
        req_files = file_list(source_file_path,"txt")

        count = 1

        for file in req_files:
        	print("for file: {}".format(file))
        	# convert data in text files to sentnces , entities , entity tyeps
        	sentences, entities_type, entities = create_df(file)
        	# dividing txt files to handle n sentences
        	n = 10
        	for k in range(0,len(entities) - n,n):
        		length = 0
				sents = []
				text = ""
				# create columns needed to create brat annotation files
				start  = []
				end = []
				words = []
				tags = []
                for i,ent in enumerate(entities[k:(k+n)]):
                    begin = k + i
                    sent_len = 0
                    text = text  + " ".join(entities[begin]) + " "
                    for j,e in enumerate(entities[begin]):
                        if entities_type[begin][j] != 'O':
                            words.append(e)
                            tags.append(entities_type[begin][j][2:])
                            start.append(length + sent_len)
                            end.append(length + sent_len + len(e))
                        sent_len = sent_len + len(e) + 1
                    length = length + len(ent)  + sum([len(e) for e in entities[begin]])


				file_name = folder + "_text_00" + str(count)
				dest_file_text = os.path.join(dest_file_path,file_name + ".txt")
				dest_file_ann = os.path.join(dest_file_path,file_name + ".ann")
				# create destination text file if it does not exist
		        if not(pathlib.Path(dest_file_text).exists()):
		            pathlib.Path(dest_file_text).touch()
		        # create destination ann file if it does not exist
		        if not(pathlib.Path(dest_file_ann).exists()):
		            pathlib.Path(dest_file_ann).touch()

		        # add text files
		        fwrite = open(dest_file_text,"+w")
		        fwrite.writelines(text.strip())

		        # add ann files
		        fwrite = open(dest_file_ann,"+w")
		        for ind in range(len(start)):
		        	line = "T" + str(ind+1) + '\t' + tags[ind] + ' ' + start[ind] + ' ' + end[ind] + '\t' + words[ind] + "\n"
		        	fwrite.writelines(line)

		        count += 1


    		length = 0
			sents = []
			text = ""
			# create columns needed to create brat annotation files
			start  = []
			end = []
			words = []
			tags = []
			for i,ent in enumerate(entities[math.floor(len(entities)/n):]):
                begin = i + math.floor(len(entities)/n)
                sent_len = 0
                text = text  + " ".join(entities[begin]) + " "
                for j,e in enumerate(entities[begin]):
                    if entities_type[begin][j] != 'O':
                        words.append(e)
                        tags.append(entities_type[begin][j][2:])
                        start.append(length + sent_len)
                        end.append(length + sent_len + len(e))
                    sent_len = sent_len + len(e) + 1
                length = length + len(ent)  + sum([len(e) for e in entities[begin]])

            file_name = folder + "_text_00" + str(count)
            dest_file_text = os.path.join(dest_file_path,file_name + ".txt")
            dest_file_ann = os.path.join(dest_file_path,file_name + ".ann")
            # create destination text file if it does not exist
            if not(pathlib.Path(dest_file_text).exists()):
                pathlib.Path(dest_file_text).touch()
            # create destination ann file if it does not exist
            if not(pathlib.Path(dest_file_ann).exists()):
                pathlib.Path(dest_file_ann).touch()

            # add text files
            fwrite = open(dest_file_text,"+w")
            fwrite.writelines(text.strip())

            # add ann files
            fwrite = open(dest_file_ann,"+w")
            for ind in range(len(start)):
                line = "T" + str(ind+1) + '\t' + tags[ind] + ' ' + str(start[ind]) + ' ' + str(end[ind]) + '\t' + words[ind] + "\n"
                fwrite.writelines(line)
                    
            count += 1

		      






