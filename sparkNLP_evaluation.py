# Import Spark NLP
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
# other packages
import pandas as pd
import time
import pathlib
import csv

def create_df(filepath):
	# read text tab-delimited file
	df = pd.read_csv(filepath,delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8',header=None, names=["Word","POS","DEREP","TYPE","SENT_NO"])
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



# function to get the ner for test files and save it csv file
def ner_onto(file_dir):
	'''
	input - 
	file_dir is path of directory containing txt files
	output -  
	csv created from the results spark dataframe that has the final ners
	'''
	# Start Spark Session with Spark NLP
	spark = sparknlp.start()

	# get list of files 
	flist = file_list(file_dir,"txt")

	# to create loop that will allow text to be extracted, ners extracted and result saved in csv
	for f in flist:

		# get sentences
		print("processing {}".format(f))
		sentences, _, _ = create_df(f)

		# combine text and get spark dataframe
		text = [" ".join(sentences)]
		df = spark.createDataFrame(pd.DataFrame({'text':text}))

		# defines the annotators
		document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document").setCleanupMode("shrink")
		sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
		tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token").setSplitChars(['-']).setContextChars(['(', ')', '?', '!', '.']) 

		# defines the embeddings and ner models
		embeddings = BertEmbeddings.pretrained("bert_base_cased", "en").setInputCols("sentence", "token").setOutputCol("embeddings")

		ner_onto = NerDLModel.pretrained("onto_bert_base_cased", "en").setInputCols(["document", "token", "embeddings"]).setOutputCol("ner")


		# define the pipeline
		nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner_onto])
		pipeline_model = nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text'))
		result = pipeline_model.transform(df)

		# create destination text file
		filename = "test_" + pathlib.Path(f).name.split(".")[0] + ".csv"

		# saving the results in csv
		print("starting to generate results {}".format(f))
		start = time.time()
		result.toPandas().to_csv(filename)
		print("Done for {} in {:3.3f}".format(f,time.time()-start))


if __name__ == "__main__":
	ner_onto("/Users/ramybal/Downloads/bio/spark/test")
