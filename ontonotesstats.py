#Just collecting some genre wise stats for the entire corpus
# (Before train/dev/test splits, in original LDC supplied form)

from bs4 import BeautifulSoup
import glob
file_path = "C:/Users/vajjalas/Desktop/NERProject2021/ontonotes/ontonotes-release-5.0/data/files/data/english/annotations/"
subfolders = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
for subfolder in subfolders:
    files = glob.glob(file_path + subfolder + '/**/*.name', recursive=True)
    mydict = {} #keys are (string, NEcat) tuples, and values are counts of their occurences.
    for afile in files:
        text = open(afile, encoding="utf-8").read()
        soup = BeautifulSoup(text, 'html.parser')
        enamexes = soup.find_all('enamex')
        #print(enamexes)
        for enamex in enamexes:
            mytuple = (enamex.get_text(), enamex["type"])
            mydict[mytuple] = mydict.get(mytuple, 0) +1

    sorteddict = sorted(mydict.items(), key=lambda x: x[1], reverse=True)
    output_path = open(subfolder+"-dist.txt", "w", encoding="utf-8")
    for key,val in sorteddict:
        output_path.write(str(key) + "\t"+str(val))
        output_path.write("\n")
    output_path.close()


