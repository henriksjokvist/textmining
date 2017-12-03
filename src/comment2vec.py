from fastText import load_model
import numpy as np
import pandas as pd
import os.path
import csv
from sklearn.cluster import KMeans

class Data:
    def __init__(self):
        self.id = 0
        self.comment = ""
        self.vector = []
        self.group = -1

    def setID(self, idno):
        self.id = idno

    def setComment(self, commentstring):
        self.comment = commentstring

    def setWV(self, wordvector):
        self.vector = wordvector

    def setGroup(self, clusterID):
        self.group = clusterID


def comment2vec():
    """Takes one comment string and finds the vector representation of it"""

def KM(X, clu):#X = np array, clu = #clusters
    kmeans = KMeans(n_clusters=clu, random_state=0)
    kmeans.fit(X)
    return kmeans

def main():

    ROOT_PATH = os.path.abspath(__file__ + '/../..')

    pretrainedModel = ROOT_PATH + "\wiki.sv.bin"   # path of your pre-trained model, must be .bin file

    filepath = ROOT_PATH + '\data\FT_cleaned_data.csv' # pre-cleaned data in csv, first column is comment, second is ID

    fastTextModel=load_model(pretrainedModel)
    print("Pre-trained model loaded successfully!\n")

    df = pd.read_csv(filepath, sep=';', error_bad_lines=False, encoding="ISO-8859-1", dtype={'Comment': object})

    FT_model = []

    for row in df.as_matrix():
        newEntry = Data()
        newEntry.setComment(str(row[0]))
        newEntry.setID(row[1])
        newEntry.setWV(fastTextModel.get_sentence_vector(str(row[0])))  # get sentence vectors
        FT_model.append(newEntry)

    wordVectors = []
    for data_entry in FT_model:
        wordVectors.append(data_entry.vector)
    np_wordVectors = np.array(wordVectors)
    np.save(ROOT_PATH + "\data\wordVectors.npy", np_wordVectors)    # save word vectors of corpus for other uses

    """ k-Means clustering """
    k = 50      # number of clusters
    km = KM(np_wordVectors, k)
    labels = km.labels_     # cluster ID's

    reader = csv.reader(open(ROOT_PATH + '/data/FT_cleaned_data.csv', 'r'))
    writer = csv.writer(open(ROOT_PATH + '/data/FT_output.csv', 'w', newline=''))
    i = 0
    for row in reader:
        if i == 0:  # header
            i+=1
            continue
        row.append(labels[i-1])
        writer.writerow(row)
        i += 1

main()
