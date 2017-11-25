import csv
import numpy as np
from sklearn.cluster import KMeans
from cucco import Cucco

def rsw(infile,outfile):
    l = []
    cucco = Cucco()
    with open(infile,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for line in row:
                l.append(cucco.remove_stop_words(line,language = 'sv'))
    with open(outfile, 'wt') as w:
        wr = csv.writer(w)
        wr.writerow(l)

def KM(X, clu):#X = np array, clu = #clusters
    kmeans = KMeans(n_clusters=clu, random_state=0).fit(X)
    return kmeans
