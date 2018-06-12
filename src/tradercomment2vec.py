"""TraderComment2vec by Henrik Sjökvist, June 2018"""

from fastText import load_model
import numpy as np
import pandas as pd
import os.path
import csv
from sklearn.cluster import KMeans, DBSCAN
from sklearn.utils.extmath import randomized_svd


def cluster_KM(df, k):
    """Performs k-Means clustering"""
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df)
    labels = [str(i) for i in kmeans.labels_]
    return labels


def cluster_DBSCAN(df, epsilon, minPts):
    """Performs DBSCAN clustering"""
    dbscan = DBSCAN(eps=epsilon, min_samples=minPts)
    dbscan.fit(df)
    labels = [str(i) for i in dbscan.labels_]
    return labels


def average(corpus, ft_model):
    """Performs averaging to generate sentence embeddings"""
    sentence_embeddings = []
    for row in corpus:
        sentence_embeddings.append(ft_model.get_sentence_vector(str(row[0])))

    np_sentence_embeddings = np.array(sentence_embeddings)
    return np_sentence_embeddings


def SIF(corpus, ft_model, a):
    """Performs Smooth Inverse Frequency weighting to generate sentence embeddings"""
    freqlist = pd.read_csv(os.path.abspath(__file__ + '/../..') + '\data\parole_most_freq_10k.txt', sep="\t", header=None, encoding="ISO-8859-1")
    freqlist.columns = ['freq', 'word']
    freqlist = freqlist[['word', 'freq']]
    freqlist.freq = freqlist.freq / freqlist.freq.sum()
    b_list = []
    for row in corpus:
        v_list = []
        for word in str(row[0]).split():
            if word in freqlist.word.unique():
                fq = float(freqlist.loc[freqlist['word']==word].freq)
            else:
                fq = 0
            wv = ft_model.get_word_vector(word)
            SIF_weights = a/(a+fq)
            wv = SIF_weights * wv
            v_list.append(wv)
        v_list = np.array(v_list)
        b_list.append(np.mean(v_list, axis=0))

    b_list = np.array(b_list)
    U, S, PCA1 = randomized_svd(b_list, n_components=1)
    c_list = []
    for b in b_list:
        c = b - np.dot(np.transpose(PCA1)*PCA1,b)
        c_list.append(c)
    c_list = np.array(c_list)
    return c_list


def convert_to_xml(data):
    xml = ""
    for ii, row in data.iterrows():
        xml += convert_row(row)

    return xml


def convert_row(row):
    return """\n<row>
    <ID>%s</ID>
    <cluster>%s</cluster>
    </row>""" % (
    row.ID, row.Cluster)


def tradercomment2vec():
    """Model settings"""
    SEGT = "Avg"                        # Sentence Embedding Generation Technique, "Avg" or "SIF"
    a = 1e-3                            # for SIF
    clustering_algorithm = "KM"     # "KM" or "DBSCAN"
    k = 130                             # for k-means
    epsilon = 0.1                           # for DBSCAN
    minPts = 9                          # for DBSCAN
    output_embeddings = True            # Output set of sentence embeddings as numpy array
    output_xml = True                   # Output cluster ID's as XML file
    output_csv = True                   # Output cluster ID's as CSV file

    ROOT_PATH = os.path.abspath(__file__ + '/../..')

    # Load fastText model
    pretrainedModel = ROOT_PATH + "\word_embeddings\wiki.sv.bin"  # path of your pre-trained model, must be .bin file
    fastTextModel=load_model(pretrainedModel)
    print("\nPre-trained model loaded successfully!\n")

    # Load dataset of trader comments
    filepath = ROOT_PATH + '\data\FT_cleaned_data.csv'  # pre-cleaned data in csv, first column is comment, second is ID
    df = pd.read_csv(filepath, sep=';', error_bad_lines=False, encoding="ISO-8859-1", dtype={'Comment': object})
    print("Trader comments loaded successfully!\n")

    # Find sentence embedding for each trader comment
    if SEGT == "Avg":
        sentence_embeddings = average(df.as_matrix(), fastTextModel)
    elif SEGT == "SIF":
        sentence_embeddings = SIF(df.as_matrix(), fastTextModel, a)
    else:
        print("Error: Invalid Sentence Embedding Generation Technique")
        return
    print("Sentence embeddings generated successfully!\n")

    outputPath = ROOT_PATH + "\output"
    if output_embeddings:
        np.save(outputPath + "\sentence_embeddings\wordVectors.npy", sentence_embeddings)  # save word vectors of corpus for other uses

    # Cluster sentence embeddings
    if clustering_algorithm == "KM":
        cluster_IDs = cluster_KM(sentence_embeddings, k)
    elif clustering_algorithm == "DBSCAN":
        cluster_IDs = cluster_DBSCAN(sentence_embeddings, epsilon, minPts)
    else:
        print("Error: Invalid Clustering Algorithm")

    # Output numerical features
    output = df
    output["Cluster"] = cluster_IDs
    output = output.drop(["Comment"], axis=1)
    if output_csv:
        if clustering_algorithm == "KM":
            output.to_csv(outputPath + '/csv/FT_' + clustering_algorithm + '_' + str(k) + '.csv', index=False)
        else:
            output.to_csv(outputPath + '/csv/FT_' + clustering_algorithm + '_' + str(epsilon).strip('.') +
                          '_' + str(minPts) + '.csv', index=False)

    if output_xml:
        xml_file = convert_to_xml(output)
        if clustering_algorithm == "KM":
            text_file = open(outputPath + '/xml/FT_' + clustering_algorithm + '_' + str(k) + '.xml', 'w')
        else:
            text_file = open(outputPath + '/xml/FT_' + clustering_algorithm + '_' + str(epsilon).strip('.') +
                             '_' + str(minPts) + '.xml', 'w')

        text_file.write("""<?xml version="1.0" encoding="UTF-8"?>
        <DataTable>""")
        text_file.write(xml_file)
        text_file.write("\n</DataTable>")
        text_file.close()


tradercomment2vec()


# create new project with only necessary files
# push everything to github
# create a folder where outputs are dumped