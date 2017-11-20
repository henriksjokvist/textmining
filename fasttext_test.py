# https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27

from __future__ import print_function

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import KeyedVectors

# Creating the model
sv_model = KeyedVectors.load('sv.bin')

# Getting the tokens
words = []
for word in sv_model.vocab:
    words.append(word)

# Printing out number of tokens available
print("Number of Tokens: {}".format(len(words)))

print("Dimension of a word vector: {}".format(
    len(sv_model[words[0]])
))
#print(words[:10])
#print(sv_model['bank'])

# Pick a word
find_similar_to = 'handelsbanken'

# Finding out similar words [default= top 10]
print("\nWords most similar to " + find_similar_to + ":")
for similar_word in sv_model.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(
        similar_word[0], similar_word[1]
    ))


print("\n")

for similar_word in sv_model.wv.most_similar_cosmul(positive=['yrke', 'kvinna'], negative=['män']):
    print("Word: {0}, Similarity: {1:.2f}".format(
        similar_word[0], similar_word[1]
    ))

