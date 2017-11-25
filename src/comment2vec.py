from fastText import load_model
import numpy as np

def comment2vec():
    """Takes one comment string and finds the vector representation of it"""

def corpus2vecspace():
    """Takes a pre-trained fastText model and an array of comment strings and maps each comment to a vector"""



def main():
    pretrainedModel="wiki.sv.bin"   # path of your pre-trained model, must be .bin file

    model=load_model(pretrainedModel)
    print("Pre-trained model loaded successfully!\n")
