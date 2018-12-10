import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from models import *
from translator import Translator


if __name__ == '__main__':

    # Set up arguement parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", "--network_architecture", help="String - Specify what network architecture to use - Options: (BasicRNN, EmbeddingRNN, EncDecRNN, EmbedBiRncDecRNN)")
    args = parser.parse_args()

    if args.network_architecture == 'BasicRNN':
        translator = Translator(BasicRNN, 'BasicRNN', embed=False)
    elif args.network_architecture == 'EmbeddingRNN':
        translator = Translator(EmbeddingRNN, 'EmbeddingRNN')
    elif args.network_architecture == 'EncDecRNN':
        translator = Translator(EncDecRNN, 'EncDecRNN', embed=False)
    elif args.network_architecture == 'EmbedBiEncDecRNN':
        translator = Translator(EmbedBiEncDecRNN, 'EmbedBiEncDecRNN')
    else:
        translator = Translator(EmbedBiEncDecRNN, 'EmbedBiEncDecRNN')  
    
    translator.load()
    
    while True:
        input_var = input("Enter English phrase: ")
        translator.translate(input_var.lower())