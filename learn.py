import argparse
from models import *
from translator import Translator
from utils import load_data


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

    # Load English data
    eng_sentences = load_data('data/small_vocab_en')

    # Load French data
    fra_sentences = load_data('data/small_vocab_fr')
    
    translator.fit(eng_sentences, fra_sentences)
    
    translator.save()
 
