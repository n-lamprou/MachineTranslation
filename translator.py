from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from models import BasicRNN, BidirectionalRNN, EncDecRNN, EmbeddingRNN
import numpy as np
import pickle as pkl


class Translator:
    """
    Translator
    """
        
    def __init__(self, Model, name, embed=True):
        """
        :param Model: model object to fit and use
        """
        self.Model = Model
        self.name = name
        self.embed = embed
        self.model = None
        self.x_tokenizer = None
        self.y_tokenizer = None
        self.max_y_sequence_length = None
        
    def fit(self, x_sentences, y_sentences, epochs=1):
        
        # Preprocess sentences
        preproc_x_sentences, preproc_y_sentences = self._preprocess(x_sentences, y_sentences)
    
        self.max_y_sequence_length = preproc_y_sentences.shape[1]
        x_vocab_size = len(self.x_tokenizer.word_index)
        y_vocab_size = len(self.y_tokenizer.word_index)
        
        # Reshape the input 
        tmp_x = self._pad(preproc_x_sentences, self.max_y_sequence_length)
        if not self.embed:
            tmp_x = tmp_x.reshape((-1, preproc_y_sentences.shape[-2], 1))
        
        print(tmp_x.shape)
        self.model = self.Model(tmp_x.shape, self.max_y_sequence_length, x_vocab_size + 1, y_vocab_size + 1)
        self.model.fit(tmp_x, preproc_y_sentences, self.name, epochs=20, batch_size=512)
   
    def translate(self, sentence):
        
        sentence = [self.x_tokenizer.word_index[word] for word in sentence.split()]
        sentence = self._pad([sentence], self.max_y_sequence_length)
        if not self.embed:
            sentence = sentence.reshape(1, -1, 1)
        predictions = self.model.predict(sentence)
        
        index_to_words = {id: word for word, id in self.y_tokenizer.word_index.items()}
        index_to_words[0] = '<PAD>'
        
        translation = ' '.join([index_to_words[np.argmax(x)] for x in predictions[0]])
        print(translation)
        
    def save(self):
        save_dict = {'embed': self.embed,
                     'x_tokenizer': self.x_tokenizer,
                     'y_tokenizer': self.y_tokenizer,
                     'max_y_sequence_length': self.max_y_sequence_length}
        
        with open('{}.pkl'.format(self.name), 'wb') as handle:
            pkl.dump(save_dict, handle)
             
    def load(self):
        with open('{}.pkl'.format(self.name), 'rb') as handle:
            load_dict = pkl.load(handle)
        
        self.embed = load_dict['embed']
        self.x_tokenizer = load_dict['x_tokenizer']
        self.y_tokenizer = load_dict['y_tokenizer']
        self.max_y_sequence_length = load_dict['max_y_sequence_length']
        self.model = load_model('{}.h5'.format(self.name))
            
    def _preprocess(self, x, y):
        """
        Preprocess x and y
        :param x: Feature List of sentences
        :param y: Label List of sentences
        :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
        """
        preprocess_x, self.x_tokenizer = self._tokenize(x)
        preprocess_y, self.y_tokenizer = self._tokenize(y)

        preprocess_x = self._pad(preprocess_x)
        preprocess_y = self._pad(preprocess_y)

        # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
        preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

        return preprocess_x, preprocess_y
        
    def _tokenize(self, x):
        """
        Tokenize x
        :param x: List of sentences/strings to be tokenized
        :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
        """
        keras_tokenizer = Tokenizer()
        keras_tokenizer.fit_on_texts(x)
        return keras_tokenizer.texts_to_sequences(x), keras_tokenizer

    def _pad(self, x, length=None):
        """
        Pad x
        :param x: List of sequences.
        :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
        :return: Padded numpy array of sequences
        """
        padded_x = pad_sequences(x, maxlen=length, padding='post')
        return padded_x

