import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.losses import sparse_categorical_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam


class BaseModel:
    """
    DNN Model 
    """
        
    def __init__(self, input_shape, output_sequence_length, source_vocab_size, target_vocab_size):
        """
        :param input_dim: dimensions of input layer
        :param output_dim: dimensions of output layer
        :param hidden_nodes: number of hidden layers
        :param learn_rate: learning rate
        """
        self.input_shape = input_shape
        self.output_sequence_length = output_sequence_length
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.learn_rate = 0.005
        self._build()

    def _build(self):
        raise NotImplementedError("{} must override _build()".format(self.__class__.__name__))
        
    def fit(self, input_sentences, target_sentences, name, epochs=20, batch_size=512, val_split=0.2):
        # Define Callbacks
        f_path = '{}.h5'.format(name)
        checkpointer = ModelCheckpoint(filepath=f_path, monitor='val_loss', verbose=0, save_best_only=True)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=2, verbose=0, mode='min')
        self.model.fit(input_sentences, target_sentences, batch_size=batch_size, epochs=epochs, validation_split=val_split,
                       callbacks=[checkpointer, early_stopper])
        
    def predict(self, predict_data):
        return self.model.predict(predict_data)
    
    def save(self, file):
        self.model.save(file)

        
class BasicRNN(BaseModel):
        
    def __init__(self, input_shape, output_sequence_length, source_vocab_size, target_vocab_size):
        """
        """
        super().__init__(input_shape, output_sequence_length, source_vocab_size, target_vocab_size)
        
    def _build(self):
        self.model = Sequential()
        self.model.add(GRU(256, input_shape=self.input_shape[1:], return_sequences=True))
        self.model.add(GRU(128, return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.target_vocab_size)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=sparse_categorical_crossentropy,
                           optimizer=Adam(self.learn_rate),
                           metrics=['accuracy'])
        

class BidirectionalRNN(BaseModel):
        
    def __init__(self, input_shape, output_sequence_length, source_vocab_size, target_vocab_size):
        """
        """
        super().__init__(input_shape, output_sequence_length, source_vocab_size, target_vocab_size)
        
    def _build(self):
        self.model = Sequential()
        self.model.add(GRU(256, input_shape=self.input_shape[1:], return_sequences=True))
        self.model.add(GRU(128, return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.target_vocab_size)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=sparse_categorical_crossentropy,
                           optimizer=Adam(self.learn_rate),
                           metrics=['accuracy'])
        

class EncDecRNN(BaseModel):
        
    def __init__(self, input_shape, output_sequence_length, source_vocab_size, target_vocab_size):
        """
        """
        super().__init__(input_shape, output_sequence_length, source_vocab_size, target_vocab_size)
        
    def _build(self):
        self.model = Sequential()
        self.model.add(GRU(256, return_sequences=True, input_shape=self.input_shape[1:]))
        self.model.add(GRU(128, return_sequences=False))
        self.model.add(RepeatVector(self.output_sequence_length))
        self.model.add(GRU(256, return_sequences=True))
        self.model.add(GRU(128, return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.target_vocab_size)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=sparse_categorical_crossentropy,
                           optimizer=Adam(self.learn_rate),
                           metrics=['accuracy'])
        
        
class EmbeddingRNN(BaseModel):
        
    def __init__(self, input_shape, output_sequence_length, source_vocab_size, target_vocab_size):
        self.embed_size = 32
        super().__init__(input_shape, output_sequence_length, source_vocab_size, target_vocab_size)
        
    def _build(self):
        self.model = Sequential()
        self.model.add(Embedding(self.source_vocab_size, self.embed_size, input_length=self.input_shape[1]))
        self.model.add(GRU(256, return_sequences=True))
        self.model.add(GRU(128, return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.target_vocab_size)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=sparse_categorical_crossentropy,
                           optimizer=Adam(self.learn_rate),
                           metrics=['accuracy'])
        
        
class EmbedBiEncDecRNN(BaseModel):
        
    def __init__(self, input_shape, output_sequence_length, source_vocab_size, target_vocab_size):
        self.embed_size = 32
        self.learn_rate = 0.003
        super().__init__(input_shape, output_sequence_length, source_vocab_size, target_vocab_size)
        
    def _build(self):
        self.model = Sequential()
        self.model.add(Embedding(self.source_vocab_size, self.embed_size, input_length=self.input_shape[1]))
        self.model.add(Bidirectional(GRU(256, return_sequences=True)))
        self.model.add(Bidirectional(GRU(128, return_sequences=False)))
        self.model.add(RepeatVector(self.output_sequence_length))
        self.model.add(Bidirectional(GRU(256, return_sequences=True)))
        self.model.add(Bidirectional(GRU(128, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(self.target_vocab_size)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=sparse_categorical_crossentropy,
                           optimizer=Adam(self.learn_rate),
                           metrics=['accuracy'])
