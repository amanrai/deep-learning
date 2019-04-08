#paper reference:https://arxiv.org/pdf/1409.0473.pdf
#converts the model to use teacher forcing
#does not implement the max(2j, 2j+1) selection for output
#Aman Rai, Subash Chellary

import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec

class AttentionDecoder(Recurrent):

    def __init__ (self, 
                    units, 
                    output_dim, 
                    actual_source_timesteps=200,
                    actual_oneside_source_dim = 128,
                    output_to_vocab = False, 
                    output_vocab_size = 65004,
                    embedding_dim = 300,
                    alignment_dim = 1000,
                    output_timesteps = 20,
                    return_probabilities = True,
                    name="BahdanauAttention",
                    **kwargs):
                
        self.embedding_dim = embedding_dim
        self.hidden_units = units
        self.alignment_dim = alignment_dim
        
        self.return_probabilities = return_probabilities
        
        self.actual_source_timesteps = actual_source_timesteps
        self.actual_oneside_source_dim = actual_oneside_source_dim
        self.actual_bothsides_source_dim = actual_oneside_source_dim*2        
        
        self.output_to_vocab = output_to_vocab        
        self.output_dim = output_dim
        self.output_vocab_size = output_vocab_size
        self.output_timesteps = output_timesteps

        self.kernel_initializer = 'glorot_uniform'        
        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences


    def build(self, input_shape):
        assert type(input_shape) is list
        #inputs will be an array (he_x, and he_y)
        #(inputs coming in from the encoder representing x and y)
        self.x_input_shape = input_shape[0]
        self.y_input_shape = input_shape[1]

        self.states = [None, None]
        #initial state matrix
        
        self.Ws = self.add_weight(shape=(self.actual_oneside_source_dim, self.hidden_units), 
            name="ws", 
            initializer=self.kernel_initializer)
        self.bs = self.add_weight(shape=(self.hidden_units,), name="bs",
            initializer=self.kernel_initializer)
        
        #context vector matrices        
        #Uahj = hj DOT Ua
        #(batch, timesteps, actual_bothsides_source_dim) dot Ua (actual_bothsides_source_dim DOT alignment_dim)        
        self.Ua = self.add_weight(shape=(self.actual_bothsides_source_dim, self.alignment_dim), 
            name="ua", initializer=self.kernel_initializer)
        
        #s_p DOT Wa
        #(1, self.hidden_units) DOT (self.hidden_units, self.alignment_dim)
        self.Wa = self.add_weight(shape=(self.hidden_units, self.alignment_dim), 
            name="wa", initializer=self.kernel_initializer)
        self.Va = self.add_weight(shape=(self.alignment_dim,), 
            name="va", initializer=self.kernel_initializer)
        self.ba = self.add_weight(shape=(self.alignment_dim,), 
            name="ba", initializer=self.kernel_initializer)
        

        #Gate matrices
        #'reset' gate, r = sigmoid( DOT(W1, yi-1) + DOT(W2, si-1) + DOT(W3, Ci))
        #r.shape = (1, self.hidden_units)
        self.Wr = self.add_weight(shape=(self.embedding_dim, self.hidden_units), 
        name="wr", initializer=self.kernel_initializer)
        self.Ur = self.add_weight(shape=(self.hidden_units, self.hidden_units), 
        name="ur", initializer=self.kernel_initializer)
        self.Cr = self.add_weight(shape=(self.alignment_dim, self.hidden_units), 
        name="cr", initializer=self.kernel_initializer)
        self.br = self.add_weight(shape=(self.hidden_units,), 
        name="br", initializer=self.kernel_initializer)

        #'update gate' gate, z = sigmoid( DOT(W1, yi-1) + DOT(W2, si-1) + DOT(W3, Ci))
        #z.shape = (1, self.hidden_units)
        self.Wz = self.add_weight(shape=(self.embedding_dim, self.hidden_units), 
        name="wz", initializer=self.kernel_initializer)
        self.Uz = self.add_weight(shape=(self.hidden_units, self.hidden_units),
        name="uz", initializer=self.kernel_initializer)
        self.Cz = self.add_weight(shape=(self.alignment_dim, self.hidden_units), 
        name="cz", initializer=self.kernel_initializer)
        self.bz = self.add_weight(shape=(self.hidden_units,), 
        name="bz", initializer=self.kernel_initializer)

        """
        """
        #S-Candidate matrices
        self.W = self.add_weight(shape=(self.embedding_dim, self.hidden_units), 
        name="w", initializer=self.kernel_initializer)
        self.U = self.add_weight(shape=(self.hidden_units, self.hidden_units),
        name="u", initializer=self.kernel_initializer)
        self.C = self.add_weight(shape=(self.alignment_dim, self.hidden_units), 
        name="c", initializer=self.kernel_initializer)
        self.bc = self.add_weight(shape=(self.hidden_units,), 
        name="bc", initializer=self.kernel_initializer)
        
        """
        """
        #outputs
        #NOTE: not implementing the max(2j, 2j+1) selection for ti
        self.Uo = self.add_weight(shape=(self.hidden_units, self.output_dim), 
            name="uo", initializer=self.kernel_initializer)
        self.Wo = self.add_weight(shape=(self.embedding_dim, self.output_dim), 
        name="wo", initializer=self.kernel_initializer)
        self.Co = self.add_weight(shape=(self.alignment_dim, self.output_dim), 
        name="co", initializer=self.kernel_initializer)
        self.bo = self.add_weight(shape=(self.output_dim,), 
        name="bo", initializer=self.kernel_initializer)
        
        """
        """
        #Output to vocab directly       
        if (self.output_to_vocab == True):
            #output directly to vocab space
            self.WVo = self.add_weight(shape=(self.output_dim, self.output_vocab_size),
            name="wvo", initializer=self.kernel_initializer)
            self.bvo = self.add_weight(shape=(self.output_vocab_size,),
            name="bvo", initializer=self.kernel_initializer)
        
        #self.input_spec = [
        #    InputSpec(()), InputSpec(shape=self.y_input_shape)
        #    ]
        self.built = True

    def call(self, x):
        he_x = x[0]
        he_y = x[1]
        self.x_last = he_x[:,-1,self.actual_oneside_source_dim:]
        self.uahj = K.dot(he_x, self.Ua) + self.ba
        return super(AttentionDecoder, self).call(he_y)
    
    
    def step(self, x,  states):
        s_p = states[1]
        y_p = x
        
        #calculate context vector
        wasi_p = K.dot(s_p, self.Wa) #(batch_size, alignment_dim)
        ei_ = K.dot(K.tanh(self.uahj + K.expand_dims(wasi_p, axis=1)), K.expand_dims(self.Va, -1)) #(batch_size, t, 1)
        alpha_ = K.squeeze(K.exp(ei_), axis=-1)
        alpha_sum_ = K.sum(alpha_, axis=1, keepdims=True)
        alpha = alpha_/alpha_sum_ #batch, t, 1
        cti = K.batch_dot(self.uahj, alpha, axes=[1,1]) #batch, alignment_dim

        #calculate reset gate
        rt = K.sigmoid( K.dot(y_p, self.Wr) + K.dot(s_p, self.Ur) + K.dot(cti, self.Cr) + self.br)
        #calculate update gate
        zt = K.sigmoid( K.dot(y_p, self.Wz) + K.dot(s_p, self.Uz) + K.dot(cti, self.Cz) + self.bz)
        
        #calculate new s
        s_cand = K.tanh(K.dot(rt*s_p, self.U) + K.dot(y_p, self.W) + K.dot(cti, self.C) + self.bc)        
        st = (1-zt)*s_p + (zt)*s_cand
        
        #calculate the output
        ti = K.dot(s_p, self.Uo) + K.dot(y_p, self.Wo) + K.dot(cti, self.Co) + self.bo

        #Output to vocab directly       
        if (self.output_to_vocab == True):
            yi = K.softmax(K.dot(ti, self.WVo) + self.bvo)
            return yi, [yi, st]
        else:
            return ti, [ti, st]


    def get_initial_state(self, x):     
        s0 = K.tanh(K.dot(self.x_last, self.Ws) + self.bs)        
        o = K.zeros_like(x)  # (samples, timesteps, input_dims)
        o = K.sum(o, axis=(1, 2))  # (samples, )        
        o = K.expand_dims(o)  # (samples, 1)
        if (self.output_to_vocab):
            o = K.tile(o, [1, self.output_vocab_size])
        else:
            o = K.tile(o, [1, self.output_dim])
        return [o, s0]

    def compute_output_shape(self, input_shape):
        assert type(input_shape) is list
        if self.output_to_vocab:
            return (input_shape[0][0], self.output_timesteps, self.output_vocab_size)
        else:
            return (input_shape[0][0], self.output_timesteps, self.output_dim)
    
    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'units': self.hidden_units,
            'output_vocab_size': self.output_vocab_size,
            'return_probabilities': self.return_probabilities,
            'name':self.name,
            'actual_source_timesteps':self.actual_source_timesteps,
            'actual_oneside_source_dim':self.actual_oneside_source_dim,
            'output_to_vocab':self.output_to_vocab, 
            'output_vocab_size':self.output_vocab_size,
            'embedding_dim':self.embedding_dim,
            'alignment_dim':self.alignment_dim,
            'output_timesteps':self.output_timesteps,
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        #this is thre git commit comment