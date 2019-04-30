import numpy as np
import torch
import torch.nn.functional as F

def Attention(to_, from_, w, v):
    if (from_.size()[1] > 1):
        assert "From is longer than one timestep!"
    
    _f = from_.repeat(1, to_.size()[1], 1)
    _f = torch.cat([to_, _f], dim=-1)
    _o = torch.tanh(w(_f))
    return F.softmax(v(_o), dim=1)

def ContextVector(input, attention):
    _i = input*attention
    return torch.sum(_i, dim=1)

def gru_forward(gru_cell, input, hidden_state):
    out = gru_cell(input, hidden_state)
    return out

class Seq2SeqDecoderCell(torch.nn.Module):
    def __init__(self, 
                 bert_model = "bert-base-uncased",
                 output_embeddings=None,
                 attention_dim = 512,
                 tf = True,
                 isCuda = True):
        super(Seq2SeqDecoderCell, self).__init__()
        self.bert_width = 768
        self.bert_model = bert_model
        if ("-large-" in bert_model):
            self.bert_width = 1024
        self.iscuda = isCuda
        self.teacherForcing = tf
        
        self.gru = torch.nn.GRUCell(self.bert_width * 2, self.bert_width)
        self.attention_w = torch.nn.Linear(self.bert_width*2, attention_dim)
        self.attention_v = torch.nn.Linear(attention_dim, 1)

        self.embedding = output_embeddings
        if (output_embeddings == None):
            self.embedding = torch.nn.Embedding(30522, self.bert_width)

    def genHiddenState(self, size):
        if (self.iscuda):
            _hs = torch.zeros(size).cuda()
        else:
            _hs = torch.zeros(size)
        return _hs

    def forward(self, docs, last_hidden_state, input, previous_words):
        if (len(previous_words) > 0):
            for word in previou_words:
                print(word.size())
            _prev_collection = torch.stack(previous_words, dim=-1)
            print(_prev_collection.size())                 
        att = Attention(docs, last_hidden_state.unsqueeze(1), self.attention_w, self.attention_v)
        dcv = ContextVector(docs, att)
        _input = self.embedding(input)        
        _input = _input.squeeze(1)        
        _input = torch.cat([dcv, _input], dim=-1)        
        hs = gru_forward(self.gru, _input, last_hidden_state)
        word = torch.matmul(hs, self.embedding.weight.transpose(-2,-1))
        return word, att, hs