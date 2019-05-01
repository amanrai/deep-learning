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
                 bert_width=768,
                 tf = True,
                 isCuda = True):
        super(Seq2SeqDecoderCell, self).__init__()
        
        self.bert_width = bert_width
        self.iscuda = isCuda
        
        self.gru = torch.nn.GRUCell(self.bert_width * 3, self.bert_width)
        self.attention_w = torch.nn.Linear(self.bert_width*2, attention_dim)
        self.attention_v = torch.nn.Linear(attention_dim, 1)

        self.intra_w = torch.nn.Linear(self.bert_width*2, attention_dim)
        self.intra_v = torch.nn.Linear(attention_dim, 1)

        self.embedding = output_embeddings
        if (output_embeddings == None):
            self.embedding = torch.nn.Embedding(30522, self.bert_width) #30522 is the size of the bert base uncased vocabulary

    def genHiddenState(self, size):
        if (self.iscuda):
            _hs = torch.zeros(size).cuda()
        else:
            _hs = torch.zeros(size)
        return _hs

    def forward(self, docs, last_hidden_state, last_Word, previous_words):
        #intra decoder attention
        _prev_emb = self.embedding(previous_words)
        _prev_words_att = Attention(_prev_emb, last_hidden_state.unsqueeze(1), self.intra_w, self.intra_v)
        intra_cv = ContextVector(_prev_emb, _prev_words_att)

        #document to decoder attention
        att = Attention(docs, last_hidden_state.unsqueeze(1), self.attention_w, self.attention_v)
        dcv = ContextVector(docs, att)

        #GRU
        _last_Word = self.embedding(last_Word)        
        _last_Word = _last_Word.squeeze(1)        
        _last_Word = torch.cat([dcv, intra_cv, _last_Word], dim=-1)
            
        hs = gru_forward(self.gru, _last_Word, last_hidden_state)

        #final word prediction
        word = torch.matmul(hs, self.embedding.weight.transpose(-2,-1))

        return word, att, hs