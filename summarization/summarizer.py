import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel

def Attention(to_, from_, w, v):
    if (from_.size()[1] > 1):
        assert "From is longer than one timestep!"
    

    _f = from_.repeat(1, to_.size()[1], 1)
    _f = torch.cat([to_, _f], dim=-1)
    _o = w(_f)
    return F.softmax(v(_o), dim=1)


class SummarizerCell(torch.nn.Module):
    def __init__(self, 
                 bert_model = "bert-base-uncased",
                 attention_dim = 512,
                 tf = True,
                 isCuda = True):
        super(SummarizerCell, self).__init__()
        self.bert_width = 768
        self.bert_model = bert_model
        self.iscuda = isCuda
        self.teacherForcing = tf
        
        self.gru = torch.nn.GRU(self.bert_width * 2, self.bert_width)
        self.attention_w = torch.nn.Linear(self.bert_width*2, attention_dim)
        self.attention_v = torch.nn.Linear(attention_dim, 1)

        if (self.iscuda):
            self.bert = BertModel.from_pretrained(bert_model).cuda()
        else:
            self.bert = BertModel.from_pretrained(bert_model)

    def genHiddenState(self, size):
        _hs = torch.ones(size)
        return _hs

    def forward(self, docs, segments, masks, last_hidden_state, input):
        _d, _ = self.bert(docs, segments, masks, output_all_encoded_layers = False)
        print(last_hidden_state.size(), _d.size())
        Attention(_d, last_hidden_state, self.attention_w, self.attention_v)
        return "Forward The Summarizer!"