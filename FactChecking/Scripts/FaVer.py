import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel
from QA_Attentions import *

class FaVer(torch.nn.Module):
    def __init__(self, bert_model = "bert-base-uncased", claim_cutoff_length = 20):
        super(FaVer, self).__init__()
        self.bert_model = bert_model
        self.bert_width = 768
        self.claim_cutoff_length = claim_cutoff_length
        if ("-large-" in self.bert_model):
            self.bert_width = 1024
        self.bert = BertModel.from_pretrained(bert_model)
        self.wd = torch.nn.Parameter(torch.FloatTensor(np.random.uniform(0, 1, (3*self.bert_width,))))
        self.innerAttDoc = torch.nn.Parameter(torch.FloatTensor(np.random.uniform(0, 1, (self.bert_width*4, 512))))
        self.out = torch.nn.Linear((self.bert_width*4),1)
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, t, s, a):
        """
            :param t: (b, t) #text -> [CLS] <claim> [SEP] <evid> [SEP] <[PAD]>
            :param s: (b, s) #text -> <claim> = 0 [SEP] <evid> = 1
            :param a: (b, a) #text -> t = 1; [PAD] = 0

            :output _f: (b,1) #sigmoid has not been applied. 
        """

        text, pooled = self.bert(t,
                        token_type_ids=s, 
                        attention_mask=a, 
                        output_all_encoded_layers=False)
        
        text = self.dropout(text)
        cl_ = s == 0
        ev_ = s == 1
        claims = text * cl_.unsqueeze(-1).float()
        claims = claims[:,:self.claim_cutoff_length, :]
        evidences = text * ev_.unsqueeze(-1).float()
        evidences = evidences * a.unsqueeze(-1).float()
        bdaf, ad2q, aq2d = biDAF(evidences, claims, self.wd)
        _f = self.out(InnerAttention(bdaf, self.innerAttDoc))
        return _f