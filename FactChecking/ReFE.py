import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel
from QA_Attentions import *

class ReFE(torch.nn.Module):
    def __init__(self, bert_model = "bert-base-uncased", do_highway = False, train_fp16 = False):
        super(ReFE, self).__init__()
        self.train_fp16 = False
        self.do_highway = do_highway
        self.bert_model = bert_model
        self.bert_width = 768
        if ("-large-" in self.bert_model):
            self.bert_width = 1024
        self.bert = BertModel.from_pretrained(bert_model)
        self.wd = torch.nn.Parameter(torch.FloatTensor(np.random.uniform(0, 1, (3*self.bert_width,))))
        if (self.do_highway):
            self.innerAttQuery = torch.nn.Parameter(torch.FloatTensor(np.random.uniform(0, 1, (self.bert_width, 512))))
            self.out = torch.nn.Linear((self.bert_width*5),1)
        else:
            self.out = torch.nn.Linear((self.bert_width*4),1)
        self.innerAttDoc = torch.nn.Parameter(torch.FloatTensor(np.random.uniform(0, 1, (self.bert_width*4, 512))))
        
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, dt, ds, da, qt, qs, qa):
        """
            :param dt: (b, t) #text -> [CLS] <evid> <[PAD]>
            :param ds: (b, s) #text -> <evid> = 0
            :param da: (b, a) #text -> <evid> = 1; [PAD] = 0

            :param qt: (b, t) #text -> [CLS] <claim> <[PAD]>
            :param qs: (b, s) #text -> <claim> = 0 
            :param qa: (b, a) #text -> <claim> = 1; [PAD] = 0

            :output _f: (b,1) #sigmoid has not been applied. 
        """

        queries, pooled = self.bert(qt, 
                         token_type_ids=qs, 
                         attention_mask=qa, 
                         output_all_encoded_layers=False)
        
        documents, pooled = self.bert(dt, 
                         token_type_ids=ds, 
                         attention_mask=da, 
                         output_all_encoded_layers=False)
        
        if (self.train_fp16):
            queries = self.dropout(queries * qa.unsqueeze(-1).half().cuda())
            documents = self.dropout(documents * da.unsqueeze(-1).half().cuda())
        else:
            queries = self.dropout(queries * qa.unsqueeze(-1).float())
            documents = self.dropout(documents * da.unsqueeze(-1).float())
        
        bdaf, ad2q, aq2d = biDAF(documents, queries, self.wd)
        d = InnerAttention(bdaf, self.innerAttDoc)
        
        out_ = None
        if (self.do_highway):
            q = InnerAttention(queries, self.innerAttQuery)
            _f = torch.cat([q,d],dim=-1)
            out_ = self.out(_f)
        else:
            out_ = self.out(d)    
        
        return out_