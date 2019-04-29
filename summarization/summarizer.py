import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel

class Summarizer(torch.nn.Module):
    def __init__(self, 
                 bert_model = "bert-base-uncased",
                 attention_dim = 512,
                 tf = True,
                 isCuda = True):
        super(Summarizer, self).__init__()
        self.bert_width = 768
        self.bert_model = bert_model
        self.iscuda = isCuda
        self.teacherForcing = tf

    def forward(self):
        return "Forward The Summarizer!"