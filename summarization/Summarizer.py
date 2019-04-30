import torch
import torch.nn.functional as F
from SummarizerCell import Seq2SeqDecoderCell as SummarizerCell
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
import random

class BertSummarizer(torch.nn.Module):
    def __init__(self, 
                 bert_model = "bert-base-uncased",
                 tf = True,
                 isCuda = True):
        super(BertSummarizer, self).__init__()
        self.bert_width = 768
        self.bert_model = bert_model
        if ("-large-" in bert_model):
            self.bert_width = 1024
        self._cuda = isCuda
        self.teacherForcing = tf
        self.summarizer = SummarizerCell(isCuda=self._cuda)
        if (self._cuda):
            _bert = BertModel.from_pretrained(self.bert_model).cuda()
        else:
            _bert = BertModel.from_pretrained(self.bert_model)

    def genHiddenState(self,size):
        return self.summarizer.getHiddenState(size)

    def forwardBert(self, d, se, m):
        _d, _ = self.bert(d, se, m, output_all_encoded_layers = False)
        _d = _d * m.unsqueeze(-1).float()
        return _d

    def forwardSummary(self, _d, _hs, _prev_word):
        return self.summarizer.forward(_d, _hs, _prev_word)