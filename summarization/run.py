import pickle
import numpy as np
import torch
import torch.nn.functional as F
from SummarizerCell import Seq2SeqCell as SummarizerCell
from dataOps import *
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel

network_testing_data = pickle.load(open("./network_testing.pickle", "rb"))
print(len(network_testing_data))

max_doc_length = 100
max_summary_length = 10
_cuda = torch.cuda.is_available()

s = SummarizerCell(isCuda=_cuda)
if (_cuda):
    s.cuda()
bs = 5

d, se, m, su, po = genBatch(bs = bs,
                            data=network_testing_data, 
                            _cuda = _cuda, 
                            max_doc_length = max_doc_length, 
                            max_summary_length=max_summary_length)

_hs = s.genHiddenState((d.size()[0], 768))
_prev_word = None
if (_cuda):
    _prev_word = torch.LongTensor([101]).cuda()
else:
    _prev_word = torch.LongTensor([101]).cuda()

_prev_word = _prev_word.repeat(bs, 1)
gen_words = []
gen_atts = []

_d, _ = self.bert(d, se, m, output_all_encoded_layers = False)
_d = _d * m.unsqueeze(-1).float()   

for i in range(5):
    new_words, atts, _hs = s.forward(d, se, m, _hs, _prev_word)
    actual_words = F.softmax(new_words, dim=-1)
    actual_words = torch.max(actual_words, dim=-1)[1]
    _prev_word = actual_words.unsqueeze(-1)
    gen_words.append(_prev_word.detach())
    gen_atts.append(atts.detach())

print(gen_words)
print(gen_atts)