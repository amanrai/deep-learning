import pickle
import numpy as np
import torch
import torch.nn.functional as F
from SummarizerCell import Seq2SeqDecoderCell as SummarizerCell
from dataOps import *
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel

network_testing_data = pickle.load(open("./network_testing.pickle", "rb"))

wordCriterion = torch.nn.CrossEntropyLoss()


max_doc_length = 100
max_summary_length = 10
_cuda = torch.cuda.is_available()

s = SummarizerCell(isCuda=_cuda)
optimizer = torch.optim.Adam(s.parameters(), lr=1e-3)
if (_cuda):
    s.cuda()

bert = None
bert_model = "bert-base-uncased"
if (_cuda):
    bert = BertModel.from_pretrained(bert_model).cuda()
else:
    bert = BertModel.from_pretrained(bert_model)
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
coverages = []

gen_logits = []
act_words = []
_d, _ = bert(d, se, m, output_all_encoded_layers = False)
_d = _d * m.unsqueeze(-1).float()   

optimizer.zero_grad()
for i in range(5):
    act_words.append(su[:,i])
    new_words, atts, _hs = s.forward(_d, _hs, _prev_word)
    actual_words = F.softmax(new_words, dim=-1)
    actual_words = torch.max(actual_words, dim=-1)[1]
    _prev_word = actual_words.unsqueeze(-1)
    gen_words.append(_prev_word.detach())
    gen_atts.append(atts.detach())
    gen_logits.append(new_words)
gen_logits = torch.stack(gen_logits, dim=0).view(-1, 30000)
act_words = torch.stack(act_words, dim=0).view(-1, 1)
loss = wordCriterion(gen_logits, act_words)
loss.backward()
optimizer.step()