import pickle
from SummarizerCell import SummarizerCell
import numpy as np
import torch
from dataOps import *
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
new_words, atts = s.forward(d, se, m, _hs, _prev_word)
print("In run; new words:", new_words.size(), atts.size())
