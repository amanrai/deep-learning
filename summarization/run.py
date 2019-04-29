import pickle
from summarizer import SummarizerCell
import numpy as np
import torch
from dataOps import *
network_testing_data = pickle.load(open("./network_testing.pickle", "rb"))
print(len(network_testing_data))

max_doc_length = 100
max_summary_length = 10
_cuda = torch.cuda.is_available()
    
d, se, m, su, po = genBatch(_data=network_testing_data, _cuda = _cuda, max_doc_length = 100, max_summary_length=12)
print(d.size(), se.size(), m.size(), su.size(), po.size())