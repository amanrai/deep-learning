"""
Testing code.
"""
import torch
import torch.nn.functional as F
import numpy as np

from QA_Attentions import *

dims = 11
docs = torch.Tensor(np.random.uniform(0, 1, (2, 10, dims)))
queries = torch.Tensor(np.random.uniform(0, 1, (2, 5, dims)))
q_i = torch.nn.Linear(dims, dims)
wd = torch.Tensor(np.random.uniform(0,1,(3*dims,)))



#Dynamic CoAttention
cd, cq, att_d, att_q = DynamicCoAttention(docs, queries, q_i)
print(cd.size(), cq.size(), att_d.size(), att_q.size())

v, ad2q, aq2d = biDAF(docs, queries, wd)

print(v.size(), ad2q.size(), aq2d.size())

inner_att_w = torch.Tensor(np.random.uniform(0, 1, (11, 22)))
q = InnerAttention(docs, inner_att_w)

print(q.size())