#Aman Rai, April 2019. 

import torch
import torch.nn.functional as F
import numpy as np

def DynamicCoAttention(documents, queries, queries_intermediate_layer):
    """
        https://arxiv.org/pdf/1611.01604.pdf

        :param documents: (b, td, dim)
        :param queries:  (b, tq, dim)
        :param queries_intermediate_layer: torch.nn.Linear(dim, dim)

        :output CD: (b, td, 2*dim) #Summary of the document with relation to the query
        :output CQ: (b, tq, dim) #Summary of the query with relation to the document
        :output att_d: (b, tq, dim) #Softmax'd attention of the query with relation to the document
        :output att_q: (b, td, dim) #Softmax'd attention of the document with relation to the query
    """
    queries_int = torch.tanh(queries_intermediate_layer(queries))
    affinity = torch.matmul(queries_int, documents.transpose(-2, -1))
    att_d = F.softmax(affinity, dim=-1)
    att_q = F.softmax(affinity.transpose(-2, -1), dim= -1)
    cq = torch.matmul(documents.transpose(-2,-1), att_q).transpose(-2,-1)

    cd = torch.matmul(
        torch.cat([queries, cq], dim=-1).transpose(-2, -1), 
        att_d).transpose(-2,-1)

    return cd, cq, att_d, att_q

def biDAF(docs, queries, wd):
    """
        http://people.cs.vt.edu/mingzhu/papers/conf/www2019.pdf
        https://arxiv.org/pdf/1611.01603.pdf

        :param docs: (b, dt, dim)
        :param queries: (b, qt, dim)
        :param wd: (dim*3,)
        
        :output v: (b, dt, 4*dim)
        :output att_d2q: (b, dt, qt)
        :output att_q2d: (b, qt, dt)
    """

    dq = []
    for i in range(queries.size()[1]):
        qi = queries[:,i,:].unsqueeze(1) #(b, 1, dim)
        qi = qi.expand(-1,docs.size()[1],-1) #(b, dt, dim)
        dqi = torch.cat([docs, qi, docs*qi], dim=-1) #(b, dt, 3*dim)
        dq.append(dqi)
        
    dq = torch.stack(dq) #(qt, b, dt, 3*dim)
    dq = dq.transpose(0, 1) #(b, qt, dt, 3*dim)
    dq = dq.transpose(1,2) #(b, dt, qt, 3*dim)

    dq = torch.matmul(dq, wd) #(b, dt, qt) 
    
    att_d2q = F.softmax(dq, dim=-1) #along the rows
    att_q2d = F.softmax(dq, dim=1) #along the columns
    
    ad2q = torch.matmul(att_d2q, queries) #(b, dt, dim)
    aq2d = torch.matmul(att_d2q, att_q2d.transpose(-2,-1)) #(b, dt, dt)
    aq2d = torch.matmul(aq2d,docs) #(b, dt, dim)
    v = torch.cat([docs, ad2q, ad2q*docs, aq2d*docs], dim=-1) #(b, dt, 4*dim)
    
    return v, att_d2q, att_q2d.transpose(-2,-1)


def InnerAttention(matrix, att_weights):
    """
        http://people.cs.vt.edu/mingzhu/papers/conf/www2019.pdf

        :param matrix: (b, t, dim)
        :param att_weights: (dim, att_dim)
        
        :output qv: (b, 1, dim)
    """

    _m = torch.tanh(torch.matmul(matrix, att_weights))
    _m = torch.matmul(_m, att_weights.transpose(1,0))
    _m = F.softmax(_m, dim=1)
    return torch.sum(_m * matrix, dim=1)


def dotProductAttention(mat_a, mat_b, weights):
    """
        :param mat_a: (b, at, dim)
        :param mat_b: (b, 1, dim)
        :param weights: (dim, dim)
        
        :output att: (b, at)
    """
    _a = torch.tanh(torch.matmul(mat_a, weights)) # b, at, dim; [-1,1]
    _b = torch.matmul(_a, mat_b.transpose(-2, -1)) #b, at, bt
    att = F.softmax(_b, dim=1) #b, at
    return att
    
    
    