#Aman Rai, March 2020
#Basic Transformer architecture from Vaswani et al, Attention is all you need. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TransformerFF(torch.nn.Module):
    def __init__(self, inner_dims=3072, model_dims=768, dropout=0.1, inner_act_fn=None, eps=1e-12):
        super(TransformerFF, self).__init__()
        self.inner_act_fn = F.gelu #modified from relu by almost everybody
        if (inner_act_fn is not None):
            self.inner_act_fn = inner_act_fn
        self.fc1 = nn.Linear(model_dims, inner_dims)
        self.fc2 = nn.Linear(inner_dims, model_dims)
        self.LayerNorm = nn.LayerNorm(model_dims, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        int_ = self.dropout(self.fc1(inputs))
        int_ = self.inner_act_fn(int_)
        int_ = self.dropout(self.fc2(int_))
        return self.LayerNorm(int_ + inputs)

class TransformerAttention(torch.nn.Module):
    def __init__(self, heads, dims, model_dims = 768, dropout=0.1, causal_mask=True):
        super(TransformerAttention, self).__init__()
        self.heads = heads
        self.dims = dims
        self.temp = math.sqrt(dims)
        self.query = nn.Linear(model_dims, heads*dims)
        self.key = nn.Linear(model_dims, heads*dims)
        self.value = nn.Linear(model_dims, heads*dims)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.causal_mask = causal_mask

    def forward(self, prev_hidden_states, attn_mask = None, encoder_outputs = None, causal_mask = True, return_attn = False):
        """ 
            prev_hidden_states: Comes from the previous transformer block/embedding outputs, has the shape b, t, model_dims
            encoder_outputs: Comes from a bidirectional encoder of some kind, has the shape b, t, model_dims. To use as encoder only, send None
            causal_mask is left to right masking. it will prevent the decoder from attending to future states. 
        """
        if (self.causal_mask == True and encoder_outputs == False):
            causal_attn_mask = torch.tril(torch.zeros((prev_hidden_states.size()[1], prev_hidden_states.size()[1]))).unsqueeze(0) # 1, seq, seq
            causal_mask = causal_attn_mask.repeat(prev_hidden_states.size()[0], 1, 1) #bs, seq, seq
            padding_attn_mask = attn_mask.unsqueeze(1) #bs, 1, seq
            padding_attn_mask = padding_attn_mask.repeat(1, prev_hidden_states.size()[1], 1) #bs, seq, seq
            attn_mask = causal_attn_mask * padding_attn_mask #bs, seq, seq
            attn_mask = attn_mask.unsqueeze(1) #bs, 1, seq, seq
            attn_mask = attn_mask.repeat(1, self.heads, 1, 1) #bs, heads, seq, seq
        elif (len(attn_mask.size()) == 2): #(bs, seq_len)
            attn_mask = attn_mask[:, None, None, :] #(bs, 1, 1, seq_len) 
        
        attn_mask = (1.0 - attn_mask) * -10000.0
        attn_mask = attn_mask.to(device = prev_hidden_states.device)
        query = self.query(prev_hidden_states) #b, td, total_dim
        
        if (encoder_outputs is None): #in this case, tE = tD
            key = self.key(prev_hidden_states) #b, tE, total_dim 
            value = self.value(prev_hidden_states) #b, tE, total_dim            
        else:
            key = self.key(encoder_outputs) #b, tE, total_dim
            value = self.value(encoder_outputs) #b, tE, total_dim

        #reshape to b, t, heads, dims
        key = key.view((key.size()[0], -1, self.heads, self.dims)) #b, tE, self.heads, self.dims
        query = query.view((query.size()[0], -1, self.heads, self.dims)) #b, tD, self.heads, self.dims
        value = value.view((value.size()[0], -1, self.heads, self.dims)) #b, tE, self.heads, self.dims

        #reshape to b, heads, t, dims <- so that matmul will result in: b, heads, tD, tE
        key = key.permute(0, 2, 1, 3) #b, heads, tE, dims
        query = query.permute(0, 2, 1, 3) #b, heads, tD, dims
        value = value.permute(0, 2, 1, 3) #b, heads, tE, dims

        #first q dot k <- should result in b, heads, t, t
        int_q = torch.matmul(query, key.transpose(-2, -1)) #<-this will give key the shape b, heads, dim, tE
        # => b, heads, tD, tE
        int_q /= self.temp 

        if (attn_mask is not None):
            int_q = int_q + attn_mask
        
        int_q = self.dropout(self.softmax(int_q)) #b, heads, tD, tE

        out = torch.matmul(int_q, value).permute(0,2,1,3).contiguous() #(b, heads, tD, tE) DOT (b, heads, tE, dims) = (b, heads, tD, dims) 
        #   => b, tD, heads, dims
        out = out.view(out.size()[0], out.size()[1], -1) #b, t, model_dims

        """
        if (return_attn):
            return out , int_q
        """

        return out, int_q
        
class TransformerAttentionOut(torch.nn.Module):
    def __init__(self, model_dims = 768, dropout=0.1):
        super(TransformerAttentionOut, self).__init__()
        self.linear = nn.Linear(model_dims, model_dims)
        self.LayerNorm = nn.LayerNorm(model_dims, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, attn_out, inputs):
        int_x = self.dropout(self.linear(attn_out))
        return self.LayerNorm(int_x + inputs)

class TransformerBlock(torch.nn.Module):
    def __init__(self, config, init_as_decoder=False):
        """

        """
        model_dims = config["model_dims"]
        attn_heads = config["blocks"]["attn_heads"]
        attn_dim = config["blocks"]["attn_dims"]
        ff_inner_dims = config["blocks"]["ff_inner_dims"]
        layerNormEps = config["layerNormEps"]
        decoder = init_as_decoder

        super(TransformerBlock, self).__init__()
        assert (model_dims % attn_dim == 0)
        self.decoder = decoder
        self.SelfAttn = TransformerAttention(attn_heads, attn_dim, model_dims=model_dims, causal_mask=False)
        self.SelfAttnOut = TransformerAttentionOut(model_dims=model_dims)
        if (self.decoder):
            self.crossAttn = TransformerAttention(attn_heads, attn_dim, model_dims=model_dims)
            self.crossAttnOut = TransformerAttentionOut(model_dims=model_dims)
        self.FFStack = TransformerFF(ff_inner_dims, model_dims=model_dims)
        #self.LayerNorm = nn.LayerNorm(model_dims, eps=layerNormEps)
        
    def forward(self, inputs, attn_mask = None, decode=False, encoder_outputs=None, return_cross_attn = False, return_self_attn = False):
        int_, att = self.SelfAttn(inputs, attn_mask = attn_mask, causal_mask = decode, encoder_outputs=None, return_attn = return_self_attn)
        resid = self.SelfAttnOut(int_, inputs)
        
        cross_attn = None
        if (decode):
            int_2, cross_attn = self.crossAttn(resid, encoder_outputs=encoder_outputs, return_attn = return_cross_attn)
            int_y = self.crossAttnOut(int_2)
            resid = self.LayerNorm(resid + int_y)
        
        out = self.FFStack(resid)        
        
        return out, att, cross_attn

class TransformerEmbedding(torch.nn.Module):
    def getPositionalBase(self, position):
        return [position/x for x in self.pos_encoding_base]

    def encodePositionalBase(self, base):
        base[0::2] = np.sin(base[0::2])
        base[1::2] = np.cos(base[1::2])
        return base

    def __init__(self, vocab_size, segment_size, model_dims, dropout=0.1, legacy_max_pos=512):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dims)
        self.segments = nn.Embedding(segment_size, model_dims)
        self.legacy_positional_embeddings = nn.Embedding(legacy_max_pos, model_dims)
        self.pos_encoding_base = [np.power(10000, 2*(i//2)/model_dims) for i in range(model_dims)]
        self.positional_embeddings = torch.FloatTensor([self.encodePositionalBase(self.getPositionalBase(k)) for k in range(2048)])
        self.LayerNorm = nn.LayerNorm(model_dims, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, segments, use_legacy_positional_embeddings=False): #shape must be (b, t)
        _len = inputs.size()[1]
        pos_vecs = self.positional_embeddings[:_len, :].unsqueeze(0)
        if (use_legacy_positional_embeddings):
            pos_vecs = self.legacy_positional_embeddings.weight[:_len,:].unsqueeze(0)
        f_embedding =  self.embedding(inputs) + pos_vecs + self.segments(segments)        
        f_embedding = self.LayerNorm(f_embedding)
        f_embedding = self.dropout(f_embedding)
        return f_embedding

class Transformer(torch.nn.Module):
    def __init__(self, config):
        """
            the config file:

            config.vocab_size
            config.model_dims
            config.segments
            config.numblocks
            config.blocks.attn_heads
            config.blocks.attn_dims
            config.blocks.ff_inner_dims
            config.dropout
        """
        super(Transformer, self).__init__()
        self.config = config
        self.embeddings = TransformerEmbedding(config["vocab_size"], config["segments"], config["model_dims"])
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config["numblocks"])])

    def forward(self, inputs, segments, attn_masks, encoder_outputs = None, return_attn = True, use_legacy_positional_embeddings = False):                
        embeddings_out = self.embeddings(inputs, segments, use_legacy_positional_embeddings= use_legacy_positional_embeddings)        
        out_ = embeddings_out
        intermediate_outputs = []
        self_attentions = []
        cross_attentions = []
        for i in range(len(self.blocks)):
            block_out, self_attn, cross_attn = self.blocks[i](out_, attn_mask = attn_masks, encoder_outputs=encoder_outputs, return_self_attn=return_attn, return_cross_attn=return_attn)             
            #print(i, block_out)
            intermediate_outputs.append(block_out)
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)
            out_ = block_out
            block_out = None
        return out_, intermediate_outputs
        
        
if __name__ == "__main__":
    config = {
        "vocab_size":30522,
        "model_dims":768,
        "segments":2,
        "blocks":{
            "attn_heads":12,
            "attn_dims":64,
            "ff_inner_dims":3072
        },
        "numblocks":12,
        "dropout":0.1,
        "layerNormEps":1e-12
    }
    print(config)
    t = Transformer(config)
    from transformers import BertModel, BertConfig
    model = BertModel.from_pretrained("bert-base-uncased")
    print(model.config)