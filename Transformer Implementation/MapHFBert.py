from Mapper import *
from Transformer import Transformer
from transformers import BertModel, BertConfig
import torch
import torch.nn.functional as F

def MapFFStack(from_intermediate, from_output, to_):
    MapLinearLayer(from_intermediate.dense, to_.fc1)
    MapLinearLayer(from_output.dense, to_.fc2)
    MapLayerNorm(from_output.LayerNorm, to_.LayerNorm)
    
def MapAttention(from_, to_):
    MapLinearLayer(from_.query, to_.query)
    MapLinearLayer(from_.key, to_.key)
    MapLinearLayer(from_.value, to_.value)

def MapAttentionOut(from_, to_):
    MapLinearLayer(from_.dense, to_.linear)
    MapLayerNorm(from_.LayerNorm, to_.LayerNorm)

def MapBertEmbeddingsToTransformerEmbeddings(from_, to_):
    """
    Huggingface's bert implementation (and presumably the TF model it derives from) uses an embedding matrix
    for positional embeddings as well. This is a deviation from Vaswani et al, where an interleaved sin/cos wave is used.
    """
    MapEmbedding(from_.embeddings.word_embeddings, to_.embeddings.embedding)
    MapEmbedding(from_.embeddings.position_embeddings, to_.embeddings.legacy_positional_embeddings)
    MapEmbedding(from_.embeddings.token_type_embeddings, to_.embeddings.segments)
    MapLayerNorm(from_.embeddings.LayerNorm, to_.embeddings.LayerNorm)
    
def MapBertLayerToBlock(from_, to_):
    MapAttention(from_.attention.self, to_.SelfAttn)
    MapAttentionOut(from_.attention.output, to_.SelfAttnOut)
    MapFFStack(from_.intermediate, from_.output, to_.FFStack)

def MapHFBertModelToTransformer(from_, to_):
    """
        Currently supports only BertModel form Huggingface
    """
    assert "BertForMaskedLM" in from_.config.architectures
    assert from_.config.hidden_size == to_.config["model_dims"]

    print("Mapping Embedding matrices...")
    MapBertEmbeddingsToTransformerEmbeddings(from_, to_)
    
    print("Mapping layers...")
    for i, layer in enumerate(model.encoder.layer):
        MapBertLayerToBlock(layer, to_.blocks[i])

if __name__ == "__main__":
    from transformers import BertTokenizer
    import numpy as np
    sample_config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model = BertModel.from_pretrained("bert-base-uncased", config=sample_config)    

    hfConfig = model.config
    config = {
        "vocab_size":hfConfig.vocab_size,
        "model_dims":hfConfig.hidden_size,
        "segments":hfConfig.type_vocab_size,
        "numblocks":hfConfig.num_hidden_layers,
        "dropout":hfConfig.hidden_dropout_prob,
        "layerNormEps":hfConfig.layer_norm_eps,
        "blocks":{
            "attn_heads":hfConfig.num_attention_heads,
            "attn_dims":int(hfConfig.hidden_size/hfConfig.num_attention_heads),
            "ff_inner_dims":hfConfig.intermediate_size
        }
    }

    print(config)
    t = Transformer(config)
    MapHFBertModelToTransformer(model, t)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    x = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("this is some text [PAD] [PAD]"))).unsqueeze(0)
    print(x)
    seg = torch.LongTensor([1,1,1,1,1,1]).unsqueeze(0)
    att = torch.FloatTensor([1,1,1,1,0,0]).unsqueeze(0)
    t.cuda()

    model.config.output_hidden_states = True
    model.eval()
    t.eval()

    with torch.no_grad():
        check_layer = 12
        a1 = model.forward(x, token_type_ids=seg, attention_mask=att)
        b1 = t.forward(x.cuda(), seg.cuda(), att.cuda(), use_legacy_positional_embeddings=True)        

        print(a1[2][check_layer])
        print(b1[1][check_layer-1])
