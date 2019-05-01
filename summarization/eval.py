import pickle
import numpy as np
import torch
import torch.nn.functional as F
from SummarizerCell import Seq2SeqDecoderCell as SummarizerCell
from Summarizer import BertSummarizer
from dataOps import *
from losses import *
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
import random
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--eval_model', 
                    type=str, 
                    help='model to continue training', 
                    default="summarizer_BestTrainingLoss.h5")

args = parser.parse_args()

testing = pickle.load(open("./network_testing.pickle", "rb"))
print(len(testing))

_cuda = torch.cuda.is_available()
if (_cuda):
    print("Cuda is available:", torch.cuda.device_count(), "GPUs.")

print("Loading Model...")    

network = BertSummarizer(isCuda = _cuda)
network.load_state_dict(torch.load(args.eval_model))
network.eval()
d, se, m, su, po = genBatch(bs=1, data=testing)
with torch.no_grad():
    _d = network.forwardBert(d, se, m)
    _hs = network.genHiddenState((d.size()[0], 768))
    _prev_word = torch.LongTensor([101]).cuda()
    gen_words = []
    gen_words.append(_prev_word)
    _all_previous_words = gen_words[0]
    if (len(gen_words) > 1):
        _all_previous_words = torch.stack(gen_words, dim=1).squeeze(-1)
    print(_all_previous_words.size())
    words, atts, hs = network.forwardSummary(_d, _hs, _prev_word, _all_previous_words)

    print(words)