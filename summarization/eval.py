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

parser.add_argument('--bs', 
                    type=int, 
                    help='batch size', 
                    default=1)

args = parser.parse_args()

testing = pickle.load(open("./network_testing.pickle", "rb"))
print(len(testing))

_cuda = torch.cuda.is_available()
if (_cuda):
    print("Cuda is available:", torch.cuda.device_count(), "GPUs.")

print("Loading Model...")    

network = BertSummarizer(isCuda = _cuda)
network.load_state_dict(torch.load(args.eval_model))
if (_cuda):
    network.cuda()
network.eval()
d, se, m, su, po = genBatch(bs=args.bs, 
                            data=testing,
                            _cuda=_cuda)
_prev_word = torch.LongTensor([101]).cuda()
_prev_word = _prev_word.repeat(args.bs, 1)
with torch.no_grad():
    _d = network.forwardBert(d, se, m)
    _hs = network.genHiddenState((d.size()[0], 768))    
    gen_words = []
    beams = []
    gen_words.append(_prev_word)
    for i in range(10):
        _all_previous_words = gen_words[0]
        if (len(gen_words) > 1):
            _all_previous_words = torch.stack(gen_words, dim=1).squeeze(-1)
        words, atts, _hs = network.forwardSummary(_d, _hs, _prev_word, _all_previous_words)
        _words = F.softmax(words, dim=-1)
        _xword = torch.topk(_words, 1, dim=-1)[1]
        _prev_word = _xword
        beams.append(torch.topk(_words, 5, dim=-1)[1])
        gen_words.append(_prev_word)

    print("\n\nActuals")
    print(su)
    print("\n\nPredictions:")
    gen_words = torch.stack(gen_words)
    print(gen_words.flatten().detach().cpu().numpy()[1:])
    print("\n\n\nPotential Beams")
    beams = torch.stack(beams, dim=1)
    print(beams.detach().cpu().numpy())
    