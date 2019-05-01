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

d, se, m, su, po = genBatch(bs=1, data=testing)
print(d)