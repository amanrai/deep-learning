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

testing = pickle.load(open("./network_testing.pickle", "rb"))

