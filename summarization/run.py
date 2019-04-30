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

parser.add_argument('--reuse_saved_model', type=str, help='model to continue training')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batches_per_epoch', type=int, default=100)
parser.add_argument('--bs', type=int, default=22)
parser.add_argument('--summary_length', type=int, default=10)
parser.add_argument('--doc_length', type=int, default=100)
parser.add_argument('--bert_model', type=str, default="bert-base-uncased")
parser.add_argument('--tf_rate', type=float, default=0.25)
parser.add_argument('--lr', type=float, default=1e-4)

args = parser.parse_args()
epochs = args.epochs
batches_per_epoch = args.batches_per_epoch
max_doc_length = args.doc_length
max_summary_length = args.summary_length
_bs = args.bs 
bert_model = args.bert_model
tf_rate = args.tf_rate
lr = args.lr

def train(bs = 5, 
            epochs = 1,
            batches = 1000,
            network=None, 
            _data=None, 
            optim = None,
            cuda = True,
            teacher_forcing_rate = 0.25,
            max_doc_length = 100,            
            max_summary_length = 10):    

    epoch_losses = []
    for epoch in range(epochs):

        batch_losses = []

        for batch in range(batches):

            d, se, m, su, po = genBatch(bs = bs,
                                        data=_data, 
                                        _cuda = cuda, 
                                        max_doc_length = max_doc_length, 
                                        max_summary_length=max_summary_length)

            _hs = network.genHiddenState((d.size()[0], 768))
            _prev_word = None

            if (cuda):
                _prev_word = torch.LongTensor([101]).cuda()
            else:
                _prev_word = torch.LongTensor([101]).cuda()

            _prev_word = _prev_word.repeat(bs, 1)
            gen_words = []
            gen_words.append(_prev_word)
            gen_atts = []
            coverages = []
            gen_logits = []
            act_words = []

            optimizer.zero_grad()
            
            _d = network.forwardBert(d, se, m)
            coverage = torch.zeros((d.size()[0], d.size()[1])).cuda()
            zeros = torch.zeros((d.size()[0], d.size()[1])).cuda()

            for i in range(max_summary_length):
                _all_previous_words = gen_words[0]
                if (len(gen_words) > 1):
                    _all_previous_words = torch.stack(gen_words, dim=1)
                    print(_all_previous_words.size())        
                act_words.append(su[:,i])
                new_words, atts, _hs = network.forwardSummary(_d, _hs, _prev_word, _all_previous_words)
                actual_words = F.softmax(new_words, dim=-1)
                actual_words = torch.max(actual_words, dim=-1)[1]
                _prev_word = actual_words.unsqueeze(-1)
                if (random.random() < teacher_forcing_rate):
                    print("Teacher forcing!")
                    _prev_word = su[:,i].detach().unsqueeze(-1)
                
                print(_prev_word.size())
                gen_words.append(_prev_word)

                if (i > 0): #coverage loss will be 0 for the first step. 
                    gen_atts.append(atts)
                    coverages.append(coverage + zeros)
                
                coverage = coverage + atts.squeeze(-1)
                gen_logits.append(new_words)


            gen_logits = torch.stack(gen_logits, dim=0).view(-1, 30522)
            act_words = torch.stack(act_words, dim=0).view(-1).squeeze(-1)
            coverages = torch.stack(coverages, dim=0).view(-1, d.size()[1])
            gen_atts = torch.stack(gen_atts, dim=0).view(-1, d.size()[1])

            w_loss = wordLoss(gen_logits, act_words)
            c_loss = coverageLoss(coverages, gen_atts)
            loss = w_loss + c_loss
            loss.backward()

            batch_losses.append(loss.data.item())
            _loss_str = "Epoch:" + str(epoch + 1) + \
                        " (" + str(batch+1) + "/" + str(batches) + "); " + \
                        "avg loss:" + str(np.round(np.mean(batch_losses), 5)) + \
                        " (" + str(np.round(w_loss.data.item(), 5)) + ";" + str(np.round(c_loss.detach().cpu().numpy(), 5)) + ") "

            print(_loss_str, end="\r")
            optim.step()
        epoch_losses.append(np.mean(batch_losses))
        modelSaver(network, epoch_losses)
        print("\n")

print("Summarizer...\nLoading data...")
all_data = pickle.load(open("./training_0.pickle", "rb"))

_cuda = torch.cuda.is_available()
if (_cuda):
    print("Cuda is available:", torch.cuda.device_count(), "GPUs.")
    _bs = args.bs * torch.cuda.device_count()

print("Creating Model...")    

network = BertSummarizer(isCuda = _cuda)

if (args.reuse_saved_model is not None):
    print("\treusing weights from:", args.reuse_saved_model)
    network.load_state_dict(torch.load(args.reuse_saved_model))

optimizer = torch.optim.Adam(network.parameters(), lr=lr)

if (_cuda):
    network.cuda()

print("Training...\n")
train(bs=_bs, 
        epochs = epochs,
        batches = batches_per_epoch,
        network=network, 
        _data=all_data,  
        optim=optimizer, 
        cuda=_cuda,
        teacher_forcing_rate=tf_rate)