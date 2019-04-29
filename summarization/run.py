import pickle
import numpy as np
import torch
import torch.nn.functional as F
from SummarizerCell import Seq2SeqDecoderCell as SummarizerCell
from dataOps import *
from losses import *
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
import random

def train(bs = 5, 
            epochs = 1,
            batches = 1000,
            network=None, 
            _data=None, 
            bert=None, 
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
            gen_atts = []
            coverages = []
            gen_logits = []
            act_words = []

            _d, _ = bert(d, se, m, output_all_encoded_layers = False)
            _d = _d * m.unsqueeze(-1).float()   

            optimizer.zero_grad()
            coverage = torch.zeros((d.size()[0], d.size()[1])).cuda()
            zeros = torch.zeros((d.size()[0], d.size()[1])).cuda()

            for i in range(max_summary_length):        
                act_words.append(su[:,i])
                new_words, atts, _hs = network.forward(_d, _hs, _prev_word)
                actual_words = F.softmax(new_words, dim=-1)
                actual_words = torch.max(actual_words, dim=-1)[1]
                _prev_word = actual_words.unsqueeze(-1)
                if (random.random() < teacher_forcing_rate):
                    gen_words.append(su[:,i].detach())
                else:
                    gen_words.append(_prev_word.detach())

                if (i > 0): #coverage loss will be 0 for the first step. 
                    gen_atts.append(atts)
                    coverages.append(coverage + zeros)
                
                coverage = coverage + atts.squeeze(-1)
                gen_logits.append(new_words)

            gen_logits = torch.stack(gen_logits, dim=0).view(-1, 30000)
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

print("\n\nSummarizer, summarizer!")
print("Loading data...")
all_data = pickle.load(open("./training_0.pickle", "rb"))
max_doc_length = 100
max_summary_length = 10
_cuda = torch.cuda.is_available()
if (_cuda):
    print("Cuda is available.")
print("Creating Model...")    
sc = SummarizerCell(isCuda=_cuda)
optimizer = torch.optim.Adam(sc.parameters(), lr=1e-3)

if (_cuda):
    sc.cuda()

_bert = None
bert_model = "bert-base-uncased"
if (_cuda):
    _bert = BertModel.from_pretrained(bert_model).cuda()
else:
    _bert = BertModel.from_pretrained(bert_model)
_bs = 30
print("Training...\n")
train(bs=_bs, 
        epochs = 5,
        batches = 3000,
        network=sc, _data=all_data, bert=_bert, optim=optimizer, cuda=_cuda)