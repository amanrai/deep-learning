import numpy as np
import torch
import json

def pointLessFunction():
	print("This is a pointless function!")

def _save(model, cause, epoch_losses):
    print("\n\t\t...saving model for cause", cause)
    torch.save(model.state_dict(), "./summarizer_" + cause + ".h5")
    with open("./Summarizer_training_cycle_"  + cause + ".json", "w") as f:            
        f.write(json.dumps(
            {
                "training_losses":epoch_losses
            }
        ))
        f.close()

def modelSaver(model, losses):
    if (losses[-1] == np.min(losses)):
        _save(model, "BestTrainingLoss", losses)
        
    _save(model, "LastEpoch", losses)

def genBatch(bs = 5, validation = False, data = None, _cuda=True, max_doc_length=100, max_summary_length=10):
    if (data == None):
        return
        
    indices = np.random.randint(0, len(data), (bs,))
    docs = [data[index]["story_tokens"] for index in indices]
    _pointers = [data[index]["pointers"] for index in indices]
    
    documents = []
    summaries = []
    pointers = []
    for doc in docs:
        doc = doc[1:]
        doc.insert(0, 101) #<- 101 is the token id for the CLS token
        while (len(doc) < max_doc_length):
            doc.append(0)
        doc = doc[:max_doc_length]
        documents.append(doc)
    
    sums = [data[index]["summary_tokens"] for index in indices]
    for k in range(len(sums)):
        summ = sums[k]
        _point = _pointers[k]
        while (len(summ) < max_summary_length):
            summ.append(0)
        summ = summ[:max_summary_length]
        summaries.append(summ)
        points = np.zeros((len(summ),))
        _point_choice = np.asarray(_point) < max_summary_length
        _point = np.asarray(_point)[_point_choice]
        if (len(_point) > 0):
            points[_point] = 1
        pointers.append(points)
        
    if _cuda:
        documents = torch.LongTensor(documents).cuda()
        summaries = torch.LongTensor(summaries).cuda()
        segments = torch.zeros_like(documents).cuda()
        pointers = torch.FloatTensor(pointers).cuda()
    else:
        documents = torch.LongTensor(documents)
        summaries = torch.LongTensor(summaries)
        segments = torch.zeros_like(documents)
        pointers = torch.FloatTensor(pointers)
    mask = documents > 0
    
    return documents, segments, mask, summaries, pointers