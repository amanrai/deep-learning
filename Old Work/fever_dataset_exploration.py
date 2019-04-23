print("Loading data...")
import json
data = []
with open("../Data/Fever/train.jsonl", "r") as f:
    lines = f.read().split("\n")
    for line in lines:
        data.append(json.loads(line))

print("Loading wiki data...")
base_data = {}
ignored_lines = []
base_data_path = "../Data/Fever/wiki-pages/"
from os import listdir
base_files = listdir(base_data_path)
for file in base_files:
    with open(base_data_path + file, "r") as f:
        lines = f.read().split("\n")
        for line in lines:
            try:
                _js = json.loads(line)
                base_data[_js["id"]] = _js
            except:
                ignored_lines.append(line)

print("Extracting usable data...")
from nltk.tokenize import sent_tokenize
count = 0
usable_data = []
not_verifiable = 0
verifiable_but_data_missing = 0
sentence_not_found_in_evidence = 0
for dp in data:
    usable = True
    count += 1
    if (dp["verifiable"] == "VERIFIABLE"):
        if (len(dp["evidence"]) > 0):
            for evidence in dp["evidence"][0]:
                if (evidence[2] in base_data):
                    text = sent_tokenize(base_data[evidence[2]]["text"])
                    try:
                        a = len(text[evidence[3]])
                    except:
                        sentence_not_found_in_evidence = sentence_not_found_in_evidence + 1
                        usable = False
                else:
                    verifiable_but_data_missing = verifiable_but_data_missing + 1
                    usable = False
        else:
            usable=False
    else:
        not_verifiable += 1
        usable=False
    if (usable):
        usable_data.append(dp)
        #print(dp["label"])
print(len(usable_data), not_verifiable, verifiable_but_data_missing, sentence_not_found_in_evidence)

classes = {
    "SUPPORTS":1,
    "REFUTES":0
}
  
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_claim_len = 30
max_evidence_len = 512
padding_string = "[PAD]" # from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py
verifiable_data = [] 
count = 0
for line in usable_data:
    count = count + 1
    tokenized_text = tokenizer.tokenize(line["claim"])
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    _d = {
        "claim":indexed_tokens,
        "class":classes[line["label"]]
    }
    _evidence = []
    add = True        
    if (line["verifiable"] == "VERIFIABLE"):
        evid = ""
        for evidence in line["evidence"]:
            evid = evid + evidence + "\n"
        if (len(evid) > 10):
            evid_tokens = tokenizer.tokenize(evid)
            while (len(evid_tokens) < max_evidence_len):
                evid_tokens.append(padding_string)
            _evidence.extend(tokenizer.convert_tokens_to_ids(evid_tokens)[:512])
        else:
            add = False
    _d["evidence"] = _evidence
    if (add):
        verifiable_data.append(_d)
    print(str(count) + "/" + str(len(usable_data)), end="\r")
    #print(len(_d["evidence"]))
    #print(_d)

import pickle
pickle.dump(verifiable_data, open("../Data/usable_verifiable_fever_data.pickle", "wb"))
