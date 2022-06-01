import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig
from hyperparams import HyperParams as params



class IemocapBertDataset(Dataset):

    def __init__(self,data):
        super(IemocapBertDataset, self).__init__()
        self.speeches, self.sentences, self.labels = data

        config = AutoConfig.from_pretrained(params.bert_model_path) 
        self.tokenizer = AutoTokenizer.from_pretrained(params.bert_model_path,config=config ,do_lower_case=True)
        self.max_sequence_length = params.max_sequence_length

    def __len__(self):
        return len(self.speeches)

    def __getitem__(self, idx):
        
        encoded_dict = self.tokenizer.encode_plus(self.sentences[idx], 
                                                add_special_tokens=True,
                                                max_length = self.max_sequence_length,
                                                padding='max_length',
                                                truncation = True,
                                                return_attention_mask = True, # Construct attn. masks.
                                                return_tensors = 'pt'         # Return pytorch tensors.
                                                )
        return self.speeches[idx], encoded_dict, self.labels[idx] 


class HanAttention(nn.Module):

    def __init__(self,hidden_dim):
        super(HanAttention,self).__init__() 
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                nn.Tanh(),
                                nn.Linear(hidden_dim, 1)
                               )
        self.m = nn.Softmax(dim=1)

    def forward(self, inputs):
        v = self.fc(inputs).squeeze(-1)
        alphas = self.m(v)
        outputs = inputs * alphas.unsqueeze(-1)
        outputs = torch.sum(outputs, dim=1)
        return outputs

def weighted_accuracy(y_true, y_pred):
    return np.sum((np.array(y_pred).ravel() == np.array(y_true).ravel()))*1.0/len(y_true)


def unweighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    classes = np.unique(y_true)
    classes_accuracies = np.zeros(classes.shape[0])
    for num, cls in enumerate(classes):
        classes_accuracies[num] = weighted_accuracy(y_true[y_true == cls], y_pred[y_true == cls])
    return np.mean(classes_accuracies)


def accuracy_7(out, labels):
    return np.sum(np.round(out) == np.round(labels)) / float(len(labels))


def accuracy(out, labels):
    num = 0
    for i in range(len(out)):
        if out[i]>=0 and labels[i]>=0:
            num = num+1
        else:
            if out[i]<0 and labels[i]<0:
                num = num+1
    return num

def F1_score(out):
    outputs = np.argmax(out, axis=1)
    return outputs
