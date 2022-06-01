#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:11:29 2020

@author: adam
"""
import os
import torch
import random
from torch import nn
import numpy as np
from tqdm import tqdm
from hyperparams import HyperParams as hp
from torch.utils.data import DataLoader
import torch.nn.functional as F
from modules import weighted_accuracy,unweighted_accuracy
from model import NIONM
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from modules import IemocapBertDataset
import pickle
import wandb

import warnings
warnings.filterwarnings("ignore")
np.random.seed(960124)

# is cuda available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"start train with device : {torch.cuda.get_device_name(0)}")



def clean_labels(sp,tp, p, t):
    _sp, _tp, _p, _t = [], [], [], []
    for i in range(len(t)):
        if sp[i] == tp[i] == p[i] == t[i]:
            continue
        else:
            _sp.append(sp[i])
            _tp.append(tp[i])
            _p.append(p[i])
            _t.append(t[i])
    return _sp, _tp, _p, _t 


def evaluate(model,data_loader,best_wa,history_was):
    model.eval()
    losses = []
    y_trues = []
    y_predicts = []
    text_predicts = []
    speech_predicts = []
    probs = []
    #global i
    for sample_speech,sample_sentence,sample_labels in data_loader:

        y_true = sample_labels.view(-1).numpy().tolist()
        sample_speech = sample_speech.float().to(device)
        sample_sentence = {k:v.squeeze().long().to(device) for k , v in sample_sentence.items()}
        sample_labels = sample_labels.long().to(device)

        speech_prob, text_prob, prob = model(sample_speech,sample_sentence)

        predict = prob.cpu().detach()
        y_predict = predict.argmax(1).numpy().tolist()
        speech_predicts += speech_prob.cpu().detach().argmax(1).numpy().tolist()
        text_predicts += text_prob.cpu().detach().argmax(1).numpy().tolist()


        loss = crossen(prob, sample_labels.view(-1))
        losses.append(loss.cpu().item())
        probs += predict.numpy().tolist()
        y_trues += y_true
        y_predicts += y_predict


    
    vs = np.vstack((speech_predicts, text_predicts, y_predicts, y_trues))
    _vs = np.vstack(clean_labels(speech_predicts, text_predicts, y_predicts, y_trues))
    
    y_trues = np.asarray(y_trues)
    y_predicts = np.asarray(y_predicts)
    print(classification_report(y_trues, y_predicts, target_names= hp.available_emotions))
    wa = weighted_accuracy(y_trues,y_predicts)
    ua = unweighted_accuracy(y_trues,y_predicts)

    if best_wa < wa:
        cov = np.corrcoef(vs)
        print(cov)
        print("clean:")
        cov = np.corrcoef(_vs)
        print(cov)

        best_wa = wa
        torch.save(model.state_dict(), f'./model/{hp.data_name}.pth')

    if len(history_was)>hp.patience and max(history_was[-hp.patience:]) < best_wa and hp.learn_rate>hp.min_learn_rate: 
        history_was = []
        hp.learn_rate = hp.decay_rate * hp.learn_rate 
    history_was.append(wa)
    print(f"evaluate loss: {np.mean(losses):.3f} \t learn rate {hp.learn_rate:.6f} \t wa: {wa:.3f} \t ua: {ua:.3f} \t best_accuracy {best_wa:.3f}")
    model.train()

    return best_wa,history_was, np.mean(losses), wa, ua

def load_dataset(path):
    print("load dataset form exists pkl file")
    with open(path, 'rb') as f:
        dataset =  pickle.load(f)
    return dataset

def load_cmu_mosei_dataset(path):
    print("load dataset form exists pkl file")
    paths = [os.path.join(path, filename) for filename in os.listdir(path)]
    print(paths)
    speeches,sentences,labels = [],[],[]
    for path in paths:
        with open(path, 'rb') as f:
            samples =  pickle.load(f)
            speeches+=samples["audio"]
            sentences+=samples["sentence"]
            labels+=samples["label"] 
    return (speeches,sentences,labels)

def train_test_splite(speeches,sentences,labels,test_size=0.2):
    speeches,sentences, labels = np.asarray(speeches), np.asarray(sentences), np.asarray(labels)
    random_index = np.random.permutation(len(sentences)) 
    split_point = int((1-test_size)*len(sentences))
    train_sample = random_index[:split_point] 
    test_sample = random_index[split_point:] 
    train_speeches = speeches[train_sample]
    train_sentences = sentences[train_sample]
    train_labels = labels[train_sample]
    test_speeches = speeches[test_sample]
    test_sentences = sentences[test_sample]
    test_labels = labels[test_sample]
    return (train_speeches,train_sentences,train_labels),(test_speeches,test_sentences,test_labels)

def check_labels(speeches, sentences, labels, dataset_name = "iemocap" ):
    _speeches, _sentences, _labels = [], [], []
    for a, t, l in zip(speeches, sentences, labels):
        if dataset_name == "iemocap":
            if l == 1:
                _speeches.append(a)
                _sentences.append(t)
                _labels.append(0)
            if l == 2:
                _speeches.append(a)
                _sentences.append(t)
                _labels.append(1)
        else:
            if l <-1:
                _speeches.append(a)
                _sentences.append(t)
                _labels.append(0)
            if 1 < l:
                _speeches.append(a)
                _sentences.append(t)
                _labels.append(1)
    return _speeches, _sentences, _labels

if hp.data_name == "cmu_mosi":
    samples = load_dataset(hp.self_dataset)
    speeches, sentences, labels = samples["audio"], samples["sentence"], samples["label"]
elif hp.data_name == "cmu_mosei":
    speeches, sentences, labels = load_cmu_mosei_dataset(hp.self_dataset)
else:
    speeches, _, _, sentences, labels, _, _ = load_dataset(hp.self_dataset)
    sentences = [' '.join(words) for words in sentences]


speeches, sentences, labels = check_labels(speeches, sentences, labels, hp.data_name)

print('all data: ', len(labels))
print(set(labels))
# split dataset
train_data, test_data = train_test_splite(speeches, sentences, labels, hp.test_size)
print('train data : ', len(train_data[0]),'test data: ',len(test_data[0]))
# create dataset & model 
train_data, test_data = map(IemocapBertDataset,[train_data, test_data])
model = NIONM(hp).to(device)
wandb.init(project=f"{hp.data_name}_base")
wandb.watch(model)
print("NIONM model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
# creat dataloader
train_loader = DataLoader(train_data, batch_size= hp.batch_size, shuffle=True)
test_loader =  DataLoader(test_data, batch_size= hp.vaild_batch_size, shuffle=True)

# create loss funcation
crossen = nn.CrossEntropyLoss()
def compute_kl_loss(p, q):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss, p_loss, q_loss

# create optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = hp.learn_rate, betas=(0.9, 0.999), weight_decay=hp.weight_decay)

# init best accuracy
best_wa = 0
history_was =[]
for epoch in range(hp.epochs):
    losses = []
    y_trues = []
    y_predicts = []

    kl_losses = []
    ce_losses1 = []
    ce_losses2 = []
    p_losses = []
    q_losses = []
    for sample_speech,sample_text,sample_label in tqdm(train_loader):
        optimizer.zero_grad()
        try:
            sample_text = {k:v.squeeze().long().to(device) for k , v in sample_text.items()}
            sample_speech = sample_speech.float().to(device)
            sample_label = sample_label.long().to(device)
        
            speech_prob1, text_prob1, prob_1 = model(sample_speech,sample_text)
            speech_prob2, text_prob2, prob_2 = model(sample_speech,sample_text)

            loss_1 = crossen(prob_1, sample_label.view(-1))
            loss_1 += crossen(speech_prob1, sample_label.view(-1))
            loss_1 += crossen(text_prob1, sample_label.view(-1))
            loss_2 = crossen(prob_2, sample_label.view(-1))
            loss_2 += crossen(speech_prob2, sample_label.view(-1))
            loss_2 += crossen(text_prob2, sample_label.view(-1))


            loss_3, p_loss, q_loss = compute_kl_loss(prob_1, prob_2)


            loss = loss_1 + loss_2 + loss_3

            # backward 
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            loss.backward()
            optimizer.step()

            p_losses.append(p_loss.cpu().item())
            q_losses.append(q_loss.cpu().item())
            kl_losses.append(loss_3.cpu().item())
            ce_losses1.append(loss_1.cpu().item())
            ce_losses2.append(loss_2.cpu().item())
            losses.append(loss.cpu().item())

            y_trues += sample_label.cpu().view(-1).numpy().tolist()
            y_predicts += prob_1.cpu().argmax(1).numpy().tolist()
        except Exception as e:
            print("error: ", e)
            

    wa = weighted_accuracy(y_trues,y_predicts)
    ua = unweighted_accuracy(y_trues,y_predicts)
    print(f"----------epoch {epoch} / {hp.epochs}-----------------------")
    print(f"train    loss: {np.mean(losses):.3f} \t learn rate {hp.learn_rate:.6f} \t wa: {wa:.3f} \t ua: {ua:.3f} \t")
    
    wandb_log_dict = {}
    wandb_log_dict['Train/Loss'] = np.mean(losses)
    wandb_log_dict['Train/ce_loss_1'] = np.mean(ce_losses1)
    wandb_log_dict['Train/ce_loss_2'] = np.mean(ce_losses2)
    wandb_log_dict['Train/p_loss'] = np.mean(p_losses)
    wandb_log_dict['Train/q_loss'] = np.mean(q_losses)
    wandb_log_dict['Train/kl_loss'] = np.mean(kl_losses)
    wandb_log_dict['Train/wa'] = wa
    wandb_log_dict['Train/ua'] = ua

    best_wa,history_was, loss, wa, ua = evaluate(model, test_loader, best_wa, history_was)

    wandb_log_dict['Test/Loss'] = np.mean(losses)
    wandb_log_dict['Test/wa'] = wa
    wandb_log_dict['Test/ua'] = ua
    wandb.log(wandb_log_dict, step=epoch)


