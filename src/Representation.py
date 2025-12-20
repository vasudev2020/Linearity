import numpy as np
import torch

from transformers import AutoTokenizer, RobertaModel, ModernBertModel
device = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(100)

class Representation:
    def __init__(self,lm='glove'): # GloVe, BERT, RoBERTa, ModernBERT
        self.lm = lm
        if lm=='glove':
            f = open('../../Data/glove.840B.300d.txt','r')
            self.gloveModel = {}
            for line in f:
                splitLines = line.split()
                if len(splitLines)!=301:    continue
                self.gloveModel[splitLines[0]] = np.array([float(value) for value in splitLines[1:]])
            print(len(self.gloveModel)," words loaded!")
        if lm=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base",clean_up_tokenization_spaces=True)
            self.robertaModel = RobertaModel.from_pretrained("roberta-base",add_pooling_layer=False).to(device)
        if lm=='mbert':
            self.tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
            self.mbertModel = ModernBertModel.from_pretrained('answerdotai/ModernBERT-base').to(device)

            # tt = self.tokenizer('this is a sample sentence')['input_ids']
            # print(self.tokenizer.batch_decode(tt[1:-1]))

    def glove(self, t):
        T = t if type(t)==list else [t]
        E = []
        for t in T:
            for w in t.split(): 
                if w not in self.gloveModel:   self.gloveModel[w]= np.random.rand(300)
            # E.append(np.stack([self.gloveModel[w] for w in t.split()]))
            E.append(np.stack([self.gloveModel[w] for w in t.split()]))
        # print(len(E),E[0].shape)
        return E

    def roberta(self, t):
        inputs = self.tokenizer(t, return_tensors="pt",truncation=True, padding=True, return_offsets_mapping=True).to(device)
        out = self.robertaModel(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state.detach()
        E = [torch.stack([e for e,m in zip(emb,mask) if m==1][1:-1]).cpu().numpy() for emb,mask in zip(out,inputs['attention_mask'])]
        return E
    
    def mbert(self, t):
        inputs = self.tokenizer(t, return_tensors="pt",truncation=True, padding=True, return_offsets_mapping=True).to(device)
        out = self.mbertModel(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state.detach()
        E = [torch.stack([e for e,m in zip(emb,mask) if m==1][1:-1]).cpu().numpy() for emb,mask in zip(out,inputs['attention_mask'])]
        return E
        
    # Get representation of a text by using LM
    def getEmbs(self, t):
        if self.lm=='glove':    E = self.glove(t)
        elif self.lm=='roberta':  E = self.roberta(t)
        elif self.lm=='mbert':    E = self.mbert(t)
        else:   raise Exception('Undefined lm: '+self.lm)
        return E
        # return [e*self.scale for e in E]
    def getPairEmbs(self,pairs):
        prefs = [p for p,_ in pairs]
        suffs = [s for _,s in pairs]
        
        pref_embs = [np.mean(e,axis=0) for e in self.getEmbs(prefs)]
        suff_embs = [np.mean(e,axis=0) for e in self.getEmbs(suffs)]

        return [np.concatenate([p,s]) for p,s in zip(pref_embs,suff_embs)]

    # # Get representation of a list of pairs of texts by using LM
    # def getAvgEmbs(self, T):
    #     return np.mean(self.getEmbs(T),axis=0)

    # def getRepresentation(self, T)
    #     return np.stack([np.concatenate((self.getTextEmb(t1),self.getTextEmb(t2))) for t1,t2 in T]).T
