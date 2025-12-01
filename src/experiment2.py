import numpy as np
import torch
import os,json,argparse, time, datetime

import scipy.optimize as opt
# from random import sample
import random
from nltk.tokenize import sent_tokenize

np.random.seed(100)

from transformers import AutoTokenizer, RobertaModel, ModernBertModel


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
            self.robertaModel = RobertaModel.from_pretrained("roberta-base",add_pooling_layer=False)
        if lm=='mbert':
            self.tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
            self.mbertModel = ModernBertModel.from_pretrained('answerdotai/ModernBERT-base')

            # tt = self.tokenizer('this is a sample sentence')['input_ids']
            # print(self.tokenizer.batch_decode(tt[1:-1]))

    def glove(self, t):
        T = t if type(t)==list else [t]
        E = []
        for t in T:
            for w in t.split(): 
                if w not in self.gloveModel:   self.gloveModel[w]= np.random.rand(300)
            E.append(np.stack([self.gloveModel[w] for w in t.split()]))
        return E

    def roberta(self, t):
        inputs = self.tokenizer(t, return_tensors="pt",truncation=True, padding=True, return_offsets_mapping=True)
        # out = self.robertaModel(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state[0][1:-1].detach().numpy()
        out = self.robertaModel(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state.detach().numpy()
        E = [np.stack([e for e,m in zip(emb,mask) if m==1][1:-1]) for emb,mask in zip(out,inputs['attention_mask'])]

        for e,tt in zip(E,t):   assert e.shape[0]==len(self.tokenizer(tt)['input_ids'])-2
        return E
    
    def mbert(self, t):
        inputs = self.tokenizer(t, return_tensors="pt",truncation=True, padding=True, return_offsets_mapping=True)
        # out = self.mbertModel(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state[0][1:-1].detach().numpy()

        out = self.mbertModel(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state.detach().numpy()
        E = [np.stack([e for e,m in zip(emb,mask) if m==1][1:-1]) for emb,mask in zip(out,inputs['attention_mask'])]

        for e,tt in zip(E,t):   assert e.shape[0]==len(self.tokenizer(tt)['input_ids'])-2
        return E

        # return out

    # Get representation of a text by using LM
    def getEmbs(self, t):
        if self.lm=='glove':    return self.glove(t)
        if self.lm=='roberta':  return self.roberta(t)
        if self.lm=='mbert':    return self.mbert(t)

        raise Exception('Undefined lm: '+self.lm)
    
    # # Get representation of a list of pairs of texts by using LM
    # def getAvgEmbs(self, T):
    #     return np.mean(self.getEmbs(T),axis=0)

    # def getRepresentation(self, T)
    #     return np.stack([np.concatenate((self.getTextEmb(t1),self.getTextEmb(t2))) for t1,t2 in T]).T

# Pr: (2*dim, Np), Nr: (2*dim, Nn)
def LinApprox(Pr, Nr, rank=None, verbose=False):
    assert Pr.shape[0]==Nr.shape[0]
    if verbose: print('Number of related pairs  :',Pr.shape[1])
    if verbose: print('Number of unrelated pairs:',Nr.shape[1])
    if verbose: print('Combined embedding dimension:',Pr.shape[0])

    if rank is None:   rank = Pr.shape[0]

    U,S,V = np.linalg.svd(Pr)

    C = np.square(S)
    C = np.pad(C,(0,Pr.shape[0]-C.shape[0]))[:rank] # Truncate C to size r
    U = U[:,:rank]                                  # Truncate U to shape (U.shpae[0],rank)
    A = np.square(np.matmul(Nr.T, U))

    if verbose: print('LP is starting with',C.shape, 'variables and', A.shape[0], 'constraints')
    res = opt.linprog(C, A_ub=-A, b_ub = -np.ones(A.shape[0]))
    
    try:    M = np.matmul(np.diag(np.sqrt(res.x)), U.T)
    except: return 1000000, None

    if round(np.min(np.linalg.norm(np.matmul(M, Nr),axis=0)),2)<1:   print("Infeasible solution")
    assert M.shape[0]==rank and M.shape[1]==Pr.shape[0]

    if verbose: print('Error of approximation  :',res.fun)

    return res.fun, M

# Return N sentences from Wiki
def readWiki(size):
    wiki_path='../../Data/Wiki'
    # print(os.listdir('../../data/Wiki'))
    dirs = os.listdir(wiki_path)
    dataset = []
    for dir in sorted(dirs):
        files = os.listdir(wiki_path+'/'+dir)
        for f in sorted(files):
            for line in open(wiki_path+'/'+dir+'/'+f):
                d = json.loads(line)
                samples = sorted(d['text'].split('\n'))
                samples = [ss.strip() for s in samples for ss in sent_tokenize(s) if len(ss.strip())>0 and len(ss.split())<512]
                dataset+=samples
                if len(dataset)>=size:
                    dataset = sorted(list(set(dataset)))
                    if len(dataset)>=size:  return dataset[:size]
                        
    dataset = list(set(dataset))
    return dataset[:size]

def main(lm, size, batch_size, rank=None):
    print(lm, size)
    t=time.time()
    Sents = readWiki(size=size) 
    print("Data load time:",datetime.timedelta(seconds=time.time()-t))

    Rep = Representation(lm=lm)
    Pos,Neg,E = [],[],[]

    t=time.time()
    # for sent in Sents:
    for i in range(0,len(Sents),batch_size):
        batch_embs = Rep.getEmbs(Sents[i:i+batch_size])
        assert len(batch_embs)==batch_size
        for embs in batch_embs:
            for i in range(embs.shape[0]-1):
                Pos.append((embs[i],embs[i+1]))
                index = list(range(embs.shape[0]))
                index.remove(i+1)
                random.seed(100)
                Neg.append((embs[i],embs[random.sample(index,1)[0]]))
            # Neg.append((embs[i],embs[i+3 if i < embs.shape[0]-3 else i-2]))

            for e in embs:  E.append(e)
    print("Data prep time:",datetime.timedelta(seconds=time.time()-t))


    t=time.time()
    Et = [torch.FloatTensor(e) for e in E]
    Et = torch.stack(Et).unsqueeze(0)
    dist = torch.cdist(Et,Et,p=2).squeeze()
    # dist.fill_diagonal_(0)
    dist.diagonal().zero_()
    dist = dist.flatten().tolist()
    # dist = []
    # for e1 in Et: 
    #     dist += torch.linalg.norm(torch.stack([e1-e2 for e2 in Et]),dim=-1).squeeze().tolist()

    dist = [d for d in dist if d!=0]
    print(len(dist))
    mod_1 = min(dist)/2

    print("MoD calc time:",datetime.timedelta(seconds=time.time()-t))

    t=time.time()
    dist = [np.linalg.norm(e1-e2) for e1 in E for e2 in E]
    dist = [d for d in dist if d!=0]
    mod = min(dist)/2
    print("MoD calc time:",datetime.timedelta(seconds=time.time()-t))

    print(mod, mod_1)

    t=time.time()
    Pr = np.stack([np.concatenate((t1,t2)) for t1,t2 in Pos]).T
    Nr = np.stack([np.concatenate((t1,t2)) for t1,t2 in Neg]).T
    EoA, _ = LinApprox(Pr,Nr,rank=rank,verbose=False)
    print("Lin Approx time:",datetime.timedelta(seconds=time.time()-t))

    print("Avg error of approximation:", EoA/Pr.shape[1])
    print("Avg normalized error of approximation:", EoA/(Pr.shape[1]*mod))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    '''Hyperparamters related to Dataset'''
    parser.add_argument('--lm', type=str, default='glove', help='representation model: glove/roberta/mbert')
    parser.add_argument('--size', type=int, default=100, help='Number of Wiki sentences')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')

    
    args = parser.parse_args()
    t0=time.time()

    main(lm=args.lm, size=args.size, batch_size=args.batch_size)

    print('Total Execution Time:',datetime.timedelta(seconds=time.time()-t0))
