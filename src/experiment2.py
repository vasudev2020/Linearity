import numpy as np
import torch
import os,json,argparse, time, datetime

import scipy.optimize as opt
# from random import sample
import random
from nltk.tokenize import sent_tokenize

np.random.seed(100)

from transformers import AutoTokenizer, RobertaModel, ModernBertModel
device = "cuda" if torch.cuda.is_available() else "cpu"


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
            E.append(torch.stack([torch.FloatTensor(self.gloveModel[w]) for w in t.split()]).to(device))

        return E

    def roberta(self, t):
        inputs = self.tokenizer(t, return_tensors="pt",truncation=True, padding=True, return_offsets_mapping=True).to(device)
        out = self.robertaModel(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state.detach()
        E = [torch.stack([e for e,m in zip(emb,mask) if m==1][1:-1]) for emb,mask in zip(out,inputs['attention_mask'])]
        return E
    
    def mbert(self, t):
        inputs = self.tokenizer(t, return_tensors="pt",truncation=True, padding=True, return_offsets_mapping=True).to(device)
        out = self.mbertModel(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state.detach()
        E = [torch.stack([e for e,m in zip(emb,mask) if m==1][1:-1]) for emb,mask in zip(out,inputs['attention_mask'])]
        return E
        
    # Get representation of a text by using LM
    def getEmbs(self, t):
        if self.lm=='glove':    E = self.glove(t)
        elif self.lm=='roberta':  E = self.roberta(t)
        elif self.lm=='mbert':    E = self.mbert(t)
        else:   raise Exception('Undefined lm: '+self.lm)

        return E
        # return [e*self.scale for e in E]
    
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
    count = 0
    # for sent in Sents:
    for i in range(0,len(Sents),batch_size):
        batch_embs = Rep.getEmbs(Sents[i:i+batch_size])
        for embs in batch_embs:
            for i in range(embs.shape[0]-1):
                Pos.append((embs[i].cpu().numpy(),embs[i+1].cpu().numpy()))
                index = list(range(embs.shape[0]))
                index.remove(i+1)
                random.seed(100)
                Neg.append((embs[i].cpu().numpy(),embs[random.sample(index,1)[0]].cpu().numpy()))
                # Neg.append((embs[i],embs[i+3 if i < embs.shape[0]-3 else i-2]))
            for e in embs:  E.append(e)
        count+=len(batch_embs)
    assert count==len(Sents)

    print("Data prep time:",datetime.timedelta(seconds=time.time()-t))

    # t=time.time()
    # Et = torch.stack(E).unique(dim=0).unsqueeze(0)
    # dist = torch.cdist(Et,Et,p=2).squeeze()
    # dist.fill_diagonal_(100000)
    # mod = float(dist.min()/2)
    # print("MoD calc time:",datetime.timedelta(seconds=time.time()-t))

    t=time.time()
    Pr = np.stack([np.concatenate((t1,t2)) for t1,t2 in Pos]).T
    Nr = np.stack([np.concatenate((t1,t2)) for t1,t2 in Neg]).T
    EoA, _ = LinApprox(Pr,Nr,rank=rank,verbose=False)
    print("Lin Approx time:",datetime.timedelta(seconds=time.time()-t))

    print("Error of approximation:", EoA)
    print("Avg error of approximation:", EoA/Pr.shape[1])
    # print("Avg normalized error of approximation:", EoA/(Pr.shape[1]*mod))


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
