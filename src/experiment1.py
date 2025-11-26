import random
from LinearApprox import LinApprox
import numpy as np
import os

approx = LinApprox(lm='glove')

class ApproxRelation:
    def __init__(self, Vocab, PosSamples):
        self.PosSamples = PosSamples
        self.Vocab = list(set(Vocab + [t for a in self.PosSamples for t in a]))

        self.approximator = approx
        embs = [self.approximator.getTextEmb(t) for t in self.Vocab]
        dist = [np.linalg.norm(e1-e2) for e1 in embs for e2 in embs]
        dist = [d for d in dist if d!=0]
        self.mod = min(dist)/2

    def genNegs(self):
        NegSamples = []
        for s in self.PosSamples:
            random.shuffle(self.Vocab)
            for t in self.Vocab:
                if [s[0],t] not in self.PosSamples+NegSamples:  
                    NegSamples.append([s[0],t])
                    break
        return NegSamples

    def getEoA(self, NegSamples, R=600, verbose=False):
        return [self.approximator.approximate(self.PosSamples, NegSamples, rank, verbose)[0]/self.mod for rank in range(1,R)]
        
    def optRank(self,err):
        for i,e in enumerate(err):
            if e<1.0:   return i
        return None
    
    def Analyse(self,repeat=10):
        Err = np.array([self.getEoA(NegSamples = self.genNegs()) for _ in range(repeat)])

        err = list(np.mean(Err,axis=0))
        i = self.optRank(err)
        if i is None:   i = err.index(min(err))
        return i, err[i], err

rels  = {}

cats = os.listdir('../data/BATS_3.0')
print(cats)
for cat in cats:
    if not os.path.isdir('./BATS_3.0/'+cat): continue
    for f in os.listdir('../data/BATS_3.0/'+cat):
        rels[f.split()[0]]='../data/BATS_3.0/'+cat+'/'+f


for rel in sorted(list(rels.keys())):
# for rel in ['G00']:
    data = open(rels[rel]).readlines()
    PosSamples = list([[w.strip().lower() for w in d.split('\t')[:2]] for d in data if len(d)!=0])
    PosSamples = [[s[0],w] for s in PosSamples for w in s[1].split('/') if len(s[0])>0 and len(w)>0]

    A = ApproxRelation(Vocab=[],PosSamples=PosSamples)
    i,e,err = A.Analyse()
    print(rel, i, e)