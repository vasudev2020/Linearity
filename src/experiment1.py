
import random
from LinearApprox import LinApprox
import numpy as np
import os
from itertools import cycle
import argparse, time, datetime

np.random.seed(100)
random.seed(100)

class ApproxRelation:
    def __init__(self, approx, Vocab, PosSamples):
        self.PosSamples = PosSamples
        self.Vocab = sorted(list(set(Vocab + [t for a in self.PosSamples for t in a])))
        self.approximator = approx
        self.NegSamples = self.genNegs(size=100)

    def genNegs(self,size=None):
        pre = [s[0] for s in self.PosSamples]
        if size is None:   size = len(pre) 
        maxsize = len(set(pre))*len(set(self.Vocab))-len(self.PosSamples)
        size = min(size,maxsize)

        cycle_iter = cycle(pre)
        NegSamples = []
        p = next(cycle_iter)
        while len(NegSamples)<size:
            t = random.sample(self.Vocab,1)[0]
            if [p,t] not in self.PosSamples+NegSamples:
                NegSamples.append([p,t])
                p = next(cycle_iter)

        # NegSamples = []
        # for s in self.PosSamples:
        #     while True:
        #         t = random.sample(self.Vocab,1)[0]
        #         if [s[0],t] not in self.PosSamples+NegSamples:   break
        #     NegSamples.append([s[0],t])

        # NegSamples = []
        # for s in self.PosSamples:
        #     random.shuffle(self.Vocab)
        #     for t in self.Vocab:
        #         if [s[0],t] not in self.PosSamples+NegSamples:  
        #             NegSamples.append([s[0],t])
        #             break
        return NegSamples

    def getFullRankEoA_(self,size=None):
        if size is None: size = len(self.PosSamples)
        NegSamples = self.genNegs()
        # print(self.PosSamples[0:2])
        # print(NegSamples[0:2])
        return self.approximator.approximate(Tr=self.PosSamples[:size], Tn=NegSamples[:size],verbose=False)[0]

    def getFullRankEoA(self,size=None):
        if size is None: size = len(self.PosSamples)
        return self.approximator.approximate(Tr=self.PosSamples[:size], Tn=self.NegSamples,verbose=False)[0]
    ###

    def getEoA(self, NegSamples, R=600, verbose=False):
        return [self.approximator.approximate(self.PosSamples, NegSamples, rank, verbose)[0] for rank in range(1,R)]

        
    def optRank(self,err):
        for i,e in enumerate(err):
            if e<1.0:   return i
        return None
    
    def Analyse(self,repeat=1):
        Err = np.array([self.getEoA(NegSamples = self.genNegs()) for _ in range(repeat)])

        err = list(np.mean(Err,axis=0))
        i = self.optRank(err)
        if i is None:   i = err.index(min(err))
        return i, err[i], err

def main(lm='glove',size=100):
    rels  = {}
    path = '../data/BATS_3.0/'
    cats = os.listdir(path)
    # print(cats)
    for cat in cats:
        if not os.path.isdir(path+cat): continue
        for f in os.listdir(path+cat):
            rels[f.split()[0]]=path+cat+'/'+f

    approx = LinApprox(lm)

    for rel in sorted(list(rels.keys())): #    for rel in ['G00']:
        data = open(rels[rel]).readlines()
        PosSamples = list([[w.strip().lower() for w in d.split('\t')[:2]] for d in data if len(d)!=0])
        PosSamples = [[s[0],w] for s in PosSamples for w in s[1].split('/') if len(s[0])>0 and len(w)>0]

        A = ApproxRelation(approx=approx, Vocab=[],PosSamples=PosSamples)
    
        print(rel, len(PosSamples), ' '.join([str(round(A.getFullRankEoA(s),4)) for s in range(50,len(PosSamples)+1 if size is None else min(size,len(PosSamples))+1,10)]))
        # i,e,err = A.Analyse()
        # print(rel, i, e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    '''Hyperparamters related to Dataset'''
    parser.add_argument('--lm', type=str, default='glove', help='representation model: glove/roberta/mbert')
    parser.add_argument('--size', type=int, default=None, help='Number of Wiki sentences')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')

    args = parser.parse_args()
    t0=time.time()

    main(lm=args.lm, size=args.size)

    print('Total Execution Time:',datetime.timedelta(seconds=time.time()-t0))

