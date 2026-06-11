import json
import time, datetime
import argparse
import numpy as np

from Representation import Representation
from LinearApprox import LinApprox
res_path = '../results/ModernBERT/BATS_4.0/'
# with open('../data/BATS_3.0.json') as fp: 
with open('../data/BATS_4.0.json') as fp: 
    RelPairs = json.load(fp)

# with open('../data/BATS_4.0-NegPool.json') as fp: 
#     NegPool = json.load(fp)

# Rep = Representation(lm='roberta')
# Rep = Representation(lm='glove')
Rep = Representation(lm='mbert')

batch_size = 5

for rel in RelPairs:
    # PosSamples = [[p,s] for p in RelPairs[rel] for s in RelPairs[rel][p]]
    # NegSamples = [[p,s] for p,S in NegPool for s in S if [p,s] not in PosSamples][:1000]

    PosSamples = [(p,s) for p in RelPairs[rel] for s in RelPairs[rel][p]]
    Pool = [(p,s) for p in RelPairs[rel] for p1 in RelPairs[rel] for s in RelPairs[rel][p1]] 
    NegSamples = list(set(Pool)-set(PosSamples))
    # print(rel,len(PosSamples),len(NegSamples))

    # t=time.time()
    Vocab = [ee for e in PosSamples for ee in e] + [ee for e in NegSamples for ee in e]
    Vocab = list(set(Vocab))
    Embs = [np.mean(e,axis=0) for i in range(0,len(Vocab),batch_size) for e in Rep.getEmbs(Vocab[i:i+batch_size])]

    # Embs = np.concatenate([np.mean(e,axis=0) Rep.getEmbs([Vocab[i:i+batch_size]]) for i in range(0,len(Vocab),batch_size)])
    assert len(Vocab)==len(Embs)
    EmbLookup = {t:e for t,e in zip(Vocab,Embs)}
    Pr = np.stack([np.concatenate([EmbLookup[p],EmbLookup[s]])for p,s in PosSamples]).T
    Nr = np.stack([np.concatenate([EmbLookup[p],EmbLookup[s]])for p,s in NegSamples]).T

    # Pr = np.stack([e for i in range(0,len(PosSamples),batch_size) for e in Rep.getPairEmbs(PosSamples[i:i+batch_size])]).T
    # Nr = np.stack([e for i in range(0,len(NegSamples),batch_size) for e in Rep.getPairEmbs(NegSamples[i:i+batch_size])]).T
    # print("Representation generation time:",datetime.timedelta(seconds=time.time()-t))
    # t=time.time()

    A = LinApprox(lm='none')
    # EoA = A.LinApprox(Pr,Nr,rank=None,verbose=False)
    EoA = A.LinApprox(Pr,Nr,rank=0.5,verbose=False)
    A.saveModel(res_path+'Models/'+rel+'.npy')
    # print("Lin Approx time:",datetime.timedelta(seconds=time.time()-t))
    # outliers,norms = A.getOutliers(Pr,0.5)
    outliers,norms = A.getOutliers(Pr,-0.1)

    assert len(outliers)==len(PosSamples)
    OutlierSamples = [(s,n) for o,s,n in zip(outliers,PosSamples,norms) if o]
    # print('Outliers:',len(OutlierSamples),len(PosSamples))
    print(rel, len(PosSamples),len(NegSamples),EoA,len(OutlierSamples))

    if len(OutlierSamples)>0:
        # with open('../results/BATS_4.0_Outliers/'+rel+'.tsv','w') as efp:
        with open(res_path+'Samples/'+rel+'.tsv','w') as efp:
            for s,n in OutlierSamples:  efp.write(s[0]+'\t'+s[1]+'\t'+str(n)+'\n')
    # break

        