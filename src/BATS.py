import json
import time, datetime
import argparse
import numpy as np

from Representation import Representation
from LinearApprox import LinApprox

with open('../data/BATS_4.0.json') as fp: 
    RelPairs = json.load(fp)

with open('../data/BATS_4.0-NegPool.json') as fp: 
    NegPool = json.load(fp)

Rep = Representation(lm='roberta')
batch_size = 5

for rel in RelPairs:
    PosSamples = [[p,s] for p in RelPairs[rel] for s in RelPairs[rel][p]]
    NegSamples = [[p,s] for p,S in NegPool for s in S if [p,s] not in PosSamples][:1000]

    t=time.time()

    Pr = np.stack([e for i in range(0,len(PosSamples),batch_size) for e in Rep.getPairEmbs(PosSamples[i:i+batch_size])]).T
    Nr = np.stack([e for i in range(0,len(NegSamples),batch_size) for e in Rep.getPairEmbs(NegSamples[i:i+batch_size])]).T

    A = LinApprox(lm='none')
    EoA = A.LinApprox(Pr,Nr,rank=None,verbose=False)
    # print("Lin Approx time:",datetime.timedelta(seconds=time.time()-t))
    print(rel, EoA)
    # break

        