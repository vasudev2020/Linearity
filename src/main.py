from LinearApprox import LinApprox
import os, json, random
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut


from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

from statistics import mean, stdev

approximator = LinApprox(lm='glove')

class Relations:
    def __init__(self, approximator, sample_file, sample_pool=[], neg_ratio=1, rank=None):
        self.linapprox=approximator
        self.rank=rank
        self.sample_pool=sample_pool
        self.neg_ratio = neg_ratio
        self.loadSamples(sample_file)

    def setRank(self,rank):
        self.rank=rank
    
    def loadSamples(self, sample_file):
        raise Exception('Not implemented')

    # def getApproxError(self, pos_size, verbose=False):
    #     TrainPosSamples,TestPosSamples = train_test_split(self.PosSamples, train_size=pos_size)  # Select k samples for training and remaining for testing                          
    #     # TrainNegSamples = random.choices(self.NegSamples, k=neg_size)
    #     return self.linapprox.evaluate(TrainPosSamples, self.NegSamples, TestPosSamples, self.rank, verbose)
    
    # def predict(self, pos_size, Target, verbose=False):
    #     TrainPosSamples,TestPosSamples = train_test_split(self.PosSamples, train_size=pos_size)  # Select k samples for training and remaining for testing                          
    #     self.linapprox.approximate(TrainPosSamples, self.NegSamples, self.rank, verbose)
    #     hit = 0
    #     for s in TestPosSamples:
    #         E = [self.linapprox.getApproxError([[s[0],v]])[1] for v in Target]
    #         optim =min(enumerate(E), key=lambda x: x[1])[0]
    #         hit += int(Target[optim]==s[1])
    #     return hit *100.0/len(TestPosSamples)
    
    def evaluate(self, train_index, test_index, Target, verbose=False):
        TrainPosSamples = [self.PosSamples[i] for i in train_index]
        TestPosSamples = [self.PosSamples[i] for i in test_index]
        self.linapprox.approximate(TrainPosSamples, self.NegSamples, self.rank, verbose)
        TrErr = self.linapprox.getApproxError(TrainPosSamples)[1]
        # TeErr = [self.getApproxError(TestPosSamples[i:i+1])[1] for i in range(len(TestPosSamples))]
        hit = 0
        TeErr = 0
        for s in TestPosSamples:
            E = [self.linapprox.getApproxError([[s[0],v]])[1] for v in Target]
            optim =min(enumerate(E), key=lambda x: x[1])[0]
            hit += int(Target[optim]==s[1])
            # print(s[0], Target[optim],s[1])
            TeErr += E[Target.index(s[1])]
        Acc = hit *100.0/len(TestPosSamples)
        TeErr /= len(TestPosSamples)
        return TrErr, TeErr, Acc

class W2WRelations(Relations):
    def __init__(self, approximator, sample_file, sample_pool=[], neg_ratio=1, rank=None):
        super().__init__(approximator, sample_file, sample_pool, neg_ratio, rank)

    def loadSamples(self, sample_file):
        data = open(sample_file).readlines()
        self.PosSamples = list([[w.strip().lower() for w in d.split('\t')[:2]] for d in data if len(d)!=0])
        self.PosSamples = [[s[0],w] for s in self.PosSamples for w in s[1].split('/') if len(s[0])>0 and len(w)>0]
        self.Vocab = list(set([t for a in self.PosSamples for t in a]))
        self.Vocab += self.sample_pool
        self.NegSamples = [[a,b] for a in self.Vocab for b in self.Vocab if [a,b] not in self.PosSamples]

        neg_size = int(len(self.PosSamples)*self.neg_ratio)
        self.NegSamples = random.choices(self.NegSamples, k=neg_size)
    
    def LeaveOneOutApprox(self):
        loo = LeaveOneOut()
        E = [self.evaluate(train_index, test_index, Target=self.Vocab) for train_index, test_index in loo.split(self.PosSamples)]
        return round(mean([e[0] for e in E]),2), round(mean([e[1] for e in E]),2), round(mean([e[2] for e in E]),2)
    
        # E=[]
        # for i, (train_index, test_index) in enumerate(loo.split(self.PosSamples)):   
            # TrainPosSamples = [self.PosSamples[i] for i in train_index]
            # TestPosSamples = [self.PosSamples[i] for i in test_index]
            # E.append(self.linapprox.evaluate(TrainPosSamples, self.NegSamples, TestPosSamples, self.rank))
        # return round(mean([e[0] for e in E]),2), round(mean([e[1] for e in E]),2)

    def SingleSampleApprox(self):
        loo = LeaveOneOut()
        # neg_size = len(self.PosSamples)
        # TrainNegSamples = random.choices(self.NegSamples, k=neg_size)
        E=[]
        for i, (test_index, train_index) in enumerate(loo.split(self.PosSamples)):   
            TrainPosSamples = [self.PosSamples[i] for i in train_index]
            TestPosSamples = [self.PosSamples[i] for i in test_index]
            E.append(self.linapprox.evaluate(TrainPosSamples, self.NegSamples, TestPosSamples, self.rank))
        return mean([e[0] for e in E]),mean([e[1] for e in E])
    
    def FullApprox(self,fullrank=False):
        if fullrank: return self.linapprox.evaluate(self.PosSamples, self.NegSamples, self.PosSamples, None)[0]
        return self.linapprox.evaluate(self.PosSamples, self.NegSamples, self.PosSamples, self.rank)[0]
    
    # def iterativeApprox(self, step, repeat=5):
    #     # neg_size = len(self.PosSamples)
    #     # for s in self.PosSamples:   print(s)
    #     # print(len(self.PosSamples), self.PosSamples[:2])
    #     TrE, TeE = {}, {}
    #     start = step
    #     stop = min(600, len(self.PosSamples))
    #     for i in range(start,stop,step):
    #         # print(i)
    #         try:
    #             E = [self.getApproxError(pos_size=i, verbose=False) for _ in range(repeat)]
    #             TrE[i]=mean([e[0] for e in E])
    #             TeE[i]=mean([e[1] for e in E])
    #         except: 
    #             TrE[i]=1000000
    #             TeE[i]=1000000
    #     return TrE,TeE
 
def readWiki(size):
    wiki_path='../Data/Wiki'
    dirs = os.listdir(wiki_path)
    dataset = []
    for dir in dirs:
        files = os.listdir(wiki_path+'/'+dir)
        for f in files:
            for line in open(wiki_path+'/'+dir+'/'+f):
                d = json.loads(line)
                samples = d['text'].split('\n')
                samples = [ss.strip() for s in samples for ss in sent_tokenize(s) if len(ss.strip())>0]

                dataset.extend(samples)
                if len(dataset)>=size:
                    dataset = list(set(dataset))
                    if len(dataset)>=size:  return dataset[:size]
    
    return list(set(dataset))

def BATSSizeSearch(neg_ratio=1):
    opt_rank = {'D01': 48,'D02': 48,'D03': 41,'D03': 41,'D04': 90,'D05': 42,'D06': 43,'D07': 38,'D08': 47,'D09': 58,'D10': 49,'E01': 41,'E02': 56,'E03': 44,'E04': 52,'E05': 67,'E06': 61,'E07': 61,'E08': 64,'E09': 56,'E10': 55,'G00': 94,'I01': 29,'I02': 18,'I03': 44,'I04': 29,'I05': 46,'I06': 45,'I07': 45,'I08': 47,'I09': 36,'I10': 46}
    # TODO: Use GloVe vocab instead of Wiki?
    vocab=[]
    rels  = {}
    cats = os.listdir('./BATS_3.0')
    for cat in cats:
        if not os.path.isdir('./BATS_3.0/'+cat): continue
        for f in os.listdir('./BATS_3.0/'+cat):
            rels[f.split()[0]]='./BATS_3.0/'+cat+'/'+f

    for rel in sorted(list(rels.keys()))[:-10]:
        w2wrel = W2WRelations(approximator=approximator, sample_file=rels[rel], sample_pool=vocab, neg_ratio=neg_ratio, rank=opt_rank[rel])
        TrE,TeE = w2wrel.iterativeApprox(step=1)
        s = sorted(list(TrE.keys()))
        print(rel, 'TrE,', ', '.join([str(round(TrE[i],2)) for i in s]))
        print(rel, 'TeE,', ', '.join([str(round(TeE[i],2)) for i in s]))


def BATSRankSearch(neg_ratio=1):
    sents = readWiki(1000)
    # TODO: Use GloVe vocab instead of Wiki?
    # vocab = list(set([w for s in sents for w in s.split()]))[:100]
    vocab=[]
    rels  = {}
    cats = os.listdir('./BATS_3.0')
    for cat in cats:
        if not os.path.isdir('./BATS_3.0/'+cat): continue
        for f in os.listdir('./BATS_3.0/'+cat):
            rels[f.split()[0]]='./BATS_3.0/'+cat+'/'+f

    print(', '.join(['rel', 'min_TrE', 'opt_rank', 'opt_TrE', 'opt_TeE', 'opt_TeAcc']))
    # print(', '.join(['rel','opt_rank', 'opt_TeE', 'opt_TrE', 'min_TrE', 'full_TrE']))

    # for rel in sorted(list(rels.keys()))[:-10]:
    for rel in sorted(list(rels.keys()))[:1]:
        E = []
        for rank in range(1,100,1):
            w2wrel=W2WRelations(approximator=approximator, sample_file=rels[rel], sample_pool=vocab, neg_ratio=neg_ratio, rank=rank)
            E.append(w2wrel.LeaveOneOutApprox())
            if E[-1][0]==0: break
        optim =min(enumerate(E), key=lambda x: x[1][1])[0]
        # fullapproxerr = w2wrel.FullApprox(fullrank=True)
        # print(rel, 'TrE,', optim, ', ', E[optim][0],', ', ', '.join([str(e[0]) for e in E]))
        # print(rel, 'TeE,', optim, ', ', E[optim][1],', ', ', '.join([str(e[1]) for e in E]))
        # print(rel+',', ', '.join([str(round(v,2)) for v in [optim, E[optim][1], E[optim][0], min([e[0] for e in E]), fullapproxerr]]))
        print(rel+',', ', '.join([str(round(v,2)) for v in [min([e[0] for e in E]), optim, E[optim][0], E[optim][1], E[optim][2]]]))


    # for cat in cats:
    #     if not os.path.isdir('./BATS_3.0/'+cat): continue
    #     for f in os.listdir('./BATS_3.0/'+cat):
    #         # if not f.split()[0].startswith('L02'):  continue
    #         w2wrel=W2WRelations(approximator=approximator, sample_file='./BATS_3.0/'+cat+'/'+f, sample_pool=vocab, rank=2)
    #         TrE,TeE = w2wrel.LeaveOneOutApprox()
    #         print(f.split()[0], round(TrE,2), round(TeE,2))
    #         # TrE,TeE = w2wrel.iterativeApprox(step=5)
    #         # s = sorted(list(TrE.keys()))
    #         # print(f.split('.')[0]+', TrE, ', ', '.join([str(round(TrE[i],2)) for i in s]))
    #         # print(f.split('.')[0]+', TeE, ',', '.join([str(round(TeE[i],2)) for i in s]))
    #         # print(','.join([f.split()[0], str(round(TrE[s[0]],2)), str(round(TrE[s[-1]],2)), str(round(TeE[s[0]],2)), str(round(TeE[s[-1]],2))]))


def FormatGenderFile():
    data = open('OppositeGender.csv').readlines()[1:]
    D =[[w.strip().lower() for w in d.split(',')[:2]] for d in data if len(d)!=0]
    f = open('./BATS_3.0/Extension/gender.txt','w')
    for d in D: f.write(d[0]+'\t'+d[1]+'\n')
    f.close()


BATSRankSearch(10)
# BATSSizeSearch(2)

#######################################

'''  


def genNegatives(self, Tr, size, SampleA, SampleB=None):
    Tn = []
    if SampleB is None: SampleB = SampleA
    size = min(size, len(SampleA)*len(SampleB))

    while True:
        i = SampleA[random.randint(0,len(SampleA)-1)]
        j = SampleB[random.randint(0,len(SampleB)-1)]
        if [i,j] in Tr: continue
        Tn.append([i,j])
        if len(Tn)==size:   return Tn

def GenderRelApprox(self):
    # linapprox=LinApprox(lm=lm)
    sents = self.readWiki(100000)
    VocabOut = list(set([w for s in sents for w in s.split()]))
    data = open('OppositeGender.csv').readlines()[1:]
    Tr = [[w.strip().lower() for w in d.split(',')[:2]] for d in data if len(d)!=0]
    # print(Tr)
    sampleV = set([])
    for a,b in Tr:
        sampleV.add(a)
        sampleV.add(b)
    VocabIn = list(sampleV)
    Vocab = VocabIn + VocabOut[:len(VocabIn)]
    Tn = self.genNegatives(Tr,10000,Vocab)
    # TnIn = genNegatives(Tr,5000,VocabIn)
    # TnOut = genNegatives(Tr,5000,VocabOut)

    fn,m = self.linapprox.approximate(Tr,Tn,True)
    # linapprox.getApproxError(Tn,m)

def IdentityRelApprox(self):
    sents = self.readWiki(100000)
    vocab = list(set([w for s in sents for w in s.split()]))
    Tr = [(w,w) for w in vocab][:10000]
    Tn = self.genNegatives(Tr,100000,vocab)
    fn,m = self.linapprox.approximate(Tr,Tn)

def BATS(self,rels=['gender']):
    sents = self.readWiki(100000)
    vocab = list(set([w for s in sents for w in s.split()]))

    Data = {}
    cats = os.listdir('./BATS_3.0')
    for cat in cats:
        if not os.path.isdir('./BATS_3.0/'+cat): continue

        for f in os.listdir('./BATS_3.0/'+cat):
            data = open('./BATS_3.0/'+cat+'/'+f).readlines()
            Data[f.split('.')[0]] = [[w.strip().lower() for w in d.split('\t')[:2]] for d in data if len(d)!=0]

    data = open('OppositeGender.csv').readlines()[1:]
    Data['gender']=[[w.strip().lower() for w in d.split(',')[:2]] for d in data if len(d)!=0]

    if rels==['all']: rels=list(Data.keys())
    for rel in rels:
        PosSamples = Data[rel]
        TrainNegSamples = self.genNegatives(PosSamples,10000,vocab)
        print(rel)
        for k in range(1,len(PosSamples),10):
            Error = []
            for _ in range(3):
                TrainPosSamples,TestPosSamples = train_test_split(PosSamples,train_size=k)  # Select k samples for training and remaining for testing
                Error.append(self.linapprox.evaluate(TrainPosSamples,TrainNegSamples,TestPosSamples))
            print(k, mean([e[0] for e in Error]), mean([e[1] for e in Error]))

            # TrainError, MinTestError,MeanTestError,MaxTestError,MedianTestError = [],[],[],[],[]
            # for _ in range(3):
            #     TrainPosSamples,TestPosSamples = train_test_split(PosSamples,train_size=k)  # Select k samples for training and remaining for testing
            #     trainErr, M = self.linapprox.approximate(TrainPosSamples,TrainNegSamples)
            #     TrainError.append(trainErr)
            #     Min,Mean,Max,Median = self.linapprox.getApproxError(TestPosSamples,M)
            #     # TODO: Calculate sample-wise correctness 
            #     MeanTestError.append(Mean)
            #     MinTestError.append(Min)
            #     MaxTestError.append(Max)
            #     MedianTestError.append(Median)
            # print(k, round(mean(TrainError),2), round(mean(MinTestError),2), round(mean(MeanTestError),2), round(mean(MaxTestError),2),round(mean(MedianTestError),2))

def readProbingData(self,task,size, neg_size):
    datafilename = './ProbingTasks/'+task+'.txt'
    if not os.path.exists(datafilename):
        raise Exception(f"Invalid dataset: {task}")

    data = open(datafilename).readlines()

    Data = {}

    for line in data:
        cat,label,text = line.split('\t')
        if cat not in Data:   Data[cat] = {}
        if label not in Data[cat]:    Data[cat][label] = []
        Data[cat][label].append(text)

    for cat in Data:
        for label in Data[cat]:
            Data[cat][label] = Data[cat][label][:size]

    PosSample = {}
    for cat in Data:
        PosSample[cat] = []
        for label in Data[cat]:
            # # Samples based on Clique network
            # for s1 in Data[cat][label]:
            #     for s2 in Data[cat][label]:
            #         PosSample[cat].append([s1,s2])
            # # Select based on star network. We can use this center sample for classification
            for s in Data[cat][label]:
                PosSample[cat].append([Data[cat][label][0],s])
                PosSample[cat].append([s, Data[cat][label][0]])
        # PosSample[cat] += [[s,random.choice(list(set(Data[cat][label])-set([s])))] for label in Data[cat] for s in Data[cat][label]]
        # TODO: Can create more positive samples

    TrNegSample = []
    for label1 in Data['tr']:
        for label2 in Data['tr']:
            if label1==label2:  continue
            TrNegSample+=self.genNegatives([],neg_size,Data['tr'][label1],Data['tr'][label2])

    return PosSample, TrNegSample

        
def ProbingLinApprox(self, size):

    for task in ['subj_number', 'obj_number', 'bigram_shift', 'coordination_inversion', 'odd_man_out', 'past_present']:#, 'tree_depth', 'word_content', 'sentence_length', 'top_constituents']:
        # TrPosSamples,TePosSamples,VaPosSamples,TrNegSamples = readProbingData(task,size,size)
        # PosSamples,TrNegSamples = readProbingData(task,size,size*size)
        # PosSamples,TrNegSamples = readProbingData(task,size,size)
        PosSamples,TrNegSamples = self.readProbingData(task,size,1)

        # print('Samples loaded')
        TrMean,TeMean = self.linapprox.evaluate(PosSamples['tr'],TrNegSamples,PosSamples['te'],self.rank,self.verbose)
        trainErr, M = self.linapprox.approximate(PosSamples['tr'],TrNegSamples,verbose=True)
        trMin,trMean,trMax,trMedian = self.linapprox.getApproxError(PosSamples['tr'],M)

        Min, Mean, Max, Median = self.linapprox.getApproxError(PosSamples['te'],M)
        print(task, trMean, Mean, '(', Min, Median, Max,')')

        break

def LinearProbe(lm):
    linapprox=LinApprox(lm=lm)
    for task in ['subj_number', 'obj_number', 'bigram_shift', 'coordination_inversion', 'odd_man_out', 'past_present']:#, 'tree_depth', 'word_content', 'sentence_length', 'top_constituents']:

        datafilename = './ProbingTasks/'+task+'.txt'
        if not os.path.exists(datafilename):
            raise Exception(f"Invalid dataset: {task}")

        data = open(datafilename).readlines()

        trX = [linapprox.getTextEmb(line.split('\t')[2]) for line in data if line.split('\t')[0]=='tr']
        trY = [line.split('\t')[1] for line in data if line.split('\t')[0]=='tr']

        teX = [linapprox.getTextEmb(line.split('\t')[2]) for line in data if line.split('\t')[0]=='te']
        teY = [line.split('\t')[1] for line in data if line.split('\t')[0]=='te']

        L = list(set(trY+teY))
        trY = [L.index(l) for l in trY]
        teY = [L.index(l) for l in teY]

        # model = LinearRegression().fit(trX,trY)
        trX = trX[:1000]
        trY = trY[:1000]
        model = MLPClassifier(hidden_layer_sizes=()).fit(trX,trY)
        print(task,model.score(trX,trY))

        # print(model.score(teX,teY))

'''
# sents = readWiki(100000)
# vocab = list(set([w for s in sents for w in s.split()]))
# print(len(vocab))
# IdentityRelApprox(lm='roberta')
# GenderRelApprox(lm='glove')
# BATS(lm='glove')
# ProbingLinApprox(size=1000, lm = 'glove')
# LinearProbe(lm = 'glove')