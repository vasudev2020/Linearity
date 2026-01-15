import numpy as np
import scipy.optimize as opt

np.random.seed(100)

from transformers import AutoTokenizer, RobertaModel

class LinApprox:
    def __init__(self,lm='glove'):
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

    def glove(self, t):
        for w in t.split(): 
            if w not in self.gloveModel:   self.gloveModel[w]= np.random.rand(300)
        return np.mean([self.gloveModel[w] for w in t.split()],axis=0)

    def roberta(self, t):
        inputs = self.tokenizer(t, return_tensors="pt",truncation=True, return_offsets_mapping=True)
        # print(self.tokenizer.batch_decode(inputs['input_ids']))
        # out = self.robertaModel(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state[0][1:-1].detach().numpy()
        out = self.robertaModel(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state[0].detach().numpy()

        return np.mean(out,axis=0)
    
    # Get representation of a text by using LM
    def getTextEmb(self, t):
        if self.lm=='glove':    return self.glove(t)
        if self.lm=='roberta':  return self.roberta(t)
        raise Exception('Undefined lm: '+self.lm)
    
    # Get representation of a list of pairs of texts by using LM
    def getRepresentation(self, T):
        return np.stack([np.concatenate((self.getTextEmb(t1),self.getTextEmb(t2))) for t1,t2 in T]).T
    
    def getApproxError(self, Tr):
        if self.M is None:   return 1000000, 1000000, 1000000, 1000000
        Pr = self.getRepresentation(Tr)
        norm = np.linalg.norm(np.matmul(self.M,Pr),axis=0)
        return round(np.min(norm),2), round(np.mean(norm),2), round(np.max(norm),2), round(np.median(norm),2)

    def approximate(self, Tr, Tn, rank=None, verbose=False):
        Pr = self.getRepresentation(Tr)
        Nr = self.getRepresentation(Tn)
        dim = 0
        if self.lm=='glove':    dim = 300 
        if self.lm=='roberta':    dim = 768 

        assert Pr.shape[0]==2*dim and Nr.shape[0]==2*dim
        if verbose: print('Number of related text  :',Pr.shape[1])
        if verbose: print('Number of unrelated text:',Nr.shape[1])

        if rank is None:   rank = 2*dim
        if verbose: print('Rank of approx :',rank)


        U,S,V = np.linalg.svd(Pr)
        C = np.square(S)
        C = np.pad(C,(0,Pr.shape[0]-C.shape[0]))[:rank] # Truncate C to size r
        U = U[:,:rank] # Truncate U to shape (U.shpae[0],rank)
        A = np.square(np.matmul(Nr.T, U))
        # A = A[:,:rank]# Truncate A to shape (A.shpae[0],r)
        if verbose: print('LP is starting with',C.shape, 'variables and', A.shape[0], 'constraints')
        res = opt.linprog(C, A_ub=-A, b_ub = -np.ones(A.shape[0]))
        try:    self.M = np.matmul(np.diag(np.sqrt(res.x)), U.T)
        except: return 1000000, None

        # assert round(np.min(np.linalg.norm(np.matmul(M,Nr),axis=0)),2)>=1
        if round(np.min(np.linalg.norm(np.matmul(self.M,Nr),axis=0)),2)<0.99:   print("Infeasible solution")
        # assert M.shape[0]==2*dim and M.shape[1]==2*dim
        assert self.M.shape[0]==rank and self.M.shape[1]==2*dim

        # assert self.getApproxError(Tn,M)[0] >=1
        # if self.getApproxError(Tn)[0] < 1:    print("Infeasible solution")

        if verbose: 
            print('Error of approximation  :',res.fun)
            print("Positive sample error:",self.getApproxError(Tr))
            print("Negative sample error:",self.getApproxError(Tn))

        return res.fun, self.M
        return res.fun/Pr.shape[1], self.M

    # Pr: (2*dim, Np), Nr: (2*dim, Nn)
    def LinApprox(self, Pr, Nr, rank=None, verbose=False):
        assert Pr.shape[0]==Nr.shape[0]
        if verbose: print('Number of related pairs  :',Pr.shape[1])
        if verbose: print('Number of unrelated pairs:',Nr.shape[1])
        if verbose: print('Combined embedding dimension:',Pr.shape[0])

        if rank is None:   rank = Pr.shape[0]
        if rank < 1:    rank = int(Pr.shape[0]*rank)

        U,S,V = np.linalg.svd(Pr)

        C = np.square(S)
        C = np.pad(C,(0,Pr.shape[0]-C.shape[0]))[:rank] # Truncate C to size r
        U = U[:,:rank]                                  # Truncate U to shape (U.shpae[0],rank)
        A = np.square(np.matmul(Nr.T, U))

        if verbose: print('LP is starting with',C.shape, 'variables and', A.shape[0], 'constraints')
        res = opt.linprog(method='highs-ds', c=C, A_ub=-A, b_ub = -np.ones(A.shape[0]))
        # res = opt.linprog(method='highs-ipm', c=C, A_ub=-A, b_ub = -np.ones(A.shape[0]))

        
        try:    self.M = np.matmul(np.diag(np.sqrt(res.x)), U.T)
        except: return 1000000

        if round(np.min(np.linalg.norm(np.matmul(self.M, Nr),axis=0)),2)<0.9:   print("Infeasible solution")
        assert self.M.shape[0]==rank and self.M.shape[1]==Pr.shape[0]

        if verbose: print('Error of approximation  :',res.fun)

        return res.fun/Pr.shape[1]

    def getOutliers(self,Pr,threshold):
        if self.M is None:   return np.array([True]*Pr.shape[1])
        norm = np.linalg.norm(np.matmul(self.M,Pr),axis=0)

        return norm>threshold,norm
    
    def saveModel(self,filename):
        np.save(filename,self.M)

    def loadModel(self,filename):
        self.M = np.load(filename)
    
    # def evaluate(self,TrPosSamples, TrNegSamples, TePosSamples, rank=None, verbose=False):
    #     self.approximate(TrPosSamples, TrNegSamples, rank, verbose)

    #     _,TrMean,_,_ = self.getApproxError(TrPosSamples)

    #     _,TeMean,_,_ = self.getApproxError(TePosSamples)
    #     return TrMean,TeMean
    
if __name__ == '__main__':

    linapprox=LinApprox(lm='roberta')
    Tr = [('boy','girl'),('man','woman'),('king','queen'),('one king','one queen')]
    Tn = [('card','box'),('tree','plant'),('boy','man'),('the car','the bus'),('the boy','the man')]
    fn,m = linapprox.approximate(Tr,Tn,verbose=True)    
    fn,m = linapprox.approximate(Tr,Tn,rank=200, verbose=True)    

    # fn,m = linapprox.approximate(Tr,Tr)

