import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
    model = None
    states={}
    for idx, item in enumerate(tags):
        states[item] = idx
    
    S=len(tags)
    pi=np.zeros((S))
    obs_dict={}
    
    A=np.zeros((S,S))
    l=0
    B=[]
    for j in range (len(train_data)):
        xt=train_data[j].tags
        xw=train_data[j].words
        ind=states[train_data[j].tags[0]]
        pi[ind]+=1
        tagencod=np.zeros(len(xt),dtype='int')
        for i in range(len(xt)):
            tagencod[i]=tags.index(xt[i])
        for (m,n) in zip(tagencod,tagencod[1:]):
            A[m][n] += 1 
        for k in range(len(xt)):
            if xw[k] in obs_dict.keys():
                ind=obs_dict[xw[k]]
                B[ind][states[xt[k]]]+=1
            else:
                obs_dict[xw[k]]=l
                temp=np.zeros(S)
                temp[states[xt[k]]]=1
                B.append(temp)
                l+=1
    B=np.array(B).T
    bsum=np.sum(B,axis=1).reshape((-1,1))
    B=np.divide(B,bsum, where=bsum!=0)
    asum=np.sum(A,axis=1).reshape((-1,1))
    A=np.divide(A,asum, where=asum!=0)
    k=0
    pi=pi/sum(pi)
    model = HMM(pi, A, B, obs_dict, states)
	
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    tagging = []
    S,L=model.B.shape
    for i in range(len(test_data)):
        Osequence=test_data[i].words
        for word in Osequence:
            if word not in model.obs_dict:
                temp=0.000001*np.ones((S,1))
                model.B=np.hstack((model.B,temp))
                model.obs_dict[word]=len(model.obs_dict)
        tagging.append(model.viterbi(Osequence))
    
    return tagging