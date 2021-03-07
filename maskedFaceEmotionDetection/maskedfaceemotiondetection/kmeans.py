from common import *

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import torch

class InitAlgorithm:
    MAXSEPERATION = 0
    MINGROUPING = 1
    KPLUS = 2
    RANDOM = 3
class KMeansClassifier:
    def __init__(self,trainingInputs,trainingTargets,clusters,initAlgorithim=None):
        self.__trainingInputs  = trainingInputs
        self.__trainingTargets = trainingTargets
        self.__clusters = clusters
        self.__initAlgorithim = initAlgorithim
        #distances = self.allPairDistances(self.trainingInputs,self.trainingInputs)
        self.train()

    @property 
    def trainingInputs(self):
        return self.__trainingInputs

    @property 
    def trainingTargets(self):
        return self.__trainingTargets

    @property 
    def clusters(self):
        return self.__clusters

    @property 
    def initAlgorithim(self):
        return self.__initAlgorithim

    @property 
    def model(self):
        return self.__model

    def allPairDistances(self,A,B):
        tensorA = torch.tensor(np.array(A))
        tensorB = torch.tensor(np.array(A))

        AA = torch.sum(torch.pow(tensorA,2),1,keepdim=True).expand(tensorA.shape[0],tensorB.shape[0])
        BB = torch.sum(torch.pow(tensorB,2),1,keepdim=True).expand(tensorB.shape[0],tensorA.shape[0]).t()
        AB = torch.mm(tensorA,tensorB.t())

        return torch.sqrt(AA - 2*AB + BB)

    def findInitialCentroids(self):
        if self.initAlgorithim is None or self.initAlgorithim == InitAlgorithm.RANDOM :
            return ('random',10)
        if self.initAlgorithim == InitAlgorithm.KPLUS:
            return ('k-means++',10)
        if self.initAlgorithim == InitAlgorithm.MAXSEPERATION:
            return ('k-means++',10)
        if self.initAlgorithim == InitAlgorithm.MAXSEPERATION:
            return ('k-means++',10)           

    def train(self):
        initParams = self.findInitialCentroids()
        self.__model = KMeans(n_clusters=self.clusters,init=initParams[0],n_init=initParams[1],max_iter=600,algorithm='auto')
        self.model.fit(self.trainingInputs)
        labels = self.model.labels_
        inertia = self.model.inertia_
        trueLabelsMap = np.zeros(self.clusters,dtype=int)
        t = self.trainingTargets

        for k in range(self.clusters):
            kTrueLabels = [t.iloc[i] for i in range(len(t)) if labels[i] == k]
            kLabel = statistics.mode(kTrueLabels)
            trueLabelsMap[k] = kLabel
            kAccuracy = len([i for i in kTrueLabels if i == kLabel])/len(kTrueLabels)
            #print("k = %d label = %d accuracy = %.3f intertia = %f" %(k,kLabel,kAccuracy,inertia))
        print(set(trueLabelsMap))
    
    
               

if __name__ == '__main__':
    fpath = projdir+"/datasets/facesWithoutMasks.csv"
    data = csvToDataframe(fpath,"emotion")

    targets = data.emotion
    inputs_raw = data.drop('emotion',axis=1)
    inputs = scaledDataFrame(inputs_raw)
    inputs_train,inputs_test,targets_train,targets_test=train_test_split(inputs,targets,test_size=0.2)

    #print(inputs_train.shape)
    kmeans = KMeansClassifier(inputs_train,targets_train,350)
