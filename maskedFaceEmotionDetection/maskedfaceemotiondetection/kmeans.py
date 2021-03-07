from common import *

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import torch
import sys

PLUSINF = sys.float_info.max
MINUSINF = sys.float_info.min
np.set_printoptions(precision=3)

class InitAlgorithm:
    RANDOM = 0
    KPLUS = 1
    MAXSEPERATION = 2
    MINGROUPING = 3

class KMeansClassifier:
    def __init__(self,trainingInputs,trainingTargets,clusters,initAlgorithim=None):
        self.__trainingInputs  = trainingInputs
        self.__trainingTargets = trainingTargets
        self.__clusters = clusters
        self.__initAlgorithim = initAlgorithim
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

    @property 
    def trueLabelsMap(self):
        return self.__trueLabelsMap

    def allPairDistances(self,A,B):
        tensorA = torch.tensor(np.array(A))
        tensorB = torch.tensor(np.array(A))

        AA = torch.sum(torch.pow(tensorA,2),1,keepdim=True).expand(tensorA.shape[0],tensorB.shape[0])
        BB = torch.sum(torch.pow(tensorB,2),1,keepdim=True).expand(tensorB.shape[0],tensorA.shape[0]).t()
        AB = torch.mm(tensorA,tensorB.t())

        return torch.sqrt(AA - 2*AB + BB)

    def maxSeperation(self):
        k = self.clusters
        centroids = np.empty(k,dtype=int)
        centroids.fill(-1)
        trainingInputs = self.trainingInputs
        distances = self.allPairDistances(trainingInputs,trainingInputs)
        distances = np.array(distances)
        distances = np.nan_to_num(distances, copy=True, nan=MINUSINF, posinf=None, neginf=None)
        maxNodes = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
        centroids[0] = maxNodes[0]
        centroids[1] = maxNodes[1]
        pending = np.where(centroids<0)[0]
        while len(pending) > 0:
            assigned = centroids[np.where(centroids>=0)]
            for i in [(x,y) for x in assigned for y in assigned if x != y]:
                distances[i] = MINUSINF
            distancesFromAssigned = distances[assigned,:]
            avgDistancesFromAssigned = distancesFromAssigned.sum(axis=0)
            maxNode = np.argmax(avgDistancesFromAssigned)
            centroids[pending[0]] = maxNode
            pending = np.where(centroids<0)[0]
        return np.array(trainingInputs)[centroids,:]

    def findInitialCentroids(self):
        if self.initAlgorithim is None or self.initAlgorithim == InitAlgorithm.RANDOM :
            return ('random',10)
        if self.initAlgorithim == InitAlgorithm.KPLUS:
            return ('k-means++',10)
        if self.initAlgorithim == InitAlgorithm.MAXSEPERATION:
            return (self.maxSeperation(),1)
        if self.initAlgorithim == InitAlgorithm.MINGROUPING:
            return ('k-means++',10)           

    def train(self):
        k = self.clusters
        initParams = self.findInitialCentroids()
        self.__model = KMeans(n_clusters=k,init=initParams[0],n_init=initParams[1],max_iter=600,algorithm='auto')
        self.model.fit(self.trainingInputs)
        labels = self.model.labels_
        inertia = self.model.inertia_
        trueLabelsMap = np.empty(k,dtype=int)
        trueLabelsMap.fill(-1)
        accuracy = np.zeros(k)
        t = self.trainingTargets

        for i in range(k):
            iTrueLabels = [t.iloc[j] for j in range(len(t)) if labels[j] == i]
            iLabel = statistics.mode(iTrueLabels)
            trueLabelsMap[i] = iLabel
            accuracy[i] = len([j for j in iTrueLabels if j == iLabel])/len(iTrueLabels)
        
        self.__trueLabelsMap = trueLabelsMap
    
    def classifyOne(self,inData):
        numpyInData = np.array(inData.astype(float))
        numpyInData = numpyInData.reshape(-1,len(numpyInData))
        clusterId = self.model.predict(numpyInData)
        label = self.trueLabelsMap[clusterId]
        return label
    
    def accuracy(self,inputs,targets):
        hits = 0
        inputs = np.array(inputs)
        targets = np.array(targets)
        for i in range(inputs.shape[0]):
            label = self.classifyOne(inputs[i])
            if label == targets[i]:
                hits += 1
        accuracy = hits/inputs.shape[0]
        return accuracy
    
               

if __name__ == '__main__':
    fpath = projdir+"/datasets/facesWithoutMasks.csv"
    data = csvToDataframe(fpath,"emotion")

    targets = data.emotion
    inputs_raw = data.drop('emotion',axis=1)
    inputs = scaledDataFrame(inputs_raw)
    inputs_train,inputs_test,targets_train,targets_test=train_test_split(inputs,targets,test_size=0.2)

    model = KMeansClassifier(inputs_train,targets_train,7,InitAlgorithm.RANDOM)
    testAccuracy = model.accuracy(inputs_test,targets_test)
    print("Random Initialization Accuracy = %.3f" % testAccuracy)

    model = KMeansClassifier(inputs_train,targets_train,7,InitAlgorithm.KPLUS)
    testAccuracy = model.accuracy(inputs_test,targets_test)
    print("KMeans++ Initialization Accuracy = %.3f" % testAccuracy)

    model = KMeansClassifier(inputs_train,targets_train,7,InitAlgorithm.MAXSEPERATION)
    testAccuracy = model.accuracy(inputs_test,targets_test)
    print("MaxSeperation Initialization Accuracy = %.3f" % testAccuracy)
