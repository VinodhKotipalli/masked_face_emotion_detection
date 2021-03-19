from common import *

from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import sys
import multiprocessing

PLUSINF = sys.float_info.max
MINUSINF = sys.float_info.min
np.set_printoptions(precision=3)

class DTreeClassifier:
    def __init__(self,trainingInputs,trainingTargets):
        self.__trainingInputs  = trainingInputs
        self.__trainingTargets = trainingTargets
        self.train()

    @property 
    def trainingInputs(self):
        return self.__trainingInputs

    @property 
    def trainingTargets(self):
        return self.__trainingTargets

    @property 
    def model(self):
        return self.__model

    def train(self):
        self.__model = tree.DecisionTreeClassifier()
        self.model.fit(self.trainingInputs,self.trainingTargets)
    
    def classifyOne(self,inData):
        model = self.model
        numpyInData = np.array(inData)
        numpyInData = numpyInData.reshape(-1,len(numpyInData))
        label = self.model.predict(numpyInData)
        return label

    def classifyAll(self,inputs):
        numpyInputs = np.array(inputs)
        labels = self.model.predict(numpyInputs)
        return labels

    def accuracy(self,inputs,targets):
        hits = 0
        inputs = np.array(inputs.astype(int))
        targets = np.array(targets.astype(int))
        labels = self.classifyAll(inputs)
        hits = len(np.where((labels - targets) == 0)[0])
        accuracy = hits/inputs.shape[0]
        return accuracy
    
               
def getAccuracy(trainInputs,trainTargets,k,initAlgorithim,testInputs,testTargets,tag,results=None):
    model = KMeansClassifier(trainInputs,trainTargets,k,initAlgorithim)
    testAccuracy = model.accuracy(testInputs,testTargets)
    trainAccuracy = model.accuracy(trainInputs,trainTargets)
    if initAlgorithim == InitAlgorithm.RANDOM:
        print("Tag = %s k=%d: Random Initialization Test Accuracy = %.3f, Train Accuracy = %.3f" % (tag,k,testAccuracy,trainAccuracy))
    if initAlgorithim == InitAlgorithm.KPLUS:
        print("Tag = %s k=%d: KMeans++ Initialization Test Accuracy = %.3f, Train Accuracy = %.3f" % (tag,k,testAccuracy,trainAccuracy))
    if initAlgorithim == InitAlgorithm.MAXSEPERATION:
        print("Tag = %s k=%d: MaxSeperation Initialization Test Accuracy = %.3f, Train Accuracy = %.3f" % (tag,k,testAccuracy,trainAccuracy))
    if initAlgorithim == InitAlgorithm.MINGROUPING:
        print("Tag = %s k=%d: MinGrouping Initialization Test Accuracy = %.3f, Train Accuracy = %.3f" % (tag,k,testAccuracy,trainAccuracy))
    
    
    if results is None:
        result = dict()
        result['TrainAccuracy'] = trainAccuracy 
        return testAccuracy
    else:
        results[tag]['TrainAccuracy'][initAlgorithim][k] = trainAccuracy
        results[tag]['TestAccuracy'][initAlgorithim][k] = testAccuracy


if __name__ == '__main__':


    targets = dict()
    inputs = dict()
    targets_train = dict()
    inputs_train = dict()
    targets_test = dict()
    inputs_test = dict()    

    fpath = projdir+"/datasets/facesWithoutMasks.csv"
    data = csvToDataframe(fpath,"emotion")
    targets['WithoutMasks'] = data.emotion
    inputs_raw = data.drop('emotion',axis=1)
    inputs['WithoutMasks'] = scaledDataFrame(inputs_raw)
    inputs_train['WithoutMasks'],inputs_test['WithoutMasks'],targets_train['WithoutMasks'],targets_test['WithoutMasks']=train_test_split(inputs['WithoutMasks'],targets['WithoutMasks'],test_size=0.2)

    fpath = projdir+"/datasets/facesWithMasks.csv"
    data = csvToDataframe(fpath,"emotion")
    targets['WithMasks'] = data.emotion
    inputs_raw = data.drop('emotion',axis=1)
    inputs['WithMasks'] = scaledDataFrame(inputs_raw)
    inputs_train['WithMasks'],inputs_test['WithMasks'],targets_train['WithMasks'],targets_test['WithMasks']=train_test_split(inputs['WithMasks'],targets['WithMasks'],test_size=0.2)

    tags = ['WithoutMasks','WithMasks']

    for tag in tags:
        model = DTreeClassifier(inputs_train[tag],targets_train[tag])
        trainAccuracy = model.accuracy(inputs_train[tag],targets_train[tag])
        testAccuracy = model.accuracy(inputs_test[tag],targets_test[tag])
        print("Tag = %s: Test Accuracy = %.3f, Train Accuracy = %.3f" % (tag,testAccuracy,trainAccuracy))
