from common import *

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.decomposition import NMF

import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import sys
import multiprocessing

PLUSINF = sys.float_info.max
MINUSINF = sys.float_info.min
np.set_printoptions(precision=3)

names = [
        "Nearest Neighbors", 
        "Neural Net", 
        "Linear SVM", 
        "RBF SVM", 
        "Gaussian Process",
        "Decision Tree", 
        "Random Forest", 
        "AdaBoost",
         "Naive Bayes", 
         "QDA"
         ]

classifiers = [
                KNeighborsClassifier(500),
                MLPClassifier(hidden_layer_sizes=(1024,1024), activation = 'relu', learning_rate_init = 0.1, momentum = 0.5),
                SVC(kernel="linear", C=0.025),
                SVC(gamma=2, C=1),
                GaussianProcessClassifier(1.0 * RBF(1.0)),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis()
    ]
    
class Classifier:
    def __init__(self,trainingInputs,trainingTargets,model,components):
        self.__trainingInputs  = trainingInputs
        self.__trainingTargets = trainingTargets
        self.__model = model
        self.__components = components
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

    @property 
    def components(self):
        return self.__components

    @property 
    def classifier(self):
        return self.__classifier

    @property 
    def extractor(self):
        return self.__extractor
    
    @property 
    def extractedTrainingInputs(self):
        return self.__extractedTrainingInputs

    def train(self):
        self.__extractor = NMF(n_components = self.components, init='random', random_state=0,max_iter = 500,solver='mu')
        self.__extractedTrainingInputs = self.extractor.fit_transform(self.trainingInputs)
        i = names.index(self.model)
        self.__classifier = classifiers[i]
        self.classifier.fit(self.extractedTrainingInputs,self.trainingTargets)
        if self.model == "Neural Net":
            n_layers = self.classifier.n_layers_
            n_outputs = self.classifier.n_outputs_
            print("Neural Net Params: n_layers = %d n_outputs = %d" %(n_layers,n_outputs))
    
    def classifyOne(self,inData):
        numpyInData = np.array(inData)
        numpyInData = numpyInData.reshape(-1,len(numpyInData))
        numpyExtractedInData = self.extractor.transform(numpyInData)
        label = self.classifier.predict(numpyExtractedInData)
        return label

    def classifyAll(self,inputs):
        numpyInputs = np.array(inputs)
        numpyExtractedInputs = self.extractor.transform(numpyInputs)
        labels = self.classifier.predict(numpyExtractedInputs)
        return labels

    def accuracy(self,inputs,targets):
        hits = 0
        inputs = np.array(inputs.astype(int))
        targets = np.array(targets.astype(int))
        labels = self.classifyAll(inputs)
        hits = len(np.where((labels - targets) == 0)[0])
        accuracy = hits/inputs.shape[0]
        return accuracy
    
               
def getAccuracy(trainInputs,trainTargets,model,components,testInputs,testTargets,tag,results=None):
    classifier = Classifier(trainInputs,trainTargets,model,components)
    testAccuracy = classifier.accuracy(testInputs,testTargets)
    trainAccuracy = classifier.accuracy(trainInputs,trainTargets)
    
    print("NMF_n_components = %d, Tag = %s: %s Classifier Test Accuracy = %.3f, Train Accuracy = %.3f" % (components,tag,model,testAccuracy,trainAccuracy))
    
    if results is None:
        result = dict()
        result['TrainAccuracy'] = trainAccuracy
        result['testAccuracy'] = testAccuracy 
        return testAccuracy
    else:
        results[tag]['TrainAccuracy'][model][components] = trainAccuracy
        results[tag]['TestAccuracy'][model][components] = testAccuracy


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

    manager = multiprocessing.Manager()
    returnDict = manager.dict() 
    jobs = dict()

    components = 7

    for tag in tags:
        if tag not in returnDict:
            trainingDict = dict()
            testDict = dict()
            for model in names:
                modelTrainingDict = dict()
                trainingDict[model] = modelTrainingDict
                modelTestDict = dict()
                testDict[model] = modelTestDict
            tagDict = dict()
            tagDict['TrainAccuracy'] = trainingDict
            tagDict['TestAccuracy'] = testDict

            returnDict[tag] = tagDict

        if tag not in jobs:
            tagDict = dict()
            jobs[tag] = tagDict
    for components in range(2,3):
        for tag in tags:
            print("===================================Start(NMF_n_components=%d, tag = %s)=================================" % (components,tag))
            for model in names:
                jobs[tag][model] = multiprocessing.Process(target=getAccuracy, args=(inputs_train[tag],targets_train[tag],model,components,inputs_test[tag],targets_test[tag],tag,returnDict))
                jobs[tag][model].start()
            for model in names:
                jobs[tag][model].join()
            print("===================================Start(NMF_n_components=%d, tag = %s)=================================" % (components,tag))
