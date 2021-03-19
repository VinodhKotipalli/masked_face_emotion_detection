
from common import *

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import statistics
import sys
import multiprocessing

from sklearn.decomposition import NMF


clusteringOutputModes = ['add','replace','other']
plotColors = {'WithMasks':{'add':'g','replace':'k','other': 'y'},'WithoutMasks':{'add':'b','replace':'r','other':'c'}}

names = [
        "Nearest Neighbors", 
        "Neural Net", 
        "Neural Net Simple",
        "Linear SVM", 
        "RBF SVM", 
        "Gaussian Process",
        "Decision Tree", 
        "Random Forest", 
        "Extra Trees",
        "AdaBoost",
         "Naive Bayes", 
         "QDA"
         ]

classifiers = [
                KNeighborsClassifier(30),
                MLPClassifier(hidden_layer_sizes=(7,7), activation = 'relu', learning_rate_init = 0.1, momentum = 0.5,max_iter=1000),
                MLPClassifier(alpha=1, max_iter=1000),
                SVC(kernel="linear", C=0.025),
                SVC(gamma=2, C=1),
                GaussianProcessClassifier(1.0 * RBF(1.0)),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                ExtraTreesClassifier(max_depth=5, n_estimators=10, max_features=1),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis()
    ]
class KMeansClassifierWithNMF:

    @property 
    def trainingInputs(self):
        return self.__trainingInputs

    @property 
    def trainingTargets(self):
        return self.__trainingTargets
    
    @property 
    def clusteringOutput(self):
        return self.__clusteringOutput

    @property 
    def clusters(self):
        return self.__clusters
    
    @property 
    def clusteringModel(self):
        return self.__clusteringModel
    
    @property 
    def classifier(self):
        return self.__classifier
    
    @property 
    def nmfComponents(self):
        return self.__nmfComponents
    
    @property 
    def extractor(self):
        return self.__extractor

    @property 
    def extractedTrainingInputs(self):
        return self.__extractedTrainingInputs

    @property 
    def kmeansProcessedTrainingInputs(self):
        return self.__kmeansProcessedTrainingInputs

    def __init__(self,trainingInputs,trainingTargets,clusters,clusteringOutput = 'replace',classifier = DecisionTreeClassifier(max_depth=5), nmfComponents=7):
        self.__trainingInputs  = copy.deepcopy(trainingInputs)
        self.__trainingTargets = copy.deepcopy(trainingTargets)
        self.__clusteringOutput = clusteringOutput
        self.__clusters = clusters
        self.__classifier = classifier
        self.__nmfComponents = nmfComponents
        self.trainExtractor()
        self.trainClusters()
        self.trainClassifier()
    
    def trainExtractor(self):
        self.__extractor = NMF(n_components = self.nmfComponents,max_iter = 1500,init='random', random_state=0)
        self.__extractedTrainingInputs = self.extractor.fit_transform(self.trainingInputs)

    def accuracy(self,inputs,targets):
        labels = self.classifyAll(inputs)
        accuracy = accuracy_score(targets, labels)
        return accuracy

    def classifyOne(self,input):
        numpyInput = np.array(input)
        numpyInput = numpyInData.reshape(-1,len(numpyInput))
        kmeansOutput = self.processWithKmeans(numpyInput)
        label = self.classifier.predict(kmeansOutput)
        return label
    
    def classifyAll(self,inputs):
        kmeansOutputs = self.processWithKmeans(inputs)
        labels = self.classifier.predict(kmeansOutputs)
        return labels

    def classifyAllWithJustNMF(self,inputs):
        extractedInputs = np.array(self.extractor.transform(inputs))
        labels = np.argmax(extractedInputs,axis=1)
        return labels

    def processWithKmeans(self,inputs):
        extractedInputs = np.array(self.extractor.transform(inputs))
        output = copy.deepcopy(extractedInputs)
        predictedLabels = self.clusteringModel.predict(extractedInputs)
        normalizedLabels = np.asfarray(copy.deepcopy(predictedLabels)) / np.asfarray(self.clusters)

        n = extractedInputs.shape[1]
        if self.clusteringOutput == 'add':
            output = np.hstack((extractedInputs, np.atleast_2d(predictedLabels).T))
        elif self.clusteringOutput == 'replace':
            output = predictedLabels[:, np.newaxis]
        #else:
            #print('clusterOutput is not in ["add","replace"], input data is not modified')
            #raise ValueError('output should be either add or replace')
        return output

    def trainClassifier(self):
        self.classifier.fit(self.kmeansProcessedTrainingInputs,self.trainingTargets)

    def trainClusters(self):
        self.__clusteringModel = KMeans(n_clusters = self.clusters, init="k-means++",max_iter=600,algorithm='auto')
        self.clusteringModel.fit(self.extractedTrainingInputs)

        
        self.__kmeansProcessedTrainingInputs = copy.deepcopy(self.extractedTrainingInputs)
        clusterLabels = copy.deepcopy(self.clusteringModel.labels_)
        normalizedLabels = np.asfarray(clusterLabels) / np.asfarray(self.clusters)

        if self.clusteringOutput == 'add':
            self.__kmeansProcessedTrainingInputs = np.hstack((self.extractedTrainingInputs, np.atleast_2d(clusterLabels).T))
        elif self.clusteringOutput == 'replace':
            self.__kmeansProcessedTrainingInputs = clusterLabels[:, np.newaxis]
        #else:
            #print('clusterOutput is not in ["add","replace"], training data is not modified')
            #raise ValueError('output should be either add or replace')
        return self

def getAccuracy(trainInputs,trainTargets,k,clusteringOutput,testInputs,testTargets,tag,results=None,classifierName=None):
    if classifierName is None:
        model = KMeansClassifierWithNMF(trainInputs,trainTargets,k,clusteringOutput)
    else:
        i = names.index(classifierName)
        classifier = classifiers[i]
        model = KMeansClassifierWithNMF(trainInputs,trainTargets,k,clusteringOutput,classifier)

    testAccuracy = model.accuracy(testInputs,testTargets)
    trainAccuracy = model.accuracy(trainInputs,trainTargets)
    print("Tag = %s, k=%d, Clustering Output = %s:  Training Accuracy = %.3f, Test Accuracy = %.3f" % (tag,k,clusteringOutput,trainAccuracy,testAccuracy))   

    if results is None:
        result = dict()
        result['TrainAccuracy'] = trainAccuracy
        result['TestAccuracy'] = testAccuracy 
        return result
    else:
        results[tag]['TrainAccuracy'][clusteringOutput][k] = trainAccuracy
        results[tag]['TestAccuracy'][clusteringOutput][k] = testAccuracy


if __name__ == '__main__':


    targets = dict()
    inputs = dict()
    targets_train = dict()
    inputs_train = dict()
    targets_test = dict()
    inputs_test = dict()    

    testSplit = 0.90

    fpath = projdir+"/datasets/facesWithoutMasks.csv"
    data = csvToDataframe(fpath,"emotion")
    targets['WithoutMasks'] = data.emotion
    inputs_raw = data.drop('emotion',axis=1)
    inputs['WithoutMasks'] = scaledDataFrame(inputs_raw)
    inputs_train['WithoutMasks'],inputs_test['WithoutMasks'],targets_train['WithoutMasks'],targets_test['WithoutMasks']=train_test_split(inputs['WithoutMasks'],targets['WithoutMasks'],test_size=testSplit)

    fpath = projdir+"/datasets/facesWithMasks.csv"
    data = csvToDataframe(fpath,"emotion")
    targets['WithMasks'] = data.emotion
    inputs_raw = data.drop('emotion',axis=1)
    inputs['WithMasks'] = scaledDataFrame(inputs_raw)
    inputs_train['WithMasks'],inputs_test['WithMasks'],targets_train['WithMasks'],targets_test['WithMasks']=train_test_split(inputs['WithMasks'],targets['WithMasks'],test_size=testSplit)

    manager = multiprocessing.Manager()
    returnDict = manager.dict() 
    jobs = dict()
    tags = ['WithoutMasks','WithMasks']
    kMin = [7,80,150]
    kMax = [70,100,700]
    kStep = [7,10,50]
    kRange = []
    for i in range(len(kMin)):
        kRange = kRange + list(range(kMin[i],kMax[i]+1,kStep[i]))


    classifierName  = "Neural Net"

    kRange = [7,14]

    for k in kRange:
        print("===================================Start(k=%d)=================================" % k)
        for tag in tags:
            if tag not in returnDict:
                tagDict = manager.dict()
                trainingDict = manager.dict() 
                testDict = manager.dict()
                for clusteringOutput in clusteringOutputModes:
                    clusteringOutputTrainingDict = manager.dict() 
                    clusteringOutputTestDict = manager.dict()
                    trainingDict[clusteringOutput] = clusteringOutputTrainingDict
                    testDict[clusteringOutput] = clusteringOutputTestDict
                tagDict['TrainAccuracy'] = trainingDict
                tagDict['TestAccuracy'] = testDict
                returnDict[tag] = tagDict 
            if tag not in jobs:
                tagDict = dict()
                for clusteringOutput in clusteringOutputModes:
                    tagDict[clusteringOutput] = dict()
                jobs[tag] = tagDict

        for tag in tags:
            for clusteringOutput in clusteringOutputModes:
                jobs[tag][clusteringOutput][k] = multiprocessing.Process(target=getAccuracy, args=(inputs_train[tag],targets_train[tag],k,clusteringOutput,inputs_test[tag],targets_test[tag],tag,returnDict,classifierName))
                jobs[tag][clusteringOutput][k].start()
        for tag in tags:
            for clusteringOutput in clusteringOutputModes:
                jobs[tag][clusteringOutput][k].join()
        print("===================================End(k=%d)=================================" % k)

    for dTag in ['TrainAccuracy','TestAccuracy']:
        x = kRange
        y = dict()
        for tag in tags:
            for clusteringOutput in clusteringOutputModes:
                plotColor = plotColors[tag][clusteringOutput]
                data = returnDict[tag][dTag][clusteringOutput]
                yLabel = ""
                if clusteringOutput == 'add' and tag == 'WithMasks':
                    yLabel = "Images with Masks, inputs + cluster labels"
                if clusteringOutput == 'replace' and tag == 'WithMasks':
                    yLabel = "Images with Masks, only cluster labels"   
                if clusteringOutput == 'add' and tag == 'WithoutMasks':
                    yLabel = "Images without Masks, inputs + cluster labels"
                if clusteringOutput == 'replace' and tag == 'WithoutMasks':
                    yLabel = "Images without Masks, only cluster labels"                  
                y[yLabel] = data.values()
                plt.plot(x,y[yLabel],plotColor,label = yLabel)
        plt.legend(loc='lower right')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Accuracy(%)')
        if dTag == 'TrainAccuracy':
            plt.title('Accuracy for Facial Expression Detection on Training Data')
        if dTag == 'TestAccuracy':
            plt.title('Accuracy for Facial Expression Detection on Test Data')
        plt.show()

