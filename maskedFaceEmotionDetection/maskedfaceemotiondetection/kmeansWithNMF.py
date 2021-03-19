from common import *

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import sys
import multiprocessing

from sklearn.decomposition import NMF


PLUSINF = sys.float_info.max
MINUSINF = sys.float_info.min
np.set_printoptions(precision=3)

class InitAlgorithm:
    RANDOM = 0
    KPLUS = 1
    MAXSEPERATION = 2
    MINGROUPING = 3

initAlgorithms = [InitAlgorithm.RANDOM,InitAlgorithm.KPLUS,InitAlgorithm.MAXSEPERATION,InitAlgorithm.MINGROUPING]

initAlgorithmNames = ["Random","KMeans++","MaxSeperation","MinGrouping"]

class KMeansClassifierWithNMF:
    def __init__(self,trainingInputs,trainingTargets,clusters,components=7,initAlgorithm=None,maxSample=5000):
        self.__trainingInputs  = trainingInputs
        self.__trainingTargets = trainingTargets
        self.__clusters = clusters
        self.__initAlgorithm = initAlgorithm
        self.__maxSample = maxSample
        self.__components = components
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
    def components(self):
        return self.__components

    @property 
    def initAlgorithm(self):
        return self.__initAlgorithm

    @property 
    def classifier(self):
        return self.__classifier

    @property 
    def extractor(self):
        return self.__extractor
        
    @property 
    def maxSample(self):
        return self.__maxSample

    @property 
    def trueLabelsMap(self):
        return self.__trueLabelsMap
    
    @property 
    def trueLabelsProbabilityMap(self):
        return self.__trueLabelsProbabilityMap

    @property 
    def extractedTrainingInputs(self):
        return self.__extractedTrainingInputs

    

    def allPairDistances(self,A,B):
        tensorA = torch.tensor(np.array(A,dtype=float))
        tensorB = torch.tensor(np.array(B,dtype=float))

        AA = torch.sum(torch.pow(tensorA,2),1,keepdim=True).expand(tensorA.shape[0],tensorB.shape[0])
        BB = torch.sum(torch.pow(tensorB,2),1,keepdim=True).expand(tensorB.shape[0],tensorA.shape[0]).t()
        AB = torch.mm(tensorA,tensorB.t())

        return torch.sqrt(AA - 2*AB + BB)

    def minGrouping(self):
        k = self.clusters
        maxSample = self.maxSample
        trainingInputs = self.extractedTrainingInputs
        n = trainingInputs.shape[0]
        if n > maxSample:
            trainingInputs = trainingInputs.sample(n=maxSample)      
        n = trainingInputs.shape[0]
        m = int(0.75*(n/k))

        f = trainingInputs.shape[1]
        centroids = np.empty((k,f))

        centroidNodes = np.empty(k,dtype=int)
        centroidNodes.fill(-1)

        groupedNodes = np.empty(n,dtype=int)
        groupedNodes.fill(-1)

        fillerRows = np.empty((m,n))
        fillerRows.fill(PLUSINF / n)
        fillerCols = np.empty((n,m))
        fillerCols.fill(PLUSINF / n)

        distances = self.allPairDistances(trainingInputs,trainingInputs)
        distances = np.array(distances)
        distances = np.nan_to_num(distances, copy=True, nan=PLUSINF, posinf=None, neginf=None)
        pending = np.where(centroidNodes<0)[0]
        while len(pending) > 0:
            mGroup = np.empty(m,dtype=int)
            mGroup.fill(-1)
            minNodes = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
            mGroup[0] = minNodes[0]
            mGroup[1] = minNodes[1]

            mPending = np.where(mGroup<0)[0]
            while len(mPending) > 0:
                mAssigned = mGroup[np.where(mGroup>=0)]
                for i in [(x,y) for x in mAssigned for y in mAssigned if x != y]:
                    distances[i] = PLUSINF / n
                distancesFromAssigned = distances[mAssigned,:]
                sumDistancesFromAssigned = distancesFromAssigned.sum(axis=0)
                minNode = np.argmin(sumDistancesFromAssigned)
                mGroup[mPending[0]] = minNode
                mPending = np.where(mGroup<0)[0]

            centroidNodes[pending[0]] = mGroup[0]

            mInputs = np.array(trainingInputs)[mGroup,:]
            centroids[pending[0],:] = mInputs.mean(axis=0)


            distances[mGroup,:] = copy.deepcopy(fillerRows)
            distances[:,mGroup] = copy.deepcopy(fillerCols)

            pending = np.where(centroidNodes<0)[0]
        return centroids

    def maxSeperation(self):
        k = self.clusters
        maxSample = self.maxSample
        trainingInputs = copy.deepcopy(self.extractedTrainingInputs)
        n = trainingInputs.shape[0]
        if n > maxSample:
            trainingInputs = trainingInputs.sample(n=maxSample)
        
        f = trainingInputs.shape[1]
        centroids = np.empty((k,f))

        centroidNodes = np.empty(k,dtype=int)
        centroidNodes.fill(-1)

        distances = self.allPairDistances(trainingInputs,trainingInputs)
        distances = np.array(distances)
        distances = np.nan_to_num(distances, copy=True, nan=MINUSINF, posinf=None, neginf=None)

        maxNodes = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
        centroidNodes[0] = maxNodes[0]
        centroidNodes[1] = maxNodes[1]

        centroids[0,:] = np.array(trainingInputs)[maxNodes[0],:]
        centroids[1,:] = np.array(trainingInputs)[maxNodes[1],:]
        pending = np.where(centroidNodes<0)[0]
        while len(pending) > 0:
            assigned = centroidNodes[np.where(centroidNodes>=0)]
            for i in [(x,y) for x in assigned for y in assigned if x != y]:
                distances[i] = MINUSINF
            distancesFromAssigned = distances[assigned,:]
            sumDistancesFromAssigned = distancesFromAssigned.sum(axis=0)
            maxNode = np.argmax(sumDistancesFromAssigned)

            centroidNodes[pending[0]] = maxNode

            centroids[pending[0],:] = np.array(trainingInputs)[maxNode,:]

            pending = np.where(centroidNodes<0)[0]
        return centroids

    def findInitialCentroids(self):
        if self.initAlgorithm is None or self.initAlgorithm == InitAlgorithm.RANDOM :
            return ('random',10)
        if self.initAlgorithm == InitAlgorithm.KPLUS:
            return ('k-means++',10)
        if self.initAlgorithm == InitAlgorithm.MAXSEPERATION:
            return (self.maxSeperation(),1)
        if self.initAlgorithm == InitAlgorithm.MINGROUPING:
            return (self.minGrouping(),1)      

    def train(self):
        self.__extractor = NMF(n_components = self.components, init='random', random_state=0,max_iter = 1000,solver='mu')
        self.__extractedTrainingInputs = self.extractor.fit_transform(self.trainingInputs)
        k = self.clusters
        initParams = self.findInitialCentroids()
        self.__classifier = KMeans(n_clusters=k,init=initParams[0],n_init=initParams[1],max_iter=600,algorithm='auto')
        self.classifier.fit(self.extractedTrainingInputs)
        labels = self.classifier.labels_
        inertia = self.classifier.inertia_
        trueLabelsMap = np.empty(k,dtype=int)
        trueLabelsMap.fill(-1)
        accuracy = np.zeros(k)
        t = self.trainingTargets
        l = len(t.unique())

        trueLabelsProbabilityMap = np.empty((k,l))
        trueLabelsProbabilityMap.fill(-1.0)

        for i in range(k):
            iTrueLabels = [t.iloc[j] for j in range(len(t)) if labels[j] == i]
            iLabel = statistics.mode(iTrueLabels)
            trueLabelsMap[i] = iLabel
            trueLabelsProbabilityMap[i,:] = [(iTrueLabels.count(j) / len(iTrueLabels))  for j in range(l)] 

            accuracy[i] = len([j for j in iTrueLabels if j == iLabel])/len(iTrueLabels)
            #print("iLabel = %d iLabelProbabilities = %s iAccuracy = %0.3f" % (iLabel, str(trueLabelsProbabilityMap[i,:]),accuracy[i]))
        
        self.__trueLabelsProbabilityMap = trueLabelsProbabilityMap
        self.__trueLabelsMap = trueLabelsMap
    
    def classifyOne(self,inData):
        numpyInData = np.array(inData)
        numpyInData = numpyInData.reshape(-1,len(numpyInData))
        labelsProbability = self.getLabelsProbability(numpyInData)

        #clusterId = self.classifier.predict(numpyInData)
        #label = self.trueLabelsMap[clusterId]

        #labelChoices = range(labelProbability.shape[1]) 
        #label = np.random.choice(labelChoices, p = labelsProbability[0])
        labels = np.argmax(labelsProbability,axis=1)
        return label

    def getLabelsProbability(self,inputs):
        centroids = self.classifier.cluster_centers_
        extractedInputs = self.extractor.transform(inputs)
        distances = self.allPairDistances(inputs,centroids)
        distancesInverse = np.reciprocal(np.array(distances))
        membershipsProbability = distancesInverse/distancesInverse.sum(axis=1)[:,None]
        labelsProbability = np.matmul(membershipsProbability,self.trueLabelsProbabilityMap)   
        return labelsProbability

    def classifyAll(self,inputs):
        labelsProbability = self.getLabelsProbability(inputs)

        #clusterIds = self.classifier.predict(numpyInputs)
        #labels = self.trueLabelsMap[clusterIds]

        #labels = np.empty(labelsProbability.shape[0],dtype=int)
        #labels.fill(-1)
        #labelChoices = range(labelsProbability.shape[1])
        #for i in range(labelsProbability.shape[0]):
        #    labels[i] = np.random.choice(labelChoices, p = labelsProbability[1])
        
        labels = np.argmax(labelsProbability,axis=1)
        return labels

    def accuracy(self,inputs,targets):
        hits = 0
        inputs = np.array(inputs.astype(int))
        targets = np.array(targets.astype(int))
        labels = self.classifyAll(inputs)
        hits = len(np.where((labels - targets) == 0)[0])
        accuracy = hits/inputs.shape[0]
        return accuracy
    
               
def getAccuracy(trainInputs,trainTargets,k,initAlgorithm,nmfComponents,testInputs,testTargets,tag,results=None):
    classifier = KMeansClassifierWithNMF(trainInputs,trainTargets,k,nmfComponents,initAlgorithm)
    testAccuracy = classifier.accuracy(testInputs,testTargets)
    trainAccuracy = classifier.accuracy(trainInputs,trainTargets)
    i = initAlgorithms.index(initAlgorithm)
    initAlgorithmName = initAlgorithmNames[i]
    print("Tag = %s, k = %d, Initialization = %s, NMF components = %d:  Test Accuracy = %.3f, Train Accuracy = %.3f" % (tag,k,initAlgorithmName,nmfComponents,testAccuracy,trainAccuracy))   
    
    if results is None:
        result = dict()
        result['TrainAccuracy'] = trainAccuracy 
        return testAccuracy
    else:
        results[tag]['TrainAccuracy'][nmfComponents][k] = trainAccuracy
        results[tag]['TestAccuracy'][nmfComponents][k] = testAccuracy


if __name__ == '__main__':


    targets = dict()
    inputs = dict()
    targets_train = dict()
    inputs_train = dict()
    targets_test = dict()
    inputs_test = dict()    

    testSplit = 0.95

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
    kMin = 7
    kMax = 14
    kStep = 7
    nmfMin = 7
    nmfMax = 49
    nmfStep = 7

    nmfComponentList = range(nmfMin,nmfMax+1,nmfStep)
    plotColors = ['b','r','g','k','c','m','y']
    for k in range(kMin,kMax+1,kStep):
        print("===================================Start(k=%d)=================================" % k)
        for tag in tags:
            if tag not in returnDict:
                tagDict = manager.dict()
                trainingDict = manager.dict() 
                testDict = manager.dict() 
                for nmfComponents in range(nmfMin,nmfMax+1,nmfStep):
                    nmfTrainingDict = manager.dict() 
                    nmfTestDict = manager.dict()
                    trainingDict[nmfComponents] = nmfTrainingDict
                    testDict[nmfComponents] = nmfTestDict
                tagDict['TrainAccuracy'] = trainingDict
                tagDict['TestAccuracy'] = testDict
                returnDict[tag] = tagDict

            if tag not in jobs:
                tagDict = dict()
                for nmfComponents in range(nmfMin,nmfMax+1,nmfStep):
                    tagDict[nmfComponents] = dict()
                jobs[tag] = tagDict
        for tag in tags:
            for nmfComponents in range(nmfMin,nmfMax+1,nmfStep):
                jobs[tag][nmfComponents][k] = multiprocessing.Process(target=getAccuracy, args=(inputs_train[tag],targets_train[tag],k,InitAlgorithm.KPLUS,nmfComponents,inputs_test[tag],targets_test[tag],tag,returnDict))
                jobs[tag][nmfComponents][k].start()
        for tag in tags:
            for nmfComponents in range(nmfMin,nmfMax+1,nmfStep):
                jobs[tag][nmfComponents][k].join()
        print("===================================End(k=%d)=================================" % k)

    for tag in tags:
        for dTag in ['TrainAccuracy','TestAccuracy']:
            x = range(kMin,kMax+1,kStep)
            y = dict()
            for nmfComponents in range(nmfMin,nmfMax+1,nmfStep):
                yLabel = "NMF extracted components = " + str(nmfComponents)
                y[yLabel] = returnDict[tag][dTag][nmfComponents].values()

                i = nmfComponentList.index(nmfComponents) % len(plotColors)

                plt.plot(x,y[yLabel],plotColors[i],label = yLabel )

            plt.legend(loc='lower right')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Accuracy(%)')
            if dTag == 'TrainAccuracy' and tag == 'WithMasks':
                plt.title('Training Accuracy on Images with Masks')
            if dTag == 'TrainAccuracy' and tag == 'WithoutMasks':
                plt.title('Training Accuracy on Images without Masks')
            if dTag == 'TestAccuracy' and tag == 'WithMasks':
                plt.title('Test Accuracy on Images with Masks')
            if dTag == 'TestAccuracy' and tag == 'WithoutMasks':
                plt.title('Test Accuracy on Images without Masks')
            plt.show()