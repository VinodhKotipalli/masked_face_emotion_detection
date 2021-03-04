import os
import copy
from PIL import Image
from collections.abc import Iterable 


projdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ':' in projdir:
    projdir = projdir.split(':')[1].replace('\\', '/')




class Emotion:
    ANGRY       = 0
    DISGUST     = 1
    FEAR        = 2
    HAPPY       = 3
    NEUTRAL     = 4
    SAD         = 5
    SURPRISE    = 6

def imageToPixels(ipath):
    img = Image.open(ipath).convert('L')
    WIDTH,HEIGHT = img.size
    return list(img.getdata())


def imageToLabeledData(ipath):
    data = imageToPixels(ipath)

    emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']
    labels = [Emotion.ANGRY,Emotion.DISGUST,Emotion.FEAR,Emotion.HAPPY,Emotion.NEUTRAL,Emotion.SAD,Emotion.SURPRISE]
    label = None
    for emotion in emotions:
        if emotion in ipath:
            label = labels[emotions.index(emotion)]
            break
    labeledData = copy.deepcopy(data)
    if label is None:
        labeledData.append(len(labels))
    else:
        labeledData.append(label)
    return labeledData

def imagesToLabeledCSV(dirpaths,csvpath,skipTag=None,selectTag=None):
    csvfile = open(csvpath, 'w')
    if isinstance(dirpaths, list): 
        pathList = copy.deepcopy(dirpaths)
    else:
        pathList = [dirpaths]
    for dirpath in pathList:
        print("collecting data from: %s and \n\twriting to: %s" %(dirpath,csvpath))
        if skipTag is not None:
            print("\t\tskip paths with : %s" % skipTag)
        if selectTag is not None:
            print("\t\tselecting paths with : %s" % selectTag)
        for root,dirs,files in os.walk(dirpath):
            for file in files:
                fpath = os.path.join(root,file).replace('\\', '/')
                useFile = True
                if '.jpg' not in fpath:
                    useFile = False
                if useFile and (skipTag is not None) and (skipTag in fpath):
                    useFile = False
                if useFile and (selectTag is not None) and (selectTag not in fpath):
                    useFile = False
                if useFile:
                    labeledData = imageToLabeledData(fpath)
                    labeledData_row = ','.join(map(str,labeledData))
                    csvfile.write(labeledData_row + '\n')

    csvfile.close()

if __name__ == '__main__':
    dirpath = projdir + "/data/train"
    csvpath = projdir + "/datasets/facesWithoutMasks_train.csv"
    imagesToLabeledCSV(dirpath,csvpath,"surgical",None)

    csvpath = projdir + "/datasets/facesWithMasks_train.csv"
    imagesToLabeledCSV(dirpath,csvpath,None,"surgical")

    dirpath = projdir + "/data/test"
    csvpath = projdir + "/datasets/facesWithoutMasks_test.csv"
    imagesToLabeledCSV(dirpath,csvpath,"surgical",None)

    csvpath = projdir + "/datasets/facesWithMasks_test.csv"
    imagesToLabeledCSV(dirpath,csvpath,None,"surgical")

    dirpaths = [projdir + "/data/train",projdir + "/data/test"]
    csvpath = projdir + "/datasets/facesWithoutMasks.csv"
    imagesToLabeledCSV(dirpaths,csvpath,"surgical",None)
    csvpath = projdir + "/datasets/facesWithMasks.csv"
    imagesToLabeledCSV(dirpaths,csvpath,None,"surgical")


