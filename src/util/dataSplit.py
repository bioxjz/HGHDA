from random import random
from util.io import FileIO
class DataSplit(object):

    def __init__(self):
        pass
    @staticmethod
    def crossValidation(data,k,output=False,path='./',order=1):
        if k<=1 or k>10:
            k=3
        for i in range(k):
            trainingSet = []
            testSet = []
            for ind,line in enumerate(data):
                if ind%k == i:
                    testSet.append(line[:])
                else:
                    trainingSet.append(line[:])
            yield trainingSet,testSet


