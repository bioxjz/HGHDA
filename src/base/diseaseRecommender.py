from base.recommender import Recommender
from util import config
import numpy as np
from random import shuffle
class diseaseRecommender(Recommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(diseaseRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.bestPerformance = []
        self.earlyStop = 0

    def readConfiguration(self):
        super(diseaseRecommender, self).readConfiguration()
        self.emb_size = int(self.config['num.factors'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        learningRate = config.OptionConf(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        if self.evalSettings.contains('-tf'):
            self.batch_size = int(self.config['batch_size'])
        regular = config.OptionConf(self.config['reg.lambda'])
        self.regU,self.regI,self.regB= float(regular['-u']),float(regular['-i']),float(regular['-b'])

    def printAlgorConfig(self):
        super(diseaseRecommender, self).printAlgorConfig()
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Regularization parameter: regU %.3f, regI %.3f, regB %.3f' %(self.regU,self.regI,self.regB))

    def initModel(self):
        print('iter initModel-------------------------------------------------------')
        pass




