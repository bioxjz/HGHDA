import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from rating import Rating
from util.io import FileIO
from util.config import OptionConf
from util.log import Log
from os.path import abspath
from time import strftime,localtime,time
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import random
class Recommender(object):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.data = Rating(self.config, trainingSet, testSet)
        #print(self.data.herb)
        #print(self.data.disease)
        #print(len(self.data.disease))
        self.foldInfo = fold
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.measure = []
        self.recOutput = []
        self.num_herbs, self.num_diseases, self.train_size = self.data.trainingSize()
        self.num_compounds,self.num_proteins=self.data.cpSize()

    def initializing_log(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.log = Log(self.modelName, self.modelName + self.foldInfo + ' ' + currentTime)
        #save configuration
        self.log.add('### model configuration ###')
        for k in self.config.config:
            self.log.add(k+'='+self.config[k])

    def readConfiguration(self):
        self.modelName = self.config['model.name']
        self.output = OptionConf(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()

    def printAlgorConfig(self):
        "show model's configuration"
        print('Model:',self.config['model.name'])
        print('Ratings dataset:',abspath(self.config['datapath']))
        if OptionConf(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:', abspath(OptionConf(self.config['evaluation.setup'])['-testSet']))
        #print dataset statistics
        print('Training set size: (herb count: %d, disease count %d, record count: %d)' %(self.data.trainingSize()))
        print('Test set size: (herb count: %d, disease count %d, record count: %d)' %(self.data.testSize()))
        print('='*80)
        #print specific parameters if applicable
        if self.config.contains(self.config['model.name']):
            parStr = ''
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr+=key[1:]+':'+args[key]+'  '
            print('Specific parameters:',parStr)
            print('=' * 80)

    def initModel(self):
        pass

    def trainModel(self):
        'build the model (for model-based Models )'
        pass

    def trainModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    #for rating prediction
    def predictForRating(self, u, i):
        pass

    #for disease prediction
    def predictForRanking(self,u):
        pass

    def checkRatingBoundary(self,prediction):
        pass
    def evalRatings(self):
        pass

    def softmax(self,x):
    #    orig_shape=x.shape
        if len(x.shape)>1:
            tmp=np.max(x,axis=1)
            x-=tmp.reshape((x.shape[0],1))
            x=np.exp(x)
            tmp=np.sum(x,axis=1)
            x/=tmp.reshape((x.shape[0],1))
        else:
            tmp=np.max(x)
            x-=tmp
            x=np.exp(x)
            tmp=np.sum(x)
            x/=tmp
        return x

    def evalRanking(self):
        print('recommender evalRanking-------------------------------------------------------')

        candidates = self.predictForRanking()
        print(candidates)
        #print(candidates.shape)
        print(type(candidates))
        #candidates=(candidates-np.min(candidates))/(np.max(candidates)-np.min(candidates))
        candidates=1/(1+np.exp(-candidates))
        print(candidates)
        # test_premat=[]
        # test_realmat=[]
        herb_mat = []
        lable_mat = []
        for i, herb in enumerate(self.data.testSet_h):
            diseaselist=self.data.testSet_h[herb].keys()
            for disease in diseaselist:
                herb_mat.append(candidates[self.data.herb[herb]][self.data.disease[disease]])
                lable_mat.append(self.data.testSet_h[herb][disease])


        #选取和测试样本相同数量的负样本
        #negativelist=pd.read_csv('./dataset/negativehdlist.txt',sep='\t',header=None)
        #randomlist=random.sample(range(0,len(negativelist)),len(self.data.testData))
        #negativel=negativelist.iloc[randomlist]
        #for line in negativel.values:
        #    # print(line[0])
        #    # print(line[1])
        #    herb_mat.append(candidates[self.data.herb[str(line[0])]][self.data.disease[str(line[1])]])
        #    lable_mat.append(0)
        #for i, user in enumerate(self.data.testSet_h):
        #    itemlist=self.data.testSet_h[user].keys()
        #    for item in itemlist:
        #        herb_mat.append(candidates[self.data.herb[user]][self.data.disease[item]])
        #        lable_mat.append(1)
        #        #lable_mat.append(self.data.testSet_u[user][item])
        #    # test_premat.append(np.array(user_mat))
        #    # test_realmat.append(np.array(lable_mat))
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        data0={'lable':lable_mat,'predict':herb_mat}
        dataframe0=pd.DataFrame(data0)
        dataframe0.to_csv('./results/cv/'+ currentTime + self.foldInfo + '.txt',columns=['lable','predict'],index=False,header=None)
        auc=roc_auc_score(lable_mat,herb_mat)
        print('auc:',auc)


  
    def execute(self):
        self.readConfiguration()
        self.initializing_log()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            print('Loading model %s...' %self.foldInfo)
            self.loadModel()
        else:
            print('Initializing model %s...' %self.foldInfo)
            self.initModel()
            print('Building Model %s...' %self.foldInfo)
            try:
                if self.evalSettings.contains('-tf'):
                    import tensorflow
                    self.trainModel_tf()
                else:
                    self.trainModel()
            except ImportError:
                self.trainModel()
        print('Predicting %s...' %self.foldInfo)
        self.evalRanking()
        if self.isSaveModel:
            print('Saving model %s...' %self.foldInfo)
            self.saveModel()
        return self.measure



