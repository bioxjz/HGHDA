from util.config import OptionConf
from util.dataSplit import *
from multiprocessing import Process,Manager
from util.io import FileIO
from time import strftime,localtime,time
import pandas as pd
import numpy as np
import mkl
class HDR(object):
    def __init__(self,config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.measure = []
        self.config =config
        self.ratingConfig = OptionConf(config['ratings.setup'])
        if self.config.contains('evaluation.setup'):
            self.evaluation = OptionConf(config['evaluation.setup'])
            self.trainingData = FileIO.loadDataSet(config, config['datapath'])

        else:
            print('Wrong configuration of evaluation!')
            exit(-1)

        print('Reading data and preprocessing...')

    def execute(self):
        #import the model module
        importStr = 'from ' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec (importStr)
        if self.evaluation.contains('-cv'):
            k = int(self.evaluation['-cv'])
            if k < 2 or k > 10: #limit to 2-10 fold cross validation
                print("k for cross-validation should not be greater than 10 or less than 2")
                exit(-1)
            mkl.set_num_threads(max(1,mkl.get_max_threads()//k))
            #create the manager for communication among multiple processes
            manager = Manager()
            mDict = manager.dict()
            i = 1
            tasks = []
            for train,test in DataSplit.crossValidation(self.trainingData,k):
                fold = '['+str(i)+']'
                recommender = self.config['model.name'] + "(self.config,train,test,fold)"
               #create the process
                p = Process(target=run,args=(mDict,eval(recommender),i))
                tasks.append(p)
                i+=1
            #start the processes
            for p in tasks:
                p.start()
                if not self.evaluation.contains('-p'):
                    p.join()
            #wait until all processes are completed
            if self.evaluation.contains('-p'):
                for p in tasks:
                    p.join()
            #compute the average error of k-fold cross validation
            self.measure = [dict(mDict)[i] for i in range(1,k+1)]
            res = []
            for i in range(len(self.measure[0])):
                measure = self.measure[0][i].split(':')[0]
                total = 0
                for j in range(k):
                    total += float(self.measure[j][i].split(':')[1])
                res.append(measure + ':' + str(total / k) + '\n')
            #output result
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = OptionConf(self.config['output.setup'])['-dir']
            fileName = self.config['model.name'] +'@'+currentTime+'-'+str(k)+'-fold-cv' + '.txt'
            FileIO.writeFile(outDir,fileName,res)
            print('The result of %d-fold cross validation:\n%s' %(k,''.join(res)))
        else:
 
            recommender = self.config['model.name'] + '(self.config,self.trainingData,self.testData)'
            eval(recommender).execute()


def run(measure,algor,order):
    measure[order] = algor.execute()