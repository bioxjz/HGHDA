import os.path
from os import makedirs,remove
from re import compile,findall,split
from .config import OptionConf
import random
class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+file,op) as f:
            f.writelines(content)

    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            remove(filePath)

    @staticmethod
    def loadDataSet(conf, file, bTest=False):
        trainingData = []
        testData = []
        ratingConfig = OptionConf(conf['ratings.setup'])
        if not bTest:
            print('loading training data...')
        else:
            print('loading test data...')
        with open(file) as f:
            ratings = f.readlines()
        # ignore the headline
        if ratingConfig.contains('-header'):
            ratings = ratings[1:]
        # order of the columns
        order = ratingConfig['-columns'].strip().split()
        delim = ' |,|\t'
        if ratingConfig.contains('-delim'):
            delim=ratingConfig['-delim']
        for lineNo, line in enumerate(ratings):
            hda = split(delim,line.strip())
            if not bTest and len(order) < 2:
                print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            try:
                herbId = hda[int(order[0])]
                diseaseId = hda[int(order[1])]
                if len(order)<3:
                    rating = 1 #default value
                else:
                    rating  = hda[int(order[2])]
            except ValueError:
                print('Error! Have you added the option -header to the rating.setup?')
                exit(-1)
            if bTest:
                testData.append([herbId, diseaseId, float(rating)])
            else:
                trainingData.append([herbId, diseaseId, float(rating)])

        with open('./dataset/negativehdlist.txt') as f:
            ratings = f.readlines()
        # ignore the headline
        if ratingConfig.contains('-header'):
            ratings = ratings[1:]
        # order of the columns
        order = ratingConfig['-columns'].strip().split()
        delim = ' |,|\t'
        if ratingConfig.contains('-delim'):
            delim=ratingConfig['-delim']
        new_ratings=random.sample(ratings,2354225)

        for lineNo, line in enumerate(new_ratings):
            hda = split(delim,line.strip())
            if not bTest and len(order) < 2:
                print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            try:
                herbId = hda[int(order[0])]
                diseaseId = hda[int(order[1])]
                if len(order)<3:
                    rating = 0 #default value
                else:
                    rating  = hda[int(order[2])]
            except ValueError:
                print('Error! Have you added the option -header to the rating.setup?')
                exit(-1)
            if bTest:
                testData.append([herbId, diseaseId, float(rating)])
            else:
                trainingData.append([herbId, diseaseId, float(rating)])
        
        if bTest:
            return testData
        else:
            return trainingData

    @staticmethod
    def loadHerbList(filepath):
        herbList = []
        with open(filepath) as f:
            for line in f:
                herbList.append(line.strip().split()[0])
        return herbList




