# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pandas as pd
import numpy as np
import math
import random
import time
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from scipy import interp
from math import nan,isnan

import cudf
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRFC


def getfang(ls):
    sum1=0
    mean=np.mean(ls)
    for i in ls:
        sum1+=(i-mean)**2
    return math.sqrt(sum1/10)

def partition(ls, size):
    return [ls[i:i + size] for i in range(0, len(ls), size)]

def main(options):


    ##获取所含药材和疾病的数量

    #
    HD=pd.read_csv('n2vnew_1/H_D.txt',sep='\t',header=None)
    
    
    embedding=pd.read_csv('n2vnew_1/emb.txt',sep=' ',header=None,index_col=0)
    print(embedding)
    #embedding = embedding.iloc[:,0:]
    #print(embedding)

    ###向dataframe结构中插入一列，第一个参数为插入的位置，第二个位置为列的名称，第三个参数为要插入的值

    ####在负样本中随机抽取长度和正样本相同的负样本
    negative=pd.read_csv('n2vnew_1/negativehdlist.txt',sep='\t',header=None)
    randomlist=random.sample(range(0,len(negative)),len(HD))
        
    ##按照某一行进行取值
    negativelist=negative.iloc[randomlist]
    negativelist[2]=0
    ##print(negaticelist)
    
    
    ###shift+tab键将代码往左移
    RandomList = random.sample(range(0, len(HD)), len(HD))#随机长度，有重复
    print('len(RandomList)', len(RandomList))
    ###生成进行交叉验证的数据，整数位置为交叉验证的折数
    NewRandomList = partition(RandomList, math.ceil(len(RandomList) / options.fold_num))
    print('len(NewRandomList)', len(NewRandomList))
    NewRandomList = pd.DataFrame(NewRandomList)
    NewRandomList = NewRandomList.fillna(int(0))
    NewRandomList = NewRandomList.astype(int)

    Nindex = NewRandomList
    print('lenrandom:',len(Nindex))
    for i in range(len(Nindex)):
        kk = []
        for j in range(options.fold_num):
            if j !=i:
                kk.append(j)
        index=[]
        for j in range(0,len(kk)):
            index.append(np.array(Nindex)[kk[j]])
        index=np.hstack(index)
        DTIs_train= pd.DataFrame(np.array(HD)[index])
        DTIs_train.to_csv('./n2vnew_1/DrDiIs_train'+str(i)+'.csv', header=None,index=False)
        DTIs_test=pd.DataFrame(np.array(HD)[np.array(Nindex)[i]])
        DTIs_test.to_csv('./n2vnew_1/DrDiIs_test'+str(i)+'.csv', header=None,index=False)
        print(i)
    del index, DTIs_train, DTIs_test

    data_train_names=globals()
    data_test_names=globals()
    labels_train_names=globals()
    labels_test_names=globals()


    for i in range(options.fold_num):
        train_data = pd.read_csv('./n2vnew_1/DrDiIs_train'+str(i)+'.csv',header=None)
        train_data[2] = 1#对训练样本加上标签
        kk = []
        for j in range(options.fold_num):
            if j !=i:
                kk.append(j)
        index=[]
        for j in range(0,len(kk)):
            index.append(np.array(Nindex)[kk[j]])
        index=np.hstack(index)

        result = train_data.append(pd.DataFrame(np.array(negativelist)[index]))#将对应数量的负样本加入
        labels_train = result[2]#标签值
        #print(result)
        #print("attribute:")
        #print(result[0].values.tolist())
        #print("loc:")
        #print(embedding.loc[result[0].values.tolist()])
        #time.sleep(1)

        data_train_feature =np.multiply(embedding.loc[result[0].values.tolist()],embedding.loc[result[1].values.tolist()])
        print('data_train_feature')
        print(data_train_feature)

        #将每对样本的embding作为训练的特征

        data_train_names['data_train'+str(i)]=data_train_feature.values.tolist()
        labels_train_names['labels_train'+str(i)]=labels_train

        #print(len(labels_train))
        del labels_train, result, data_train_feature
        test_data = pd.read_csv('./n2vnew_1/DrDiIs_test'+str(i)+'.csv',header=None)
        test_data[2] =1
        result = test_data.append(pd.DataFrame(np.array(negativelist)[np.array(Nindex)[i]]))
        labels_test = result[2]


        data_test_feature = np.multiply(embedding.loc[result[0].values.tolist()],embedding.loc[result[1].values.tolist()])


        data_test_names['data_test'+str(i)]=data_test_feature.values.tolist()
        labels_test_names['labels_test'+str(i)]=labels_test

        del train_data, test_data, labels_test, result, data_test_feature
  
    
    data_train=[]
    data_test=[]
    labels_train=[]
    labels_test=[]

    for i in range(options.fold_num):
        data_train.append(data_train_names['data_train'+str(i)])
        data_test.append(data_test_names['data_test'+str(i)])
        labels_train.append(labels_train_names['labels_train'+str(i)])
        labels_test.append(labels_test_names['labels_test'+str(i)])
    del data_train_names,data_test_names,labels_train_names,labels_test_names

    print(type(data_train))
    print(str(options.fold_num)+"-CV")
    tprs=[]
    aucs=[]
    accuracys=[]
    precisions=[]
    recalls=[]
    f1scores=[]
    micro_f1s=[]
    macro_f1s=[]
    
    mean_fpr=np.linspace(0,1,1000)
    AllResult = []

    Wfile=open(r"./n2vnew_1/result/RF/result.txt","a")
    Wfile1=open(r"./n2vnew_1/result/RF/mean_fpr.txt","a")
    for i in range(options.fold_num):

        X_train,X_test = data_train[i],data_test[i]
        #print("train type:",type(X_train))
        #print("train data",X_train)
        X_train=cp.asarray(X_train,dtype=cp.float32)
        X_test=cp.asarray(X_test,dtype=cp.float32)
        Y_train,Y_test = cp.asarray(labels_train[i],dtype=cp.int32),np.array(labels_test[i])
   
        #Y_train,Y_test=np.array(labels_train[i]),np.array(labels_test[i])
        #print("train type:",type(X_train))
        #print("train type:",type(Y_train))
        #print("train data",X_train)

        #best_RandomF = RandomForestClassifier(n_estimators=options.tree_number)
        best_RandomF=cuRFC(n_estimators=options.tree_number)
        best_RandomF.fit(X_train,cp.asarray(Y_train))
        y_score0=best_RandomF.predict(X_test)
        print('score0')
        print(y_score0)
        y_score_RandomF = best_RandomF.predict_proba(X_test)

        print(type(y_score_RandomF))
        y_score_RandomF=cp.asnumpy(y_score_RandomF)
        print(type(y_score_RandomF))
        y_score0=cp.asnumpy(y_score0)


        data0={'real':Y_test,'predict':y_score_RandomF[:,1]}
        dataframe0=pd.DataFrame(data0)
        dataframe0.to_csv(r'./n2vnew_1/result/RF/real_predict'+str(i)+'.csv',columns=['real','predict'],header=None,index=False)  
        
        fpr,tpr,thresholds=roc_curve(Y_test,y_score_RandomF[:,1])
        
        data={'fpr':fpr,'tpr':tpr,'thresholds':thresholds}
        dataframe=pd.DataFrame(data)
        dataframe.to_csv(r'./n2vnew_1/result/RF/fpr_tpr'+str(i)+'.csv',columns=['fpr','tpr','thresholds'],header=None,index=False)
        
        pre,re,thresholds=precision_recall_curve(Y_test,y_score_RandomF[:,1])
        #print()
        data1={'precision':pre,'recall':re}
        dataframe1=pd.DataFrame(data1)
        dataframe1.to_csv(r'./n2vnew_1/result/RF/precision_recall'+str(i)+'.csv',columns=['precision','recall'],header=None,index=False)
        
    
        tprs.append(interp(mean_fpr,fpr,tpr))
        tprs[-1][0]=0.0
        #auc
        roc_auc=auc(fpr,tpr)
        aucs.append(roc_auc)
        print('ROC fold %d(AUC=%0.4f)'% (i,roc_auc))
    
        accuracy = accuracy_score(Y_test,y_score0)
        precision = precision_score(Y_test,y_score0)
        recall = recall_score(Y_test,y_score0,average='binary')
        f1score = f1_score(Y_test,y_score0,average='binary')
        micro_f1=f1_score(Y_test,y_score0,average='micro')
        macro_f1=f1_score(Y_test,y_score0,average='macro')
    
        accuracys.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1scores.append(f1score)
        micro_f1s.append(micro_f1)
        macro_f1s.append(macro_f1)
    
     
    
        print('fold %d auc=%0.4f| accuracy=%0.4f | precision=%.4f  |recall=%0.4f  |f1score=%0.4f  |micro_f1=%0.4f  |macro_f1=%0.4f'%(i,roc_auc,accuracy,precision,recall,f1score,micro_f1,macro_f1),file=Wfile)
     
    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1.0
    print("mean_tpr:",file=Wfile1)
    print(mean_tpr,file=Wfile1)
    print("mean_fpr:",file=Wfile1)
    print(mean_fpr,file=Wfile1)
    mean_auc=auc(mean_fpr,mean_tpr)
    print('Mean ROC(AUC=%0.4f)'%(mean_auc))
    print('Mean ROC (AUC=%0.4f)'% (mean_auc),file=Wfile)
    acc_std=getfang(accuracys)
    pre_std=getfang(precisions)
    recall_std=getfang(recalls)
    f1_std=getfang(f1scores)
    micf1_std=getfang(micro_f1s)
    macf1_std=getfang(macro_f1s)
    print('mean accuracy=%0.4f+%0.4f | precision=%.4f +%0.4f |recall=%0.4f +%0.4f |f1score=%0.4f +%0.4f |micro_f1=%0.4f +%0.4f |macro_f1=%0.4f +%0.4f'
                      %(np.mean(accuracys,axis=0),acc_std,np.mean(precisions,axis=0),pre_std,np.mean(recalls,axis=0),recall_std,np.mean(f1scores,axis=0),f1_std,np.mean(micro_f1s,axis=0),micf1_std,np.mean(macro_f1s,axis=0),macf1_std),file=Wfile)
    Wfile.close()
    Wfile1.close()
        

if __name__ == '__main__':
    import optparse
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = optparse.OptionParser(usage=main.__doc__)
    parser.add_option("-d", "--dataset", action='store',
                      dest='dataset', default=1, type='int',
                      help=('The dataset of cross-validation '
                            ))
    parser.add_option('-f', '--fold num', action='store',
                      dest='fold_num', default=5, type='int',
                      help=('The fold number of cross-validation '
                            '(default: 10)'))

    parser.add_option('-n', '--tree number', action='store',
                      dest='tree_number', default=999, type='int',
                      help=('The number of tree of RandomForestClassifier '
                            '(default: 999)'))

    options, args = parser.parse_args()
    print(options)
    sys.exit(main(options))