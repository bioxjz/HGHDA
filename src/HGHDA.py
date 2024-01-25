#coding:utf8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from base.herbRecommender import herbRecommender
from scipy.sparse import coo_matrix,hstack
#import tensorflow as tf
import numpy as np
from math import sqrt
from time import strftime,localtime,time
import os
import random
os.environ["CUDA_VISIBLE_0EVICES"]="2"


class HGHDA(herbRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(HGHDA, self).__init__(conf,trainingSet,testSet,fold)

    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        i=0
        for pair in self.data.trainingData:
            # symmetric matrix
            if int(pair[2])!=0:
                row += [self.data.herb[pair[0]]]
                col += [self.data.disease[pair[1]]]
                entries += [1]
                i+=1
        print('i======i',i)
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_herbs,self.num_diseases),dtype=np.float32)
        return u_i_adj

    def buildhcAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.hcassociation:
            # symmetric matrix
            #x=random.randint(0,1008)
            #y=random.randint(0,1192)
            #row += [x]
            #col += [y]
            row += [self.data.herb[pair[0]]]
            col += [self.data.compound[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_herbs,self.num_compounds),dtype=np.float32)
        return u_i_adj

    def buildcpAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.cpassociation:
            # symmetric matrix
            row += [self.data.compound[pair[0]]]
            col += [self.data.protein[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_compounds,self.num_proteins),dtype=np.float32)
        return u_i_adj
    def buildpdAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.pdassociation:
            # symmetric matrix
            #x=random.randint(0,7257)
            #y=random.randint(0,11070)
            #row += [x]
            #col += [y]
            row += [self.data.protein[pair[0]]]
            col += [self.data.disease[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_proteins,self.num_diseases),dtype=np.float32)
        return u_i_adj
    def buildJointAdjacency(self):
        indices = [[self.data.herb[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData]
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_herbs, self.num_diseases])
        return norm_adj

    def initModel(self):
        super(HGHDA, self).initModel()
        #Build adjacency matrix
        A = self.buildAdjacencyMatrix()
        #A=A.dot(A.transpose().dot(A))
        cp=self.buildcpAdjacencyMatrix()
        pd=self.buildpdAdjacencyMatrix()
        hc=self.buildhcAdjacencyMatrix()
        #print('hc-------------type-------------------')
        
        #print(type(hc))
        #print('len hlist',len(self.data.Hlist))
        #print('len clist',len(self.data.Clist))
        #print('len plist',len(self.data.Plist))
        #print('len dlist',len(self.data.Dlist))
   
        #hp=hc.dot(cp)
        #cd=cp.dot(pd)


#####仅使用关联的
        H_c = hc.transpose()
        D_hc_v = H_c.sum(axis=1).reshape(1, -1)
        D_hc_e = H_c.sum(axis=0).reshape(1, -1)
        temp1 = (H_c.multiply(1.0 / D_hc_e)).transpose()
        temp2 = (H_c.transpose().multiply(1.0 / D_hc_v)).transpose()
        edge = temp1
        node = temp2
        A_e = edge.tocoo()
        A_n = node.tocoo()
        edge_indices = np.mat([A_e.row, A_e.col]).transpose()
        node_indices = np.mat([A_n.row, A_n.col]).transpose()
        H_e = tf.SparseTensor(edge_indices, A_e.data.astype(np.float32), A_e.shape)
        H_n = tf.SparseTensor(node_indices, A_n.data.astype(np.float32), A_n.shape)

        P_d = pd
        D_P_v = P_d.sum(axis=1).reshape(1, -1)
        D_P_e = P_d.sum(axis=0).reshape(1, -1)
        temp1 = (P_d.multiply(1.0 / D_P_e)).transpose()
        temp2 = (P_d.transpose().multiply(1.0 / D_P_v)).transpose()
        pd_edge=temp1
        pd_node=temp2
        A_pde=pd_edge.tocoo()
        A_pdn=pd_node.tocoo()
        pdedge_indices = np.mat([A_pde.row, A_pde.col]).transpose()
        pdnode_indices = np.mat([A_pdn.row, A_pdn.col]).transpose()
        P_e = tf.SparseTensor(pdedge_indices, A_pde.data.astype(np.float32), A_pde.shape)
        P_n = tf.SparseTensor(pdnode_indices, A_pdn.data.astype(np.float32), A_pdn.shape)



#####使用相似度矩阵的
#        self.compound_embeddings=coo_matrix(self.compound_embeddings,dtype=np.float32)
#        self.target_embeddings=coo_matrix(self.target_embeddings,dtype=np.float32)
#        print(type(self.compound_embeddings))
#        H_c = hc 
#        temp1 = H_c.dot(self.compound_embeddings)
#        temp2 = temp1.dot(H_c.transpose())
#        temp2.tocoo()
#        D_hc=temp2.sum(axis=1).reshape(1,-1)
#        print(type(D_hc))
#        A_h = temp2.multiply(1.0/D_hc)
#        A_h = A_h.tocoo()
#        indices = np.mat([A_h.row, A_h.col]).transpose()
#        H_c = tf.SparseTensor(indices, A_h.data.astype(np.float32), A_h.shape)

#        P_d = pd.transpose()
#        temp1 = P_d.dot(self.target_embeddings)
#        temp2 = temp1.dot(P_d.transpose())
#        temp2.tocoo()
#        D_pd=temp2.sum(axis=1).reshape(1,-1)
#        A_pd = temp2.multiply(1.0/D_pd)
#        A_pd = A_pd.tocoo()
#        indices = np.mat([A_pd.row, A_pd.col]).transpose()
#        P_d = tf.SparseTensor(indices, A_pd.data.astype(np.float32), A_pd.shape)


###使用特征向量的
#        self.compound_embeddings=coo_matrix(self.compound_embeddings,dtype=np.float32)
#        self.target_embeddings=coo_matrix(self.target_embeddings,dtype=np.float32)
#        print(type(self.compound_embeddings))
#        H_c = hc 
#        temp1 = H_c.dot(self.compound_embeddings)
#        temp2 = temp1.dot(temp1.transpose())
#        temp2.tocoo()
#        D_hc=temp2.sum(axis=1).reshape(1,-1)
#        print(type(D_hc))
#        A_h = temp2.multiply(1.0/D_hc)
#        A_h = A_h.tocoo()
#        indices = np.mat([A_h.row, A_h.col]).transpose()
#        H_c = tf.SparseTensor(indices, A_h.data.astype(np.float32), A_h.shape)

#        P_d = pd.transpose()
#        temp1 = P_d.dot(self.target_embeddings)
#        temp2 = temp1.dot(temp1.transpose())
#        temp2.tocoo()
#        D_pd=temp2.sum(axis=1).reshape(1,-1)
#        A_pd = temp2.multiply(1.0/D_pd)
#        A_pd = A_pd.tocoo()
#        indices = np.mat([A_pd.row, A_pd.col]).transpose()
#        P_d = tf.SparseTensor(indices, A_pd.data.astype(np.float32), A_pd.shape)




        #Build incidence matrix
        #H_u = hstack([A,A.dot(A.transpose().dot(A))])
#        H_u = A
#        D_u_v = H_u.sum(axis=1).reshape(1,-1)
#        D_u_e = H_u.sum(axis=0).reshape(1,-1)
#        temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()
#        temp2 = temp1.transpose()
#        A_u = temp1.multiply(1.0/D_u_e).dot(temp2)
#        A_u = A_u.tocoo()
#        indices = np.mat([A_u.row, A_u.col]).transpose()
#        H_u = tf.SparseTensor(indices, A_u.data.astype(np.float32), A_u.shape)

#        H_i = A.transpose()
#        D_i_v = H_i.sum(axis=1).reshape(1,-1)
#        D_i_e = H_i.sum(axis=0).reshape(1,-1)
#        temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()
#        temp2 = temp1.transpose()
#        A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
#        A_i = A_i.tocoo()
#        indices = np.mat([A_i.row, A_i.col]).transpose()
#        H_i = tf.SparseTensor(indices, A_i.data.astype(np.float32), A_i.shape)

        #Build network
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer =tf.keras.initializers.glorot_normal()
        self.n_layer = 5
        self.weights={}
        for i in range(self.n_layer):
            self.weights['layer_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_%d' % (i + 1))
            self.weights['layer_1_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_1_%d' % (i + 1))
            self.weights['layer_2_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='JU_2_%d' % (i + 1))
            self.weights['layer_att_%d' %(i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='layer_bias_%d' %(i+1))
        for i in range(2):
            self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                             name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]),
                                                                  name='g_W_b_%d_1' % (i + 1))

        def self_gating(em,channel):
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating%d' % channel])+self.weights['gating_bias%d' %channel]))

        compound_embeddings = self_gating(self.compound_embeddings,1)

        all_compound_embeddings = [compound_embeddings]

        protein_embeddings = self.protein_embeddings
        all_protein_embeddings = [protein_embeddings]
        all_hc_embeddings=[]
        all_pd_embeddings=[]

        
        dense_H_e=tf.sparse_tensor_to_dense(H_e,validate_indices=False)
        dense_H_n=tf.sparse_tensor_to_dense(H_n,validate_indices=False)
        dense_P_e=tf.sparse_tensor_to_dense(P_e,validate_indices=False)
        dense_P_n=tf.sparse_tensor_to_dense(P_n,validate_indices=False)
        for i in range(self.n_layer):


            new_hc_edge=tf.matmul(dense_H_e,compound_embeddings,a_is_sparse = True)
            new_compound_embeddings = tf.matmul(dense_H_n,new_hc_edge,a_is_sparse = True)
            new_pd_edge=tf.matmul(dense_P_e,protein_embeddings,a_is_sparse = True)
            new_protein_embeddings = tf.matmul(dense_P_n,new_pd_edge,a_is_sparse = True)

            compound_embeddings =tf.nn.leaky_relu(tf.matmul(new_compound_embeddings,self.weights['layer_%d' %(i+1)])+ compound_embeddings)

            protein_embeddings = tf.nn.leaky_relu(tf.matmul(new_protein_embeddings,self.weights['layer_1_%d' %(i+1)])+ protein_embeddings)

            compound_embeddings = tf.math.l2_normalize(compound_embeddings,axis=1)
            protein_embeddings = tf.math.l2_normalize(protein_embeddings,axis=1)
            new_hc_edge=tf.math.l2_normalize(new_hc_edge,axis=1)
            new_pd_edge=tf.math.l2_normalize(new_pd_edge,axis=1)


            all_compound_embeddings+=[compound_embeddings]
            all_protein_embeddings+=[protein_embeddings]
            all_hc_embeddings+=[new_hc_edge]
            all_pd_embeddings+=[new_pd_edge]


        compound_embeddings = tf.reduce_sum(all_compound_embeddings,axis=0)
        protein_embeddings = tf.reduce_sum(all_protein_embeddings, axis=0)
        hc_edge=tf.reduce_sum(all_hc_embeddings,axis=0)
        pd_edge=tf.reduce_sum(all_pd_embeddings,axis=0)


        self.neg_idx = tf.placeholder(tf.float32, name="neg_holder")

        self.neg_disease_embedding = tf.convert_to_tensor(self.neg_idx,dtype=tf.float32)
        #self.neg_disease_embedding = tf.nn.embedding_lookup(tf.convert_to_tensor(A.toarray(),dtype=tf.float32), self.u_idx)

        self.final_iembedding = protein_embeddings
        self.final_uembedding = compound_embeddings
        self.final_hcedge=hc_edge
        self.final_pdedge=pd_edge

        self.u_embedding = tf.nn.embedding_lookup(self.final_hcedge, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_pdedge, self.v_idx)
        #self.v_embedding = self.final_pdedge

    def trainModel(self):

	#quanjujiaocha
        #sigmoid_output=tf.sigmoid(tf.matmul(self.u_embedding, self.v_embedding, transpose_b=True))
        #y=tf.reduce_sum(tf.multiply(self.neg_disease_embedding,tf.log(sigmoid_output))+tf.multiply((1-self.neg_disease_embedding),tf.log(1-sigmoid_output)),1)

	#jiaochashang
        sigmoid_output=tf.sigmoid(tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding),1))
        #print("loss compute")
        #print(sigmoid_output)
        #print(self.neg_disease_embedding)
        y=tf.reduce_sum(tf.multiply(self.neg_disease_embedding,tf.log(sigmoid_output))+tf.multiply((1-self.neg_disease_embedding),tf.log(1-sigmoid_output)),0)


        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001*tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (tf.nn.l2_loss(self.final_hcedge) + tf.nn.l2_loss(self.final_pdedge))

        loss = -tf.reduce_sum(y)+reg_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                herb_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.u_idx: herb_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.isTraining:1})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)
        self.u,self.i,self.weight=self.sess.run([self.final_hcedge,self.final_pdedge,self.weights],feed_dict={self.isTraining:0})
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        np.savetxt('./results/herbedgeDHCN_herb_embedding'+currentTime+'.txt',self.u)
        np.savetxt('./results/diseaseedgeDHCN_disease_embedding'+currentTime+'.txt',self.i)
    def predictForRanking(self):
        print('hghdapredict----------------------------------------------------------------------------')
        return self.u.dot(self.i.transpose())
