import numpy as np
import tensorflow as tf
from keras import backend as K

smooth = 
stride_num = 
num_s = 
n_class = 

def Logistic(x,theta,sigma):
	return  1/(1+tf.exp(-theta*(x-sigma)))

#Sparse prediction
def Harmony_loss_S(y_true, y_pred):
	y_pred = tf.nn.softmax(y_pred)
	stride = np.linspace(start=1, stop=0,num =stride_num,dtype=np.float)
	pr = [] #precision
	re = [] #recall
	pr_sub = [] #statistics on precision
	re_sub = [] #statistics on recall
	for i in range(stride_num):
		pred_new = Logistic(y_pred, num_s, stride[i])
		for j in range(n_class):
			intersection_sub = K.sum((pred_new*y_true)[:,j])
			denominator_precision_sub = K.sum(pred_new[:,j]) + smooth
			denominator_recall_sub = K.sum(y_true[:,j]) + smooth
			recall_sub = intersection_sub/denominator_recall_sub
			precision_sub = intersection_sub/denominator_precision_sub
			pr_sub.append(precision_sub)
			re_sub.append(recall_sub)
		recall = tf.reduce_mean(re_sub,axis = 0)
		precision = tf.reduce_mean(pr_sub,axis = 0)
		pr.append(precision)
		re.append(recall)  
	pr_sort =  tf.sort(pr, direction='DESCENDING', axis = 0)
	re_sort =  tf.sort(re, direction='ASCENDING', axis = 0)
	AUC = re_sort[0] * pr_sort[0]
	for j in range(stride_num-1): 
		AUC_j = (re_sort[j+1]-re_sort[j]) * pr_sort[j+1]
		AUC = AUC + AUC_j
	loss = 1 - AUC
	return loss 


#Dense prediction
def PR_loss(y_true, y_pred ):
	y_pred = tf.nn.softmax(y_pred)
	stride = np.linspace(start=1, stop=0,num =stride_num,dtype=np.float)
	pr = []  #precision
	re = []  #recall
	for i in range(stride_num):
		pred_new = Logistic(y_pred, num_s, stride[i])
		recall = tf.reduce_sum(pred_new*y_true,axis=[])/(tf.reduce_sum(y_true,axis=[]) + smooth )      
		precision = tf.reduce_sum(pred_new*y_true,axis=[])/(tf.reduce_sum(pred_new,axis=[])+ smooth )
		pr.append(precision)
		re.append(recall)  
	pr_sort =  tf.sort(pr, direction='DESCENDING', axis = 0)
	re_sort =  tf.sort(re, direction='ASCENDING', axis = 0)
	AUC = re_sort[0] * pr_sort[0]
	for j in range(stride_num-1): 
		AUC_j = (re_sort[j+1]-re_sort[j]) * pr_sort[j+1]
		AUC = AUC + AUC_j
	loss = 1-tf.reduce_mean(AUC)
	return loss