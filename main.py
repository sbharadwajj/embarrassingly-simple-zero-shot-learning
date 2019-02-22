import numpy as np
import os
import scipy.io
from sklearn.metrics import classification_report,confusion_matrix
import argparse

parser = argparse.ArgumentParser(description='EsZSL')
parser.add_argument('--dataset', type=str, default='CUB',
					help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='xlsa17/data/',
					help='Name of the dataset')
parser.add_argument('--alpha', type=int, default=2,
					help='value of hyper-parameter')
parser.add_argument('--gamma', type=int, default=2,
					help='value of hyper-parameter')




class EsZSL():
	"""docstring for ClassName"""
	def __init__(self, args):
		res101 = scipy.io.loadmat(args.dataset_path+args.dataset+'/res101.mat')
		att_splits = scipy.io.loadmat(args.dataset_path+args.dataset+'/att_splits.mat')

		trainval_loc = 'trainval_loc'
		train_loc = 'train_loc'
		val_loc = 'val_loc'
		test_loc = 'test_unseen_loc'

		labels = res101['labels']
		self.labels_train = labels[np.squeeze(att_splits[train_loc]-1)]
		self.labels_val = labels[np.squeeze(att_splits[val_loc]-1)]
		self.labels_trainval = labels[np.squeeze(att_splits[trainval_loc]-1)]
		self.labels_test = labels[np.squeeze(att_splits[test_loc]-1)]

		self.train_labels_seen = np.unique(self.labels_train)
		self.val_labels_unseen = np.unique(self.labels_val)
		self.trainval_labels_seen = np.unique(self.labels_trainval)
		self.test_labels_unseen = np.unique(self.labels_test)		

		print("Number of overlapping classes between train and val:",len(set(self.train_labels_seen).intersection(set(self.val_labels_unseen))))
		print("Number of overlapping classes between trainval and test:",len(set(self.trainval_labels_seen).intersection(set(self.test_labels_unseen))))		

		i = 0
		for labels in self.train_labels_seen:
			self.labels_train[self.labels_train == labels] = i
			i = i+1
		j = 0
		for labels in self.val_labels_unseen:
			self.labels_val[self.labels_val == labels] = j
			j = j+1
		k = 0
		for labels in self.trainval_labels_seen:
			self.labels_trainval[self.labels_trainval == labels] = k
			k = k+1
		l = 0
		for labels in self.test_labels_unseen:
			self.labels_test[self.labels_test == labels] = l
			l = l+1

		X_features = res101['features']
		self.train_vec = X_features[:,np.squeeze(att_splits[train_loc]-1)]
		self.val_vec = X_features[:,np.squeeze(att_splits[val_loc]-1)]
		self.trainval_vec = X_features[:,np.squeeze(att_splits[trainval_loc]-1)]
		self.test_vec = X_features[:,np.squeeze(att_splits[test_loc]-1)]		

		def normalization(vec,mean,std):
			sol = vec - mean
			sol1 = sol/std
			return sol1		

		#Signature matrix
		signature = att_splits['att']
		self.train_sig = signature[:,(self.train_labels_seen)-1]
		self.val_sig = signature[:,(self.val_labels_unseen)-1]
		self.trainval_sig = signature[:,(self.trainval_labels_seen)-1]
		self.test_sig = signature[:,(self.test_labels_unseen)-1]		

		#params for train and val set
		m_train = self.labels_train.shape[0]
		z_train = len(self.train_labels_seen)	

		#params for trainval and test set
		m_trainval = self.labels_trainval.shape[0]
		z_trainval = len(self.trainval_labels_seen)	

		#ground truth for train and val set
		self.gt_train = 0*np.ones((m_train, z_train))
		self.gt_train[np.arange(m_train), np.squeeze(self.labels_train)] = 1		

		#grountruth for trainval and test set
		self.gt_trainval = 0*np.ones((m_trainval, z_trainval))
		self.gt_trainval[np.arange(m_trainval), np.squeeze(self.labels_trainval)] = 1	
	

	def find_hyperparams(self):
		#train set
		d_train = self.train_vec.shape[0]
		a_train = self.train_sig.shape[0]

		accu = 0.10
		alph1 = 4
		gamm1 = 1

		#Weights
		V = np.zeros((d_train,a_train))
		for alpha in range(-3, 4):
			for gamma in range(-3,4):
				#One line solution
				part_1 = np.linalg.pinv(np.matmul(self.train_vec, self.train_vec.transpose()) + (10**alpha)*np.eye(d_train))
				part_0 = np.matmul(np.matmul(self.train_vec,self.gt_train),self.train_sig.transpose())
				part_2 = np.linalg.pinv(np.matmul(self.train_sig, self.train_sig.transpose()) + (10**gamma)*np.eye(a_train))		

				V = np.matmul(np.matmul(part_1,part_0),part_2)
				#print(V)		

				#predictions
				outputs = np.matmul(np.matmul(self.val_vec.transpose(),V),self.val_sig)
				preds = np.array([np.argmax(output) for output in outputs])		

				#print(accuracy_score(labels_val,preds))
				cm = confusion_matrix(self.labels_val, preds)
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				avg = sum(cm.diagonal())/len(self.val_labels_unseen)

				if avg > accu:
					accu = avg
					alph1 = alpha
					gamm1 = gamma
					print(alph1, gamm1, avg)
		print("Alpha and gamma:",alph1, gamm1)
		return alpha, gamma

	def train(self,alpha,gamma):
		#trainval set
		d_trainval = self.trainval_vec.shape[0]
		a_trainval = self.trainval_sig.shape[0]
		W = np.zeros((d_trainval,a_trainval))
		part_1_test = np.linalg.pinv(np.matmul(self.trainval_vec, self.trainval_vec.transpose()) + (10**alpha)*np.eye(d_trainval))
		part_0_test = np.matmul(np.matmul(self.trainval_vec,self.gt_trainval),self.trainval_sig.transpose())
		part_2_test = np.linalg.pinv(np.matmul(self.trainval_sig, self.trainval_sig.transpose()) + (10**gamma)*np.eye(a_trainval))			
		W = np.matmul(np.matmul(part_1_test,part_0_test),part_2_test)
		return W
	def test(self,weights):
		#predictions
		outputs_1 = np.matmul(np.matmul(self.test_vec.transpose(),weights),self.test_sig)
		preds_1 = np.array([np.argmax(output) for output in outputs_1])			
		cm = confusion_matrix(self.labels_test, preds_1)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		avg = sum(cm.diagonal())/len(self.test_labels_unseen)
		print("The top 1% accuracy is:", avg*100)

if __name__ == "__main__":
	args = parser.parse_args()
	alpha = args.alpha
	gamma = args.gamma
	model = EsZSL(args=args)
	if not args.alpha and args.gamma:
		alpha, gamma = model.find_hyperparams()
	weights = model.train(alpha,gamma)
	model.test(weights)

