import os
import numpy as np
import argparse
import tensorflow as tf
from Model import network
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

number = 508


def load_txt(file_path):
	data = np.loadtxt(file_path, dtype = str, delimiter = ",")
	feature = data[:,1:133].astype(np.float32)
	target = data[:,134:].astype(np.float32)
	return (feature, target)


def train_test_split(feature, target):
	index = np.arange(feature.shape[0])
	np.random.shuffle(index)
	test_index = [index[0]]
	train_index = np.delete(index, test_index)[1:21]
	verif_index = np.delete(index, test_index)[21:31]
	test_fe, test_tg = feature[test_index], target[test_index]
	train_fe, train_tg = feature[train_index], target[train_index]
	verif_fe, verif_tg = feature[verif_index], target[verif_index]
	return (test_fe, test_tg), (train_fe, train_tg),(verif_fe, verif_tg), test_index,train_index,verif_index

class DataSet(object):
	def __init__(self, num):
		self.data_num = num
		self.index = np.arange(self.data_num)
		self.check, self.start, self.end = 0, 0, 0
		self.interation = True

	def next_batch(self, batch_size):
		self.start = self.end
		if self.check == 0:
			np.random.shuffle(self.index)
		if self.start + batch_size >= self.data_num:
			self.interation = False
			return self.index[self.start:]
		else:
			self.end = self.start + batch_size
			return self.index[self.start:self.end]

class Evaluation(object):
	def __init__(self):
		self.total_target = []
		self.total_pre = []

	def update(self, target, prediction):
		for i in range(len(target)):
			self.total_target.append(target[i])
			self.total_pre.append(prediction[i])

	def r2(self):
		return r2_score(self.total_target, self.total_pre, multioutput='uniform_average')
	def mae(self):
		return mean_absolute_error(self.total_target, self.total_pre)
	
	def rmse(self):
		return mean_squared_error(self.total_target, self.total_pre, squared=False)

def train(sess, model, feature, target, batch_size):
	Data = DataSet(feature.shape[0])
	train_ev = Evaluation()
	while Data.interation:
		ind = Data.next_batch(batch_size)
		_, loss, train_pre = sess.run(
						[model.opt, model.loss, model.pre],
						feed_dict = {
								model.x:feature[ind],
								model.y:target[ind]
							}
					)

		train_ev.update(target[ind], train_pre)
	return [loss, train_ev.r2(), train_ev.mae(), train_ev.rmse()]

def verif(sess, model, feature, target, batch_size):
	Data = DataSet(feature.shape[0])
	verif_ev = Evaluation()
	while Data.interation:
		ind =  Data.next_batch(batch_size)
		loss, verif_pre = sess.run(
					[model.loss, model.pre],
					feed_dict = {
							model.x:feature[ind],
							model.y:target[ind]
						}
				)
		verif_ev.update(target[ind], verif_pre)
	return [loss, verif_ev.r2(), verif_ev.mae(), verif_ev.rmse()]

sess_global= 0
def main(args):
	best_data = 1.0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)
	(feature, target) = load_txt(args.file_path)

	(test_fe, test_tg), (verif_fe, verif_tg), (train_fe, train_tg), test_index,train_index,verif_index = train_test_split(
																															feature = feature, 
																															target = target
																														)
	net = network(
			axis = train_fe.shape[1], 
			lr = args.lr
		)
	saver = tf.train.Saver(max_to_keep = 1)
	np.savetxt(str(number)+"_test_index.csv", test_index)
	np.savetxt(str(number)+"_train_index.csv", train_index,  delimiter = ",")
	np.savetxt(str(number)+"_verif_index.csv", verif_index,  delimiter = ",")
	global sess_global
	with tf.Session(
					config=tf.ConfigProto(gpu_options = gpu_options)
				) as sess:
		print("session assigned..........")
		saver.restore(sess, "check_point/model.ckpt")
		
		for epoch in range(args.epochs):
			train_loss = train(
								sess = sess, 
								model = net, 
								feature = train_fe, 
								target = train_tg, 
								batch_size = args.batch_size
							)
			verif_loss = verif(
								sess = sess, 
								model = net, 
								feature = verif_fe, 
								target = verif_tg, 
								batch_size = args.batch_size
							)
			
			print(
				"epoch {}, train_r2={:.4f}, train_mae={:.4f}, train_rmse={:.4f}, verif_r2={:.4f}, verif_mae={:.4f}, verif_rmse={:.4f}"
					.format(epoch, train_loss[1], train_loss[2], train_loss[3], verif_loss[1], verif_loss[2], verif_loss[3])
			)

			if verif_loss[2] < best_data:
				best_data = verif_loss[2]
				print("*"*10 + " save model " + "*"*10)
				saver.save(sess, "check_point/model.ckpt")
			
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--file_path", type = str, default = str(number)+"_magpie_target.csv", help = "file path")
	parser.add_argument("--percent", type = float, default = 0.3, help = "....")
	parser.add_argument("--batch_size", type = int, default = 64, help = "....")
	parser.add_argument("--epochs", type = int, default = 1000, help = "....")
	parser.add_argument("--lr", type = float, default = 1e-3, help = "....")
	return parser.parse_args()

if __name__ == "__main__":
	main(parse_args())
