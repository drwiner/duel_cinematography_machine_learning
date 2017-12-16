#!/usr/bin/env python3

#batch script for liblinear
import subprocess


def train():

	for label_type in ["scale", "xpos"]:
		print(label_type)
		model = 'model.' + label_type
		train = "training." + label_type
		command = 'train -s 0 ' + train + ' ' + model
		print(command)
		subprocess.call(command, shell=True)

def predict():
	print('predicting')

	for label_type in ["scale", "xpos"]:
		print(label_type)
		test_features = 'test.' + label_type
		model = 'model.' + label_type
		predict = 'predict.' + label_type
		acc = 'accuracy_' + label_type + '.txt'
		command = 'predict ' + test_features + ' ' + model + ' ' + predict + ' > ' + acc
		print(command)
		subprocess.call(command, shell=True)


if __name__ == "__main__":
	train()
	predict()