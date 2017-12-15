import numpy as np
import random
from collections import namedtuple, defaultdict
from clockdeco import clock
LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])
from generate_features import get_num_feats

# @clock
def svm(examples, weights, tradeoff, learn_rate, epochs):
	example_list = list(examples)
	initial_learning_rate = learn_rate
	bias = 0.0001

	for epoch in range(epochs):

		for t, example in enumerate(example_list):
			feats_with_bias_term = np.append(example.feats, [1.0])
			p = example.label * np.dot(np.transpose(weights), feats_with_bias_term)
			q = weights - learn_rate
			if p <= 1:
				weights = q + learn_rate * tradeoff * example.label * feats_with_bias_term
			else:
				weights = q

			bias = bias + learn_rate * example.label
			weights[-1] = bias
			learn_rate = initial_learning_rate / (1 + t)


		random.shuffle(example_list)
	# print("lr={}, tradeoff={}, acc={}\n".format(initial_learning_rate, tradeoff, num_errors / len(example_list)))
	# print('learn_rate\t{}\ttradeoff\t{}\tacc\t{}'.format(initial_learning_rate, tradeoff, str(num_errors/len(example_list))))
	return weights


def test_svm(examples, weights):
	num_errors = 0
	for t, example in enumerate(examples):
		feats_with_bias_term = np.append(example.feats, [1.0])
		p = example.label * np.dot(np.transpose(weights), feats_with_bias_term)
		if p > 1:
			num_errors += 1

	return 1 - (num_errors / len(examples))

	# print('TEST: acc\t{}'.format(str(1 - (num_errors / len(examples)))))


def parse(file_name, num_feats):
	examples = []
	with open(file_name, 'r') as fn:
		for line in fn:
			linsp = line.split()
			label = int(linsp[0])
			feats = [int(feat.split(":")[0]) for feat in linsp[1:]]
			feat_vec = np.zeros(num_feats)
			for f in feats:
				feat_vec[f] = 1.0
			examples.append(LabeledEx(label, feat_vec))

	return examples


if __name__ == '__main__':
	# read input

	test_scale = "data/test.scale"
	test_xpos = "data/test.xpos"
	training_scale = "data/training.scale"
	training_xpos = "data/training.xpos"

	# base_cvsplits = "data/CV_Splits/train_{}"

	lfi = get_num_feats()

	whole_training_scale = parse(training_scale, lfi)
	whole_training_xpos = parse(training_xpos, lfi)

	test_scale = parse(test_scale, lfi)
	test_xpos = parse(test_xpos, lfi)
	
	
	initial_weights = np.array([0.00001 for k in range(lfi+1)])
	# run_svms(whole_training_scale, len(whole_training_scale), lfi)
	w = svm(whole_training_scale, initial_weights, 10, 0.0001, 40)
	results = test_svm(test_scale, w)
	print('stop here')