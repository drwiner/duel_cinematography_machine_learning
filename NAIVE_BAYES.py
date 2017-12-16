import numpy as np
import random
from collections import namedtuple, defaultdict
import math
from clockdeco import clock
from ID3 import load_examples_from_file
from generate_features import get_num_feats

LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])

action_feats_dict = dict()
starts_feat = None
finish_feat = None
action_sh_e = dict()
action_nsh_e = dict()
action_sh = dict()
action_nsh = dict()
scene_feats = dict()
duration_feats = list()
into_scene_feats = list()
shots_into_scene_feats = list()

scale_label_dict = {'cu':0, 'waist':1, 'figure':2, 'wide':3}
xpos_label_dict = {'left': 0, 'center-left': 1, 'center': 2, 'full': 2, 'center-right': 3, 'right': 4}



def likelihood(count_feat_label, gamma, count_label, num_labels):
	return (count_feat_label + gamma) / (count_label + num_labels*gamma)


def assemble_feature_count_per_label(examples, num_labels):
	feat_freqs = [defaultdict(int) for i in range(num_labels)]
	label_freqs = [0 for i in range(num_labels)]
	for example in examples:
		for ef in example.feats:
			feat_freqs[example.label][ef] += 1
		label_freqs[example.label] += 1
	return feat_freqs, label_freqs


def assemble_likelihood_per_feat(gamma, feat_freqs, label_freqs, labels, lfi):
	"""
	p(x_i | C_k) for each i and k
	"""
	likelihoods = [[] for i in labels]
	rlikelihoods= [[] for i in labels]
	for i in range(lfi):
		for k in labels:
			p_xi_given_k = likelihood(feat_freqs[k][i], gamma, label_freqs[k], len(labels))

			likelihoods[k].append(math.log(p_xi_given_k))
			if p_xi_given_k != 1:
				rlikelihoods[k].append(math.log(1-p_xi_given_k))
			else:
				rlikelihoods[k].append(0)

	return likelihoods, rlikelihoods


def naive_bayes(examples, labels, gamma, lfi):
	print('learning naive bayes')
	ffreqs, lfreqs = assemble_feature_count_per_label(examples, len(labels))
	likes, rlikes = assemble_likelihood_per_feat(gamma, ffreqs, lfreqs, labels, lfi)
	priors = [math.log(lfreqs[k]/len(examples)) for k in range(len(labels))]

	return likes, rlikes, priors


# @clock
def logprob(prob_list, pminus_list, lfi, example):
	return sum(prob_list[f] if f in example.feats else pminus_list[f] for f in range(lfi))


def test_naive_bayes(examples, likes, rlikes, log_priors, labels, lfi):
	num_correct = 0
	correct_per_label = [0 for i in range(len(labels))]
	incorrect_per_label = [0 for i in range(len(labels))]
	incorrect_per_guess = [0 for i in range(len(labels))]
	print('testing naive bayes')
	for i, example in enumerate(examples):
		# print('Example: {}'.format(i))

		best_score = -100000
		best_label = -100000

		for k, label in enumerate(labels):
			p = logprob(likes[k], rlikes[k], lfi, example) + log_priors[k]
			if p > best_score:
				best_score = p
				best_label = k

		if best_label == -100000:
			raise ValueError("no best scores?")

		if best_label == example.label:
			num_correct += 1
			correct_per_label[best_label] += 1
		else:
			incorrect_per_guess[best_label] += 1
			incorrect_per_label[example.label] += 1

	precisions = []
	recalls = []
	fscores = []
	for k in labels:
		try:
			precision = correct_per_label[k] / (correct_per_label[k] + incorrect_per_label[k])
		except:
			precision = 0
		precisions.append(precision)
		try:
			recall = correct_per_label[k] / (correct_per_label[k] + incorrect_per_guess[k])
		except:
			recall = 0
		recalls.append(recall)
		try:
			fscore = (2 * recall * precision) / (recall + precision)
		except:
			fscore = 0
		fscores.append(fscore)
		# print("label:\t{}\tprecision:\t{}\trecall\t{}\tfscores\t{}".format(k, precision, recall, fscore))

	acc = num_correct / len(examples)
	# print("total acc:\t{}".format(acc))

	return precisions, recalls, fscores, acc


def run_naive_bayes(example_sets, labels, largest_index, gamma):

	# each 'i' is test
	total_per_fold = defaultdict(int)
	num_folds = len(example_sets)


	#2, 1.5, 1, .5]

	for i in range(num_folds):
		# each 'j' is training
		training = []
		for j in range(num_folds):
			if i == j:
				continue
			training.extend(example_sets[j])

		# print('fold:\t{}'.format(str(i)))
		for gam in gamma:
			# print('gamma=\t{}'.format(gam))
			likes, rlikes, priors = naive_bayes(training, labels, gam, largest_index)
			precisions, recalls, fscores, acc = test_naive_bayes(example_sets[i], likes, rlikes, priors, labels, largest_index)
			total_per_fold[gam] += acc

	for gam in gamma:
		print("gamma:\t{}\tacc:\t{}".format(gam, total_per_fold[gam] / num_folds))



if __name__ == '__main__':
	test = "data/test.{}"
	training_whole = "data/training.{}"
	base_cvsplits = "data/CV_Splits/train_{}"

	label_sets = {"scale": [0, 1, 2, 3], "xpos": [0, 1, 2, 3, 4]}
	lfi = get_num_feats()
	gamma = [10, 6, 4, 2, 1]
	do_cv = 0
	do_whole_test = 1

	if do_cv:

		for task in ["scale", "xpos"]:
			print(task)
			examples = load_examples_from_file(training_whole.format(task))
			random.shuffle(examples)
			num_per_fold = int(len(examples) / 5)
			splits = [examples[i:i + num_per_fold] for i in range(0, len(examples) - 1, num_per_fold)]
			run_naive_bayes(splits, label_sets[task], lfi, gamma)



	if do_whole_test:
		gamma = 10
		for task in ["scale", "xpos"]:
			print(task)
			whole_training = load_examples_from_file(training_whole.format(task))
			test_examples = load_examples_from_file(test.format(task))

			likes, rlikes, priors = naive_bayes(whole_training, label_sets[task], gamma, lfi)
			result = test_naive_bayes(whole_training, likes, rlikes, priors, label_sets[task], lfi)
			print(result[-1])
			result = test_naive_bayes(test_examples, likes, rlikes, priors, label_sets[task], lfi)
			print(result[-1])