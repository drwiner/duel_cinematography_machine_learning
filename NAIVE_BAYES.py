import numpy as np
import random
from collections import namedtuple, defaultdict
import math
from clockdeco import clock

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


def split_duration(value):
	if value <= 2.5:
		return 0
	elif value < 5.7:
		return 1
	return 2


def split_interval_into_k(interval_value, k):
	intrvl = 1/k
	for i in range(1, k):
		if interval_value < intrvl*i:
			return i-1
	return k-1


def split_shot_into_k(shot_num, num_shots, k):
	shot_intrvl = num_shots / k
	for i in range(1,k):
		if shot_num < shot_intrvl * i:
			return i-1
	return k-1


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
		precision = correct_per_label[k] / (correct_per_label[k] + incorrect_per_label[k])
		precisions.append(precision)
		recall = correct_per_label[k] / (correct_per_label[k] + incorrect_per_guess[k])
		recalls.append(recall)
		fscore = (2 * recall * precision) / (recall + precision)
		fscores.append(fscore)
		# print("label:\t{}\tprecision:\t{}\trecall\t{}\tfscores\t{}".format(k, precision, recall, fscore))

	total_p = sum(correct_per_label) / (sum(correct_per_label) + sum(incorrect_per_label))
	total_r = sum(correct_per_label) / (sum(correct_per_label) + sum(incorrect_per_guess))
	total_f = (2 * total_r * total_p) / (total_r + total_p)
	print("total:\tprecision:\t{}\trecall\t{}\tfscores\t{}".format(total_p, total_r, total_f))

	return precisions, recalls, fscores, total_p, total_r, total_f


def run_naive_bayes(example_sets, labels, largest_index):

	# each 'i' is test
	total_per_fold = defaultdict(int)
	num_folds = len(example_sets)
	gamma = [20, 10, 6, 4]
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
			precisions, recalls, fscores, total_p, total_r, total_f = test_naive_bayes(example_sets[i], likes, rlikes, priors, labels, largest_index)
			total_per_fold[gam] += total_p

	for gam in gamma:
		print("gamma:\t{}\tacc:\t{}".format(gam, total_per_fold[gam] / num_folds))


def parse_featlist(file_name, label_name):
	examples = []

	with open(file_name, 'r') as fn:
		for line in fn:
			example_tuple = eval(line)
			feat_list = []


			feat_list.append(action_feats_dict[example_tuple[1]])

			if example_tuple[2]:
				feat_list.append(starts_feat)
			if example_tuple[3]:
				feat_list.append(finish_feat)

			for item in example_tuple[4]:
				if item in action_sh_e.keys():
					feat_list.append(action_sh_e[item])
			for item in example_tuple[5]:
				if item in action_nsh_e.keys():
					feat_list.append(action_nsh_e[item])
			for item in example_tuple[6]:
				if item in action_sh.keys():
					feat_list.append(action_sh[item])
			for item in example_tuple[7]:
				if item in action_nsh.keys():
					feat_list.append(action_nsh[item])

			feat_list.append(scene_feats[example_tuple[8]])

			# shot duration (short medium long)
			feat_list.append(duration_feats[split_duration(example_tuple[9])])
			# into-scene time (beg mid end)
			feat_list.append(into_scene_feats[split_interval_into_k(example_tuple[10], 3)])
			# shot number...
			shot_num = example_tuple[11]
			# total number of shots. Into scene shot (beg mid end)
			feat_list.append(shots_into_scene_feats[split_shot_into_k(shot_num, example_tuple[12], 3)])
			scale_label = scale_label_dict[example_tuple[13]]
			# for now, not used
			zpos = example_tuple[14]
			xpos = xpos_label_dict[example_tuple[15]]

			feat_list = list(set(feat_list))
			feat_list.sort()
			if label_name == "xpos":
				examples.append(LabeledEx(xpos, feat_list))
			elif label_name == "scale":
				examples.append(LabeledEx(scale_label, feat_list))

	return examples


def get_num_feats():
	"""
	FEATURES:
	"""
	with open("action_types.txt", 'r') as att:
		for j, line in enumerate(att):
			action_feats_dict[line.strip()] = j

	num_feats = 0
	# 1 : actions (39)
	num_feats += len(action_feats_dict.values())
	# 2-3 : starts, finishes (2)
	global starts_feat
	starts_feat = num_feats
	global finish_feat
	finish_feat = num_feats + 1
	num_feats += 2
	# 4-7 : actions (39)
	global action_sh_e
	action_sh_e = {val: i + num_feats for val, i in action_feats_dict.items()}
	num_feats += len(action_feats_dict.values())
	global action_nsh_e
	action_nsh_e = {val: i + num_feats for val, i in action_feats_dict.items()}
	num_feats += len(action_feats_dict.values())
	global action_sh
	action_sh = {val: i + num_feats for val, i in action_feats_dict.items()}
	num_feats += len(action_feats_dict.values())
	global action_nsh
	action_nsh = {val: i + num_feats for val, i in action_feats_dict.items()}
	num_feats += len(action_feats_dict.values())
	# 8 : scenes (30)

	with open("scenes.txt", 'r') as stt:
		for z, line in enumerate(stt):
			scene_feats[line.strip()] = z + num_feats

	num_feats += len(scene_feats)
	# 9 : duration of shot (short, med, long)
	global duration_feats
	duration_feats = [num_feats, num_feats+1, num_feats +2]
	num_feats += 3
	# 10 : into-scene time [0-1] (beg, mid end)
	global into_scene_feats
	into_scene_feats = [num_feats, num_feats + 1, num_feats + 2]
	num_feats += 3
	# 11 : shot number (into-scene proportation, beg, mid, end)
	global shots_into_scene_feats
	shots_into_scene_feats = [num_feats, num_feats + 1, num_feats + 2]
	num_feats += 3
	# 12 : total number shots in scene
	# 13 : scale (label 1)
	# 14 : zeta
	# 15 : position (label 2)
	return num_feats + 1


if __name__ == '__main__':
	test = "data/test.txt"
	training_whole = "data/all_train.txt"
	base_cvsplits = "data/CV_Splits/train_{}"

	lfi = get_num_feats()

	do_cv = 1

	if do_cv:
		training_examples_xpos = [parse_featlist(base_cvsplits.format(i), "xpos") for i in range(5)]
		training_examples_scale = [parse_featlist(base_cvsplits.format(i), "scale") for i in range(5)]

		run_naive_bayes(training_examples_scale, [0,1,2,3], lfi)
		run_naive_bayes(training_examples_xpos, [0,1,2,3,4], lfi)

	# whole_training_scale = parse_featlist(training_whole, "scale")
	# scale_labels = [0, 1, 2, 3]
	# whole_training_xpos = parse_featlist(training_whole, "xpos")
	# xpos_labels = [0, 1, 2, 3, 4]
	#
	# test_scale = parse_featlist(test, "scale")
	# test_xpos = parse_featlist(test, "xpos")
	#
	# likes, rlikes, priors = naive_bayes(whole_training_scale, scale_labels, 0.0001, lfi)
	# print("SCALE: whole training")
	# test_naive_bayes(whole_training_scale, likes, rlikes, priors, scale_labels, lfi)
	# print("SCALE: test")
	# test_naive_bayes(test_scale, likes, rlikes, priors, scale_labels, lfi)
	#
	# likes, rlikes, priors = naive_bayes(whole_training_xpos, xpos_labels, 0.0001, lfi)
	# print("XPOS: whole training")
	# test_naive_bayes(whole_training_xpos, likes, rlikes, priors, xpos_labels, lfi)
	# print("XPOS: test")
	# test_naive_bayes(test_xpos, likes, rlikes, priors, xpos_labels, lfi)
