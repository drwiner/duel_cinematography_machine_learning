"""
ID3 with large number of binary features
"""
from clockdeco import clock
import math
import random
from collections import namedtuple, defaultdict
from generate_features import get_num_feats
from collections import Counter

LabeledEx = namedtuple('LabeledEx', ['label', 'feats'])

BINARY_LIST = [0,1]
Y_VALS = [1, -1]


"""
features are integers
"""
# @clock
def gain(examples, feature, example_entropy):
	true_in = [lbl for lbl in examples if feature in lbl.feats]
	false_in = [lbl for lbl in examples if feature not in lbl.feats]

	total_gain = (len(true_in) / len(examples)) * entropy(true_in)
	total_gain += (len(false_in) / len(examples)) * entropy(false_in)

	return example_entropy - total_gain


# @clock
def entropy(examples):
	total = len(examples)
	if total == 0:
		return 0
	sum_pos = sum(1 for lbl in examples if lbl.label == Y_VALS[0])
	sum_neg = sum(1 for lbl in examples if lbl.label == Y_VALS[1])
	p_pos = sum_pos / total
	p_neg = sum_neg / total
	if p_pos == 0:
		if p_neg == 0:
			return 0
		return - p_neg * math.log2(p_neg)
	if p_neg == 0:
		if p_pos == 0:
			return 0
		return -p_pos * math.log2(p_pos)
	entropy_value = -p_pos * math.log2(p_pos) - p_neg * math.log2(p_neg)

	return entropy_value


def gain_multiclass(examples, feature, example_entropy, classes):
	true_in = [lbl for lbl in examples if feature in lbl.feats]
	false_in = [lbl for lbl in examples if feature not in lbl.feats]

	total_gain = (len(true_in) / len(examples)) * entropy_multiclass(true_in, classes)
	total_gain += (len(false_in) / len(examples)) * entropy_multiclass(false_in, classes)

	return example_entropy - total_gain


def plogp(examples, value, total):
	if total == 0:
		raise ValueError("total is zero?")

	p = sum(1 for lbl in examples if lbl.label == value)/total

	if p == 0:
		return 0

	return p * math.log(p)


# @clock
def entropy_multiclass(examples, classes):
	total = len(examples)
	if total == 0:
		return 0

	return -1 * sum(plogp(examples, value, total) for value in classes)


# @clock
def gain_with_target(examples, feature, example_entropy, target_label):
	true_in = [lbl for lbl in examples if feature in lbl.feats]
	false_in = [lbl for lbl in examples if feature not in lbl.feats]

	total_gain = (len(true_in) / len(examples)) * entropy_with_target(true_in, target_label)
	total_gain += (len(false_in) / len(examples)) * entropy_with_target(false_in, target_label)

	return example_entropy - total_gain


def entropy_with_target(examples, target_label):
	total = len(examples)
	if total == 0:
		return 0
	sum_pos = sum(1 for lbl in examples if lbl.label == target_label)

	p_pos = sum_pos / total
	p_neg = (len(examples) - sum_pos) / total
	if p_pos == 0:
		if p_neg == 0:
			return 0
		return - p_neg * math.log2(p_neg)
	if p_neg == 0:
		if p_pos == 0:
			return 0
		return -p_pos * math.log2(p_pos)
	entropy_value = -p_pos * math.log2(p_pos) - p_neg * math.log2(p_neg)

	return entropy_value


#### ID3 and DECISION TREE ####

"""
Nodes are nested dictionaries with 3 values, a feature, a 0 : {}, and a 1 : {}
"""

def num_samples_with_label(samples, target_label):
	return sum(1 for lbl in samples if lbl.label == target_label)


def all_samples_target(samples, target_label):
	return len(samples) == num_samples_with_label(samples, target_label)


# @clock
def best_feature(examples, feature_list):
	best = [-1, None]
	entropy_examples = entropy(examples)
	for feature in feature_list:
		x = gain(examples, feature, entropy_examples)
		if x > best[0]:
			best[0] = x
			best[1] = feature
	return best[1]


# @clock
def best_feature_with_target(examples, feature_list, target_label):
	best = [-1, None]
	entropy_examples = entropy_with_target(examples, target_label)
	for feature in feature_list:
		x = gain_with_target(examples, feature, entropy_examples, target_label)
		if x > best[0]:
			best[0] = x
			best[1] = feature
	return best[1]


def best_feature_multiclass(examples, feature_list, labels):
	best = [-1, None]
	entropy_examples = entropy_multiclass(examples, labels)
	for feature in feature_list:
		x = gain_multiclass(examples, feature, entropy_examples, labels)
		if x > best[0]:
			best[0] = x
			best[1] = feature
	return best[1]


# @clock
def most_labeled(samples, target_labels):
	best = [-1, None]
	for tlabel in target_labels:
		num_with_label = num_samples_with_label(samples, tlabel)
		if num_with_label > best[0]:
			best[0] = num_with_label
			best[1] = tlabel
	return best[1]


def most_labeled_with_target(samples, target_label):
	num_with_label = num_samples_with_label(samples, target_label)
	if num_with_label > len(samples)/2:
		return target_label
	else:
		return -1


# ID3 with option to limit depth
# @clock
def ID3_depth_with_target(examples, features, label_of_interest, depth):

	# check if all have label of interest:
	if all_samples_target(examples, label_of_interest):
		return label_of_interest
	if num_samples_with_label(examples, label_of_interest) == 0:
		return -1

	if len(features) == 0:
		# return most common value of remaining examples
		return most_labeled_with_target(examples, label_of_interest)

	# Pick Best Feature (integer
	if len(features) == 1:
		best_f = list(features)[0]
	else:
		best_f = best_feature_with_target(examples, features,label_of_interest)

	children = {'feature': best_f}

	sub_samples_0 = [exmpl for exmpl in examples if best_f not in exmpl.feats]
	if len(sub_samples_0) == 0 or depth == 1:
		children[0] = most_labeled_with_target(sub_samples_0, label_of_interest)
	else:
		children[0] = ID3_depth_with_target(sub_samples_0, set(features) - {best_f}, label_of_interest, depth-1)

	sub_samples_1 = [exmpl for exmpl in examples if best_f in exmpl.feats]

	if len(sub_samples_1) == 0 or depth == 1:
		children[1] = most_labeled_with_target(sub_samples_1, label_of_interest)
	else:
		children[1] = ID3_depth_with_target(sub_samples_1, set(features) - {best_f}, label_of_interest, depth-1)

	return children


def ID3_depth_multiclass(examples, features, labels, depth):

	# check if all have label of interest:
	for label in labels:
		if num_samples_with_label(examples, label) == len(examples):
			return label

	if len(features) == 0:
		# return most common value of remaining examples
		return most_labeled(examples, labels)

	# Pick Best Feature (integer
	if len(features) == 1:
		best_f = list(features)[0]
	else:
		best_f = best_feature_multiclass(examples, features, labels)

	children = {'feature': best_f}

	sub_samples_0 = [exmpl for exmpl in examples if best_f not in exmpl.feats]
	if len(sub_samples_0) == 0 or depth == 1:
		children[0] = most_labeled(sub_samples_0, labels)
	else:
		children[0] = ID3_depth_multiclass(sub_samples_0, set(features) - {best_f}, labels, depth-1)

	sub_samples_1 = [exmpl for exmpl in examples if best_f in exmpl.feats]

	if len(sub_samples_1) == 0 or depth == 1:
		children[1] = most_labeled(sub_samples_1, labels)
	else:
		children[1] = ID3_depth_multiclass(sub_samples_1, set(features) - {best_f}, labels, depth-1)

	return children


def use_tree(tree, item):

	# base case, the tree is a value
	if type(tree) is bool or type(tree) is int:
		return tree

	# otherwise, recursively evaluate item with features
	result = tree['feature'] in item.feats
	return use_tree(tree[result], item)


def get_x(examples, x):
	z = list(examples)
	random.shuffle(z)
	return z[:x]


def get_x_trees(examples, num_examples_per_tree, num_trees, labels, d):
	dtrees = []
	for i in range(num_trees):
		print(i)
		sub_set = get_x(examples, num_examples_per_tree)
		relevant_feats = []
		for example in sub_set:
			relevant_feats.extend(example.feats)
		relevant_feats = list(set(relevant_feats))
		dtree = ID3_depth_multiclass(sub_set, relevant_feats, labels, d)
		dtrees.append(dtree)
	return dtrees


def load_examples_from_file(file_name):
	examples = []
	with open(file_name, 'r') as train:
		for line in train:
			lin_sp = line.split()
			ex = LabeledEx(label=int(lin_sp[0]), feats=[int(f.split(":")[0]) for f in lin_sp[1:]])
			examples.append(ex)
	return examples


def test_tree_multiclass(test_examples, tree, labels):
	num_correct = 0
	num_correct_per_label = [0 for i in labels]
	num_guessed_per_label = [0 for i in labels]
	num_missed_per_label = [0 for i in labels]

	for tex in test_examples:
		guess = use_tree(tree, tex)
		if guess == tex.label:
			num_correct += 1
			num_correct_per_label[guess] += 1
		else:
			num_missed_per_label[tex.label] += 1
		num_guessed_per_label[guess] += 1

	# print("Overall ACC:\t{}".format(num_correct/len(test_examples)))
	# calculate precision and fscore per label
	for label in labels:
		try:
			p = num_correct_per_label[label] / num_guessed_per_label[label]
		except:
			p = 0
		try:
			r = num_correct_per_label[label] / (num_correct_per_label[label] + num_missed_per_label[label])
		except:
			r = 0
		try:
			f = (2 * p * r) / (p + r)
		except:
			f = 0
		print("\tlabel:\t{}\tprecision:\t{}\trecall\t{}\tfscores\t{}".format(label, p, r, f))
	return num_correct/len(test_examples)


def test_tree_target(test_examples, tree, target_label):
	num_correct = 0
	for tex in test_examples:
		guess = use_tree(tree, tex)
		if guess == target_label and tex.label == target_label:
			num_correct += 1
		elif guess != target_label and tex.label != target_label:
			num_correct += 1

	print("{}\tACC:\t{}".format(target_label, str(num_correct/len(test_examples))))
	return num_correct/len(test_examples)


def cross_validation_multiclass_depth(example_sets, features, labels, depths):
	acc_dict = defaultdict(list)
	for i in range(len(example_sets)):
		training = []
		for j in range(len(example_sets)):
			if i == j:
				continue
			training.extend(example_sets[j])

			for d in depths:
				print(i, d)
				tree = ID3_depth_multiclass(training, features, labels, d)
				acc = test_tree_multiclass(example_sets[i], tree, labels)
				acc_dict[d].append(acc)

	for d in depths:
		avg_acc = sum(acc_dict[d]) / len(acc_dict[d])
		print("cv\td=\t{}\tacc=\t{}".format(d, avg_acc))


def cross_validation_with_target_depth(example_sets, features, label, depths):
	acc_dict = defaultdict(list)
	for i in range(len(example_sets)):
		training = []
		for j in range(len(example_sets)):
			if i == j:
				continue
			training.extend(example_sets[j])

			for d in depths:
				print(i, d)
				tree = ID3_depth_with_target(training, features, label, d)
				acc = test_tree_target(example_sets[i], tree, label)
				acc_dict[d].append(acc)

	for d in depths:
		avg_acc = sum(acc_dict[d]) / len(acc_dict[d])
		print("cv\td=\t{}\tacc=\t{}".format(d, avg_acc))


def record_stumps(tasks, features, label_dict, d):

	for task in tasks:
		print(task)
		examples = load_examples_from_file(train_file.format(task))
		# test_examples = load_examples_from_file(test_file.format(task))
		feats = []
		# slice_size = len(features)/(len(features)/d)
		shuffled_feats = features[:]
		random.shuffle(shuffled_feats)
		feat_slices = [shuffled_feats[i:i+d] for i in range(0, len(features), d)]
		# random.shuffle(feat_slices)
		for feat_list in feat_slices:
			label_per_feats = []
			for label in label_dict[task]:
				stump = ID3_depth_with_target(examples, feat_list, label, d)
				if stump[0] == -1 and stump[1] == -1:
					continue

				# print(str(stump))
				label_per_feats.append(stump)
			feats.append((feat_list, label_per_feats))

		with open ("task={}_depth={}.txt".format(task, d), "w") as file_name:
			for i, (feat_ids, feat) in enumerate(feats):
				for j, label_per_feat in enumerate(feat):
					file_name.write("({}, {}, ".format(feat_ids, j))
					file_name.write(str(label_per_feat))
					file_name.write(", ")
					file_name.write(str(test_tree_target(examples, label_per_feat, j)))
					file_name.write(")\n")


def run_bagged_forest_prediction_multiclass(dtrees, examples, test_examples):

	num_correct = 0
	print("accuracy for bagged trees: training")
	for i, example in enumerate(examples):
		decision = []
		for dtree in dtrees:
			decision.append(use_tree(dtree, example))
		count_decisions = Counter(decision)
		item = count_decisions.most_common(1)[0][0]
		if item == example.label:
			num_correct += 1

	print(num_correct / len(examples))

	num_correct = 0
	print("accuracy for bagged trees: test")
	for example in test_examples:
		decision = []
		for dtree in dtrees:
			decision.append(use_tree(dtree, example))
		count_decisions = Counter(decision)
		item = count_decisions.most_common(1)[0][0]
		if item == example.label:
			num_correct += 1

	print(num_correct / len(test_examples))


def run_bagged_forest_prediction_for_output(task, d, dtrees, label_sets, examples, test_examples):
	training_data = []
	for example in examples:
		decision = []
		for dtree in dtrees:
			decision.append(use_tree(dtree, example))
		training_data.append(decision)

	with open("bagged_forest_training_{}_{}.txt".format(task, d), 'w') as file_name:
		for ex, train in zip(examples, training_data):
			file_name.write(str(ex.label))
			for i, decision in enumerate(train):
				file_name.write(" " + str((i+1)*len(label_sets[task]) + decision)+ ":1")
			file_name.write("\n")

	test_data = []
	for example in test_examples:
		decision = []
		for dtree in dtrees:
			decision.append(use_tree(dtree, example))
		test_data.append(decision)

	with open("bagged_forest_test_{}_{}.txt".format(task, d), 'w') as file_name:
		for ex, train in zip(test_examples, test_data):
			file_name.write(str(ex.label))
			for i, decision in enumerate(train):
				file_name.write(" " + str((i + 1) * len(label_sets[task]) + decision) + ":1")
			file_name.write("\n")

if __name__ == '__main__':
	# read input

	train_file = "data/training.{}"
	test_file = "data/test.{}"

	# test_scale = "data/test.scale"
	# test_xpos = "data/test.xpos"
	# training_scale = "data/training.scale"
	# training_xpos = "data/training.xpos"

	do_record_stumps = 0
	do_cv_multiclass = 0
	do_cv_binaryclass = 0
	do_test_TRAIN_tree = 0
	do_test_tree = 0
	do_create_forest = 0
	do_forest_avg_prediction = 0
	do_forest_output = 1

	# // largest index
	lfi = get_num_feats()
	label_sets = {"scale": [0, 1, 2, 3], "xpos": [0, 1, 2, 3, 4]}

	multiclass_depth = {"scale": 27, "xpos": 12}

	# cross validation:
	depths = [d for d in range(3, 30, 5)]
	fine_grain_depths = {"scale": [25,26,27,28,29], "xpos": [10,11,12,13,14]}

	if do_record_stumps:
		for d in range(1,5):
			record_stumps(["scale", "xpos"], list(range(1, lfi+1)), label_sets, d)

	# cross validation for mult-class
	if do_cv_multiclass:
		for task in ["scale", "xpos"]:
			examples = load_examples_from_file(train_file.format(task))
			num_per_fold = int(len(examples) / 5)
			splits = [examples[i:i+num_per_fold] for i in range(0,len(examples)-1, num_per_fold)]
			cross_validation_multiclass_depth(splits, list(range(1, lfi+1)), label_sets[task], fine_grain_depths[task])

	# cross validation for binary target
	if do_cv_binaryclass:
		for task in ["scale", "xpos"]:
			examples = load_examples_from_file(train_file.format(task))
			num_per_fold = int(len(examples) / 5)
			splits = [examples[i:i+num_per_fold] for i in range(0,len(examples)-1, num_per_fold)]

			for label in label_sets[task]:
				cross_validation_with_target_depth(splits, list(range(1, lfi+1)), label, depths)

	if do_test_TRAIN_tree:

		best_depth_dict = {"xpos": {0: 3, 1: 3, 2: 28, 3: 3, 4: 3},
		                   "scale": {0: 28, 1: 8, 2: 3, 3: 13}}

		for task in ["scale", "xpos"]:
			examples = load_examples_from_file(train_file.format(task))
			test_examples = load_examples_from_file(test_file.format(task))

			for label in label_sets[task]:
				target_tree = ID3_depth_with_target(examples, list(range(1, lfi + 1)), label_of_interest=label,
				                                    depth=best_depth_dict[task][label])
				acc = test_tree_target(examples, target_tree, label)
				print("{}\t{}\t{}".format(task, label, acc))

			multiclass_tree = ID3_depth_multiclass(examples, list(range(1, lfi + 1)), labels=label_sets[task],
			                                       depth=multiclass_depth[task])

			acc = test_tree_multiclass(examples, multiclass_tree, label_sets[task])
			print("{}\t{}".format(task, acc))

	# create decision tree for each label for each task
	if do_test_tree:

		best_depth_dict = {"xpos": {0: 3, 1:3, 2:28, 3:3, 4:3},
		                   "scale": {0:28, 1:8, 2:3, 3:13}}

		for task in ["scale", "xpos"]:
			examples = load_examples_from_file(train_file.format(task))
			test_examples = load_examples_from_file(test_file.format(task))

			for label in label_sets[task]:
				target_tree = ID3_depth_with_target(examples, list(range(1, lfi+1)), label_of_interest=label, depth=best_depth_dict[task][label])
				acc = test_tree_target(test_examples, target_tree, label)
				print("{}\t{}\t{}".format(task, label, acc))

			multiclass_tree = ID3_depth_multiclass(examples, list(range(1, lfi+1)), labels=label_sets[task], depth=multiclass_depth[task])

			acc = test_tree_multiclass(test_examples, multiclass_tree, label_sets[task])
			print("{}\t{}".format(task, acc))


	if do_create_forest:
		# multiclass forest
		for d in range(5,16,3):
			for task in ["scale", "xpos"]:
				examples = load_examples_from_file(train_file.format(task))
				trees = get_x_trees(examples, 100, 1000, label_sets[task], d)

				with open("forest_{}_{}.txt".format(task, d), 'w') as dtree_file:
					for dtree in trees:
						dtree_file.write(str(dtree))
						dtree_file.write('\n')

	if do_forest_avg_prediction:
		example_dict = dict()
		test_dict = dict()

		for task in ["scale", "xpos"]:
			example_dict[task] = load_examples_from_file(train_file.format(task))
			test_dict[task] = load_examples_from_file(test_file.format(task))

		for d in range(5,16,3):
			print(d)
			for task in ["scale", "xpos"]:
				print(task)
				dtrees = []
				file_name = "forest_{}_{}.txt".format(task, d)
				with open(file_name, 'r') as dtree_file:
					for line in dtree_file:
						dtrees.append(eval(line))

				run_bagged_forest_prediction_multiclass(dtrees, example_dict[task], test_dict[task])

	if do_forest_output:
		example_dict = dict()
		test_dict = dict()

		for task in ["scale", "xpos"]:
			example_dict[task] = load_examples_from_file(train_file.format(task))
			test_dict[task] = load_examples_from_file(test_file.format(task))

		for d in range(5, 16, 3):
			print(d)
			for task in ["scale", "xpos"]:
				print(task)
				dtrees = []
				file_name = "forest_{}_{}.txt".format(task, d)
				with open(file_name, 'r') as dtree_file:
					for line in dtree_file:
						dtrees.append(eval(line))

				run_bagged_forest_prediction_for_output(task, d, dtrees, label_sets, example_dict[task], test_dict[task])