"""
David R. Winer
drwiner@cs.utah.edu
ID3 Algorithm
"""

from collections import namedtuple
import unicodedata
import math
from collections import Counter

#### FEATURES ####

def action_type(example):
	return example[1]

def starts(example):
	return example[2]


def finishes(example):
	return example[3]


#### EQUATIONS ####


def gain(samples, feature, values):
	total_gain = 0
	for value in values:
		sub_samples = [lbl for lbl in samples if feature(lbl) == value]
		total_gain += (len(sub_samples) / len(samples)) * entropy(sub_samples)

	return entropy(samples) - total_gain


def entropy(samples):
	total = len(samples)
	if total == 0:
		return 0
	sum_pos = sum(1 for lbl in samples if lbl.label is True)
	sum_neg = sum(1 for lbl in samples if lbl.label is False)
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
	return -p_pos * math.log2(p_pos) - p_neg * math.log2(p_neg)


#### ID3 and DECISION TREE ####

"""
Nodes are nested dictionaries with 3 values, a feature, a 0 : {}, and a 1 : {}
"""

def num_samples_with_label(samples, target_label):
	return sum(1 for lbl in samples if lbl.label == target_label)


def all_samples_target(samples, target_label):
	return len(samples) == num_samples_with_label(samples, target_label)


def best_feature(samples, features, values):
	best = (-1, None)
	for feature in features:
		x = gain(samples, feature, values[feature])
		if x > best[0]:
			best = (x, feature)
	return best[1]


def most_labeled(samples, target_labels):
	best = (-1, None)
	for tlabel in target_labels:
		num_with_label = num_samples_with_label(samples, tlabel)
		if num_with_label > best[0]:
			best = (num_with_label, tlabel)
	return best[1]


# ID3 without option to limit depth
def ID3(samples, features, values):
	# if all samples have positive label

	for tlabel in [True, False]:
		if all_samples_target(samples, tlabel):
			return tlabel

	if len(features) == 0:
		# return most common value of remaining samples
		return most_labeled(samples, [True, False])

	# Pick Best Feature
	if len(features) == 1:
		best_f = list(features)[0]
	else:
		best_f = best_feature(samples, features, values)

	children = {'feature': best_f}
	for best_f_value in values[best_f]:
		sub_samples = {lbl for lbl in samples if best_f(lbl) == best_f_value}

		if len(sub_samples) == 0:
			children[best_f_value] = most_labeled(samples, [True, False])
		else:
			children[best_f_value] = ID3(sub_samples, set(features) - {best_f}, values)

	return children


# ID3 with option to limit depth
def ID3_depth(samples, features, values, depth):

	for tlabel in [True, False]:
		if all_samples_target(samples, tlabel):
			return tlabel

	if len(features) == 0:
		# return most common value of remaining samples
		return most_labeled(samples, [True, False])

	# Pick Best Feature
	if len(features) == 1:
		best_f = list(features)[0]
	else:
		best_f = best_feature(samples, features, values)

	children = {'feature': best_f}

	for best_f_value in values[best_f]:
		sub_samples = {lbl for lbl in samples if best_f(lbl) == best_f_value}
		if len(sub_samples) == 0 or depth == 1:
			children[best_f_value] = most_labeled(sub_samples, [True, False])
		else:
			children[best_f_value] = ID3_depth(sub_samples, set(features) - {best_f}, values, depth-1)

	return children


def use_tree(tree, item):

	# base case, the tree is a value
	if type(tree) is bool:
		return tree

	# otherwise, recursively evaluate item with features
	result = tree['feature'](item)
	return use_tree(tree[result], item)


def extract_name(split_line):
	name_part = split_line[1:]
	first_name = name_part[0]

	if len(name_part) > 2:
		# check if last position is title
		last_name = name_part[-1]
		middle_name = name_part[1]
	else:
		middle_name = None
		last_name = name_part[-1]

	return (first_name, middle_name, last_name)


# Gets data from files and removes titles and punctuation
def get_data(data_file_name):
	data = []
	with open(data_file_name, 'rb') as training_data_file:
		for line in training_data_file:
			decoded_line = strip_accents(line.decode())
			new_line = decoded_line.replace(';', ' ')
			new_line = new_line.replace(' Jr.', ' ')
			new_line = new_line.replace(' Sr.', ' ')
			new_line = new_line.replace(' Dr.', ' ')
			new_line = new_line.replace(',', ' ').replace('.', ' ')
			last_namelength = len(new_line.split()[-1])

			# get length of name, but then remove them so that we calculate the first letter of the last name as
			new_line = new_line.replace(' von ', ' ').replace(' van der ', ' ').replace(' van ', ' ').replace(' de ', ' ')

			sp = new_line.split()
			if sp[0] == '+':
				label_value = True
			else:
				label_value = False

			name_parts = extract_name(sp)
			lb = Label(label_value, name_parts[0], name_parts[1], name_parts[2], last_namelength)
			data.append(lb)
	return data


# Part 1 of the implementation hw
def implementationHW():
	# Set of Features, initially

	VALUES = {firstname_longer_lastname: [True, False],
	          has_middle_name: [True, False],
	          same_first_and_last_letter: [True, False],
	          firstnameletter_same_lastnameletter: [True, False],
	          firstnameletter_is_vowel: [True, False],
	          even_length_lastname: [True, False]}

	training_data = get_data('data//updated_train.txt')

	dtree = ID3(training_data, list(VALUES.keys()), VALUES)
	num_correct = 0
	for item in training_data:
		outcome = use_tree(dtree, item)
		if outcome and item.label:
			num_correct += 1
		elif not outcome and not item.label:
			num_correct += 1

	print('Train Acc: {}'.format(num_correct / len(training_data)))

	test_data = get_data('data//updated_test.txt')

	# dtree = ID3(test_data, list(VALUES.keys()), VALUES)

	num_correct = 0
	for item in test_data:
		outcome = use_tree(dtree, item)
		if outcome and item.label:
			num_correct += 1
		elif not outcome and not item.label:
			num_correct += 1

	print('Test Acc: {}'.format(num_correct / len(test_data)))
	# print(dtree)


def standard_dev(samples, mean):
	x = sum((sample-mean)**2 for sample in samples) / (len(samples)-1)
	return math.sqrt(x)


def limit_depth():
	VALUES = {firstname_longer_lastname: [True, False],
	          has_middle_name: [True, False],
	          same_first_and_last_letter: [True, False],
	          firstnameletter_same_lastnameletter: [True, False],
	          firstnameletter_is_vowel: [True, False],
	          even_length_lastname: [True, False]}

	file_locations = ['data//Updated_CVSplits//updated_training0' + str(i) + '.txt' for i in range(4)]
	depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]
	for d in depths:

		total = 0
		avgs = []
		for i in range(4):
			training = []
			for j in range(4):
				if i == j:
					continue
				training.extend(get_data(file_locations[j]))

			test = get_data(file_locations[i])

			dtree = ID3_depth(training, list(VALUES.keys()), VALUES, d)

			acc = 0
			for item in test:
				result = use_tree(dtree, item)
				if result and item.label:
					acc += 1
				elif not result and not item.label:
					acc += 1

			total += acc / len(test)
			avgs.append(acc/len(test))

		avg = total / 4

		print('depth: {} \t average: {} \t std_dev: {}'.format(d, avg,standard_dev(avgs, avg)))


	# SECOND HALF of question: use the best depth (in this case, 6) to retrain on entire training and measure performacne on training and test data.

	training_data = get_data('data//updated_train.txt')

	dtree = ID3_depth(training_data, list(VALUES.keys()), VALUES, 6)
	num_correct = 0
	for item in training_data:
		outcome = use_tree(dtree, item)
		if outcome and item.label:
			num_correct += 1
		elif not outcome and not item.label:
			num_correct += 1

	print('Train Acc: {}'.format(num_correct / len(training_data)))

	test_data = get_data('data//updated_test.txt')

	# dtree = ID3(test_data, list(VALUES.keys()), VALUES)

	num_correct = 0
	for item in test_data:
		outcome = use_tree(dtree, item)
		if outcome and item.label:
			num_correct += 1
		elif not outcome and not item.label:
			num_correct += 1

	print('Test Acc: {}'.format(num_correct / len(test_data)))


def superiorTech(lb):
	return lb.tech

def enviro(lb):
	return lb.enviro

def likesHuman(lb):
	return lb.likeshuman

def lightYears(lb):
	return lb.lightyears


alienLabel = namedtuple('AlienLabel', ['label', 'tech', 'enviro', 'likeshuman', 'lightyears'])

def alien_test():
	alienValues = {superiorTech: [True, False],
	          enviro: [True, False],
	          likesHuman: ['like', 'hate', 'do not care'],
	          lightYears: [1,2,3,4]}

	alienData = [alienLabel(True, False, True, 'do not care', 1),
	             alienLabel(False, False, True, 'like', 3),
	             alienLabel(True, False, False, 'do not care', 4),
	             alienLabel(True, True, True, 'like', 3),
	             alienLabel(False, True, False, 'like', 1),
	             alienLabel(True, False, True, 'do not care', 2),
	             alienLabel(False, False, False, 'hate', 4),
	             alienLabel(True, False, True, 'do not care', 3),
	             alienLabel(False, True, False, 'like', 4)]

	print(entropy(alienData))

	for feature in alienValues.keys():
		print('feature: {} \t IG: {}'.format(feature.__name__, gain(alienData, feature, alienValues[feature])))
		# print(feature, gain(alienData, feature, alienValues[feature]))

	dtree = ID3_depth(alienData, list(alienValues.keys()), alienValues, 1)
	print(dtree)
	#
	test_data = [alienLabel(False, True, True, 'like', 2),
	             alienLabel(False, False, False, 'hate', 3),
	             alienLabel(True, True, True, 'like', 4)]

	acc = 0
	for item in test_data:
		result = use_tree(dtree, item)
		if result and item.label:
			acc += 1
		elif not result and not item.label:
			acc += 1
	print(1 - (acc / len(test_data)))

	for feature in alienValues.keys():
		print(feature.__name__, majority_error_gain(alienData, feature, alienValues[feature]))

	dtree = ID3_depth(alienData, list(alienValues.keys()), alienValues, 8)
	print(dtree)


def majority_error_gain(samples, feature, values):
	total_gain = 0
	for value in values:
		sub_samples = [lbl for lbl in samples if feature(lbl) == value]
		total_gain += (len(sub_samples) / len(samples)) * majority_error(sub_samples)

	return majority_error(samples) - total_gain


def majority_error(samples):
	if len(samples) == 0:
		return 1
	p_true = num_samples_with_label(samples, True) / len(samples)
	p_false = num_samples_with_label(samples, False) / len(samples)
	if p_true > p_false:
		return 1 - p_true
	else:
		return 1 - p_false


if __name__ == '__main__':
	print('David R. Winer\n')
	# alien_test()
	print('Train and Test, part 1')
	implementationHW()

	print('\nLimit Depth')
	limit_depth()