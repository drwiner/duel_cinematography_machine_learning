from ID3 import use_tree, load_examples_from_file
import math

def initialize_weights(num_examples):
	return [1/num_examples for i in range(num_examples)]


def calc_error_target(stump, weights, examples, target_label):
	error = 0
	right_or_wrong = []
	for i, ex in enumerate(examples):
		guess = use_tree(stump, ex)
		if guess != ex.label and guess != -1:
			error += weights[i]
			right_or_wrong.append(0)
		elif guess == -1 and ex.label == target_label:
			error += weights[i]
			right_or_wrong.append(0)
		else:
			right_or_wrong.append(1)

	return error, right_or_wrong


# def calc_error(stump, stump_label, weights, examples):
# 	error = 0
# 	right_or_wrong = []
# 	for i, ex in enumerate(examples):
# 		guess = use_tree(stump, ex)
#
# 		if guess != ex.label:
# 			error += weights[i]
# 			right_or_wrong.append(0)
# 		else:
# 			right_or_wrong.append(1)
#
# 	return error, right_or_wrong


def get_best_classifier_target(stumps, weights, examples, target_label):
	best = [10000, None, None]
	for stump in stumps:
		e, is_right_list = calc_error_target(stump, weights, examples, target_label)
		if e < best[0]:
			best[0] = e
			best[1] = stump
			best[2] = is_right_list

	if best[0] >= (1 / 2):
		return best[0], False, False
		# raise ValueError("the best classifier is no better than chance?")
	return tuple(best)


def get_best_classifier(stumps, weights, examples, k=2):
	best = [10000, None, None, None]
	for stump, stump_label in stumps:
		e, is_right_list = calc_error_target(stump, weights, examples, stump_label)
		if e < best[0]:
			best[0] = e
			best[1] = stump
			best[2] = stump_label
			best[3] = is_right_list
	if best[0] >= (1/k):
		# return False
		return best[0], False, False, False
		# raise ValueError("the best classifier is no better than chance?")
	return tuple(best)


def calc_alpha(error):
	return 0.5 * math.log((1-error) / error)


def calc_alpha_multiclass(error, k):
	return math.log((1-error) / error) + math.log(k-1)


def is_finished(ada_classifier, examples, num_rounds, target_label):
	"""
	ideas:
		no good classifiers left (all have error <= 1/2)
		too many rounds
		H is good enough
	"""
	x = use_classifier_target(ada_classifier, examples, target_label)
	if x == len(examples):
		return True
	if num_rounds > 10:
		return True
	return False


def is_finished_multiclass(ada_classifier, examples, num_rounds, labels):
	x = use_classifier_multiclass(ada_classifier, examples, labels)
	if x == len(examples):
		return True
	if num_rounds > 10:
		return True
	return False



def calc_new_weights(weights, error, right_or_wrong):
	new_weights = []

	for w, is_right in zip(weights, right_or_wrong):
		if is_right:
			term = (1 / (1 - error))
		else:
			term = (1 / error)
		w = (w / 2) * term
		new_weights.append(w)

	return new_weights


def update_weights(weights, error, right_or_wrong):

	# calculate weight updates
	new_weights = calc_new_weights(weights, error, right_or_wrong)

	# scaling factor
	total_weights_right = sum(w for w, is_right in zip(new_weights, right_or_wrong) if is_right)
	total_weights_wrong = sum(w for w, is_right in zip(new_weights, right_or_wrong) if not is_right)

	# if mircaculously they equal 1
	if total_weights_right + total_weights_wrong == 1:
		return new_weights

	# rescale to 1/2
	final_weights = []
	for w, is_right in zip(new_weights, right_or_wrong):
		if is_right:
			nw = w / (total_weights_right*2)
		else:
			nw = w / (total_weights_wrong*2)
		final_weights.append(nw)

	return final_weights


def use_classifier_target(H, examples, target_label):
	sign = 0
	num_correct = 0
	if len(H) == 0:
		return 0
	for ex in examples:
		for (alpha, h) in H:

			guess = use_tree(h, ex)
			if guess == target_label and ex.label == target_label:
				sign += alpha
			elif guess == -1 and ex.label != target_label:
				sign += alpha
			else:
				sign -= alpha


		if sign >= 0 and ex.label == target_label:
			num_correct += 1
		elif sign < 0 and ex.label != target_label:
			num_correct += 1

	return num_correct


def use_classifier_multiclass(H, examples, labels):
	sign = 0
	num_correct = 0
	for ex in examples:
		best = [-1000, None]
		for label in labels:

			for (alpha, h, h_label) in H:
				guess = use_tree(h, ex)
				if guess == h_label and ex.label == h_label:
					sign += alpha
				elif guess == -1 and ex.label != h_label:
					sign += alpha
				else:
					sign -= alpha

			if sign > best[0]:
				best[0] = sign
				best[1] = label

		if best[1] == ex.label:
			num_correct += 1

	return num_correct

def adaboost(examples, stumps, target_label):
	weights = initialize_weights(len(examples))
	H = []
	i = 0
	while not is_finished(H, examples, i, target_label):
		if i > 0:
			weights = update_weights(weights, e, is_right_list)
		e, h, is_right_list = get_best_classifier_target(stumps, weights, examples, target_label)
		if is_right_list is False:
			return H
		alpha = calc_alpha(e)
		H.append((alpha, h))
		i = i + 1
		# print(i)
	return H


def adaboost_multiclass(examples, stumps, labels):
	weights = initialize_weights(len(examples))
	H = []
	i = 0
	while not is_finished_multiclass(H, examples, i, labels):
		# print(i)
		if i > 0:
			weights = update_weights(weights, e, is_right_list)
		e, h, h_label, is_right_list = get_best_classifier(stumps, weights, examples, k=len(labels))
		if is_right_list is False:
			return H
		alpha = calc_alpha_multiclass(e, k=len(labels))
		H.append((alpha, h, h_label))
		i = i + 1
	return H

if __name__ == "__main__":
	"""
	1] initialize weights to 1 / m for each example
	2] calculate error for each h (decision stump)
	3] pick h w/ smallest erorr
	4] calculate alpha (voting power)
	5] decide if finished (how many rounds, is it good enough, no good classifiers left)
	6] update weights --> should sum to 1 (correct should sum to 1/2 and incorrect should sum to 1/2) i.e. scale
	7] GOTO 2
	"""

	do_binary = 1
	do_multiclass = 1


	train_file = "data/training.{}"
	test_file = "data/test.{}"

	label_sets = {"scale": [0, 1, 2, 3], "xpos": [0, 1, 2, 3, 4]}

	for task in ["scale", "xpos"]:
		training_examples = load_examples_from_file(train_file.format(task))
		test_examples = load_examples_from_file(test_file.format(task))
		if do_binary:
			for label in label_sets[task]:
				print("label:\t{}".format(label))
				# get relevant stumps
				stumps = []
				for d in range(1,5):
					with open("task={}_depth={}.txt".format(task,d)) as file_name:
						for line in file_name:
							line_tup = eval(line)
							if line_tup[1] == label:
								stumps.append(line_tup[2])
				H = adaboost(training_examples, stumps, label)
				num_correct = use_classifier_target(H, test_examples, label)
				print("{}\t{}\t{}".format(task, label, num_correct / len(test_examples)))

		if do_multiclass:
			stumps = []
			for d in range(1,5):
				with open("task={}_depth={}.txt".format(task,d)) as file_name:
					for line in file_name:
						st = eval(line)
						stumps.append((st[2], st[1]))
			H = adaboost_multiclass(training_examples, stumps, label_sets[task])
			num_correct = use_classifier_multiclass(H, test_examples, label_sets[task])
			print("multiclass:\t{}\t{}".format(task, num_correct / len(test_examples)))