# Run on Lab1-19 to access numpy module

# David R. Winer
# drwiner at cs.utah.edu

from collections import namedtuple, defaultdict
import numpy as np
from functools import partial

# LabeledEx = namedtuple('LabeledEx', ['label', 'feature_dict'])
LabeledEx = namedtuple('LabeledEx', ['label', 'feat_vec'])

#### HELPER METHODS ####

# vector is string of form <index:value, index:value,....>; indices are "int", values are "float"
def vector_to_dict(vector):
	item_dict = defaultdict(int)
	for item in vector:
		index, value = item.split(':')
		item_dict[int(index)] = float(value)
	return item_dict


def vector_to_list(vector):
	feat_dict = vector_to_dict(vector)
	local_max = max(feat_dict, key=int)
	precursor_list = [feat_dict[j] for j in range(local_max)]
	return precursor_list


def dot_weight_dict(w, d):
	dot_prod = 0
	for j, weight in enumerate(w):
		dot_prod += weight*d[j]
	return dot_prod


def dot_dict(d1, d2, n):
	dot_prod = 0
	for i in range(1,n+1):
		dot_prod += d1[i]*d2[i]
	return dot_prod


# extract labeled examples from data file
def get_data(data_file_name, largest_size=None):
	data = []
	with open(data_file_name, 'r') as training_data_file:
		for line in training_data_file:
			example = line.split()
			# arg1 is label, arg2 is dictionary (keys are indices, values are values)
			example_tup = LabeledEx(int(example[0]), vector_to_list(example[1:]))
			data.append(example_tup)

	# now, find the maximum size vector and pad all others
	if largest_size is None:
		n = get_longest_vec(data)
	else:
		n = largest_size

	training_data = []
	for example in data:
		precursor_list = example.feat_vec
		if len(example.feat_vec) < n:
			precursor_list.extend([0 for j in range(n-len(example.feat_vec))])
			if len(precursor_list) != n:
				raise ArithmeticError('bad arithmetic')
		training_data.append(LabeledEx(example[0], np.array(precursor_list)))

	return training_data


def get_longest_vec(examples):
	return max(len(example.feat_vec) for example in examples)


def get_max_index(examples):
	max_index = 0
	for ex in examples:
		feat_dict = ex.feature_dict
		local_max = max(feat_dict, key=int)
		if local_max > max_index:
			max_index = local_max
	return max_index


#### PERCEPTRON METHODS ####


def update_weights(w, lr, label, feat_vec):
	update_op = feat_vec*lr

	# mistake on positive
	if label == 1:
		new_w = w + update_op

	# mistake on negative
	else:
		new_w = w - update_op

	return new_w



def simple_perceptron_wrapper(examples, learning_rates, epochs):
	# learning_rates = [1, 0.1, 0.01]

	# examples are precompiled to have correct length
	num_feats = len(examples[0].feat_vec)

	# initialize weights
	weights = np.array([0.001 for j in range(num_feats)])

	bias = 0.001

	# run perceptron for each learning rate
	w_values = [simple_perceptron(examples, weights, bias, lr, epochs) for lr in learning_rates]

	return w_values


def simple_perceptron(examples, weights, bias, learning_rate, epochs, pc_test=None):
	num_errors = 0
	for epoch in range(epochs):

		for example in examples:
			y_prime = np.dot(np.transpose(weights), example.feat_vec) + bias

			if y_prime > 0:
				y_prime = 1
			else:
				y_prime = -1

			if y_prime != example.label:
				num_errors += 1
				# update weights + bias
				weights = update_weights(weights, learning_rate, example.label, example.feat_vec)
				bias = bias + learning_rate*example.label

		if pc_test is not None:
			print('epoch:{}\t{}'.format(epoch, pc_test(weight_vector=weights, bias=bias)))


	print('updates:\t{}'.format(num_errors))
	return weights, bias


def dynamic_perceptron_wrapper(examples, learning_rates, epochs):
	# examples are precompiled to have correct length
	num_feats = len(examples[0].feat_vec)

	# initialize weights
	weights = np.array([0.001 for j in range(num_feats)])

	bias = 0.001

	# run perceptron for each learning rate
	w_values = [dynamic_perceptron(examples, weights, bias, lr, epochs) for lr in learning_rates]

	return w_values


def dynamic_perceptron(examples, weights, bias, learning_rate, epochs, pc_test=None):
	num_errors = 0
	t = 0
	for epoch in range(epochs):

		for example in examples:
			y_prime = np.dot(np.transpose(weights), example.feat_vec) + bias

			if y_prime > 0:
				y_prime = 1
			else:
				y_prime = -1

			if y_prime != example.label:
				num_errors += 1
				# update weights + bias
				lr = learning_rate / (1 + t)
				weights = update_weights(weights, lr, example.label, example.feat_vec)
				bias = bias + lr*example.label

				t += 1

		if pc_test is not None:
			print('epoch:{}\t{}'.format(epoch, pc_test(weight_vector=weights, bias=bias)))


	print('updates:\t{}'.format(num_errors))
	return weights, bias


def margin_perceptron_wrapper(examples, learning_rates, epochs):
	# examples are precompiled to have correct length
	num_feats = len(examples[0].feat_vec)

	# initialize weights
	weights = np.array([0.001 for j in range(num_feats)])

	bias = 0.001

	# run perceptron for each learning rate
	w_values = [margin_perceptron(examples, weights, bias, lr1, lr2, epochs) for lr1 in learning_rates for lr2 in learning_rates]

	return w_values


def margin_perceptron(examples, weights, bias, learning_rate, margin, epochs, pc_test=None):
	# print('learning_rate:\t{}\nmargin:\t{}'.format(learning_rate, margin))
	num_errors = 0
	t = 0
	for epoch in range(epochs):

		for example in examples:
			y_prime = np.dot(np.transpose(weights), example.feat_vec) + bias

			if (y_prime * example.label) < margin:
				num_errors += 1
				# update weights + bias
				lr = learning_rate / (1 + t)
				weights = update_weights(weights, lr, example.label, example.feat_vec)
				bias = bias + lr*example.label

				t += 1

		if pc_test is not None:
			print('{}\t{}'.format(epoch, pc_test(weight_vector=weights, bias=bias)))

	print('lr, margin, updates:\t{}\t{}\t{}'.format(learning_rate, margin, num_errors))
	return weights, bias


def avgd_perceptron_wrapper(examples, learning_rates, epochs):
	num_feats = len(examples[0].feat_vec)

	# initialize weights
	weights = np.array([0.001 for j in range(num_feats)])
	avg_weights = weights

	bias = 0.001
	avg_bias = 0.001

	# run perceptron for each learning rate
	w_values = [avgd_perceptron(examples, [weights, avg_weights], [bias, avg_bias], lr, epochs) for lr in learning_rates]

	return w_values


def avgd_perceptron(examples, weights, bias, learning_rate, epochs, pc_test=None):
	print('learning_rate:\t{}'.format(learning_rate))
	num_errors = 0
	for epoch in range(epochs):

		for example in examples:
			y_prime = np.dot(np.transpose(weights[0]), example.feat_vec) + bias[0]

			if y_prime > 0:
				y_prime = 1
			else:
				y_prime = -1

			if y_prime != example.label:
				num_errors += 1
				# update weights + bias
				weights[0] = update_weights(weights[0], learning_rate, example.label, example.feat_vec)
				bias[0] = bias[0] + learning_rate * example.label

			# update every example no matter what
			weights[1] += weights[0]
			bias[1] += bias[0]

		if pc_test is not None:
			print('{}\t{}'.format(epoch, pc_test(weight_vector=weights[1], bias=bias[1])))

	print('updates:\t{}'.format(num_errors))
	return weights[1], bias[1]


def aggr_perceptron_wrapper(examples, learning_rates, epochs):
	num_feats = len(examples[0].feat_vec)

	# initialize weights, add extra for bias
	weights = np.array([0.001 for j in range(num_feats+1)])

	# run perceptron for each learning rate
	w_values = [aggr_perceptron(examples, weights, lr, margin, epochs) for lr in
	            learning_rates for margin in learning_rates]

	return w_values


def aggr_perceptron(examples, weights, learning_rate, margin, epochs, pc_test=None):
	# print('learning_rate X margin:\t{}\t{}'.format(learning_rate, margin))
	num_errors = 0
	for epoch in range(epochs):

		for example in examples:
			# w_vector = np.append(w_vector, bias)
			x_vector = np.append(example.feat_vec, [1])
			w_dot_x = np.dot(np.transpose(weights), x_vector)
			if (w_dot_x * example.label) <= margin:
				num_errors += 1
				# update weights + bias
				lr_num = margin - example.label * w_dot_x
				lr_denom = np.dot(np.transpose(x_vector), x_vector) + 1
				lr = lr_num / lr_denom
				weights = update_weights(weights, lr, example.label, x_vector)
				# bias += lr * example.label

		if pc_test is not None:
			print('{}\t{}'.format(epoch, pc_test(weight_vector=weights[:-1], bias=weights[-1])))

	print('lr, margin, updates:\t{}\t{}\t{}'.format(learning_rate, margin, num_errors))
	return weights[:-1], weights[-1]


# def multiclass_perceptron(examples, weights, learning_rate, epochs, pc_test=None):
# 	num_errors = 0
# 	for epoch in range(epochs):
#
# 		for example in examples:
# 			y_prime = np.dot(np.transpose(weights), example.feat_vec) + bias
#
# 			if y_prime > 0:
# 				y_prime = 1
# 			else:
# 				y_prime = -1
#
# 			if y_prime != example.label:
# 				num_errors += 1
# 				# update weights + bias
# 				weights = update_weights(weights, learning_rate, example.label, example.feat_vec)
# 				bias = bias + learning_rate * example.label
#
# 		# if pc_test is not None:
# 		# 	print('epoch:{}\t{}'.format(epoch, pc_test(weight_vector=weights, bias=bias)))
#
# 	print('updates:\t{}'.format(num_errors))
# 	return weights, bias


def perceptron_test(test_data, weight_vector, bias):
	# for each test example,
	correct = 0

	for test_item in test_data:
		# if len(test_item.feat_vec) > len(weight_vector):
		if np.dot(np.transpose(weight_vector), test_item.feat_vec) + bias > 0:
			y_prime = 1
		else:
			y_prime = -1

		if y_prime == test_item.label:
			correct += 1

	return correct / len(test_data)


def cross_validate(cv_split, perceptron_method, pc_test, epochs, margins=None):
	learning_rates = [1, 0.1, 0.01]
	results = [[], [], []]
	if margins is not None:
		results.extend([[], [], [], [], [], []])

	# each 'i' is test
	for i in range(len(cv_split)):
		# each 'j' is training
		training = []
		for j in range(len(cv_split)):
			if i == j:
				continue
			training.extend(cv_split[j])

		# train with 4/5
		# this weight_vals_list has a position for each learning rate
		print('fold:\t{}'.format(str(i)))
		weight_vals_list = perceptron_method(training, learning_rates, epochs)

		# test on i
		for j, weight_val in enumerate(weight_vals_list):

			# weight_val[0] is weight vector; weight_val[1] is bias
			result_acc = pc_test(cv_split[i], weight_val[0], weight_val[1])

			# add result to result vector (whose indices correspond to each learning rate)
			results[j].append(result_acc)


	avg_results = []
	# avg over results
	for lr_results in results:
		# each lr_results is a different learning rate
		avg_acc = sum(lr_results) / len(cv_split)
		avg_results.append(avg_acc)

	# avg_results should be a list with 3 values, one for each learning rate.
	return avg_results


def majority_baseline(dev, train, test):

	labels = [item.label for item in train]
	num_1 = sum(1 for label in labels if label == 1)
	num_neg1 = sum(1 for label in labels if label == -1)
	print(num_1, num_neg1)
	if num_1 > num_neg1:
		freq_label = 1
	else:
		freq_label = -1

	print(freq_label)
	acc = 0
	for item in dev:
		if item.label == freq_label:
			acc += 1

	print('dev_maj_baseline:\t{}'.format(acc/len(dev)))

	acc = 0
	for item in test:
		if item.label == freq_label:
			acc += 1

	print('test_maj_baseline:\t{}'.format(acc / len(test)))

if __name__ == '__main__':
	num_feats = 70
	training_dev = get_data('Dataset//phishing.dev', largest_size=num_feats)
	training_train = get_data('Dataset//phishing.train', largest_size=num_feats)
	training_test = get_data('Dataset//phishing.test', largest_size=num_feats)

	# get the longest vector to guarantee that weight vectors and examples are always the same length
	# largest_size = max(get_longest_vec(training_dev), get_longest_vec(training_test), get_longest_vec(training_train))

	training_cross_val = []
	for i in range(5):
		fold = get_data('Dataset//CVSplits//training0{}.data'.format(str(i)), largest_size=num_feats)
		training_cross_val.append(fold)


	# Majority baseline answer:
	# majority_baseline(training_dev, training_train, training_test)

	do_simple = True
	do_dynamic = True
	do_margin = True
	do_averaged = True
	do_aggr_avg = True

	if do_simple:
		# 1. Simple Perceptron
		print('# 1. Simple Perceptron')
		# part 1
		(lr1, lr2, lr3) = cross_validate(training_cross_val, simple_perceptron_wrapper, perceptron_test, 10)
		print((lr1, lr2, lr3))
		# part 2
		weights = np.array([0.001 for j in range(num_feats)])
		bias = 0.001
		ptd = partial(perceptron_test, test_data=training_dev)
		(weights, bias) = simple_perceptron(training_train, weights, bias, 0.1, 20, ptd)
		# (weights, bias) = simple_perceptron(training_train, weights, bias, 0.1, 6, ptd)
		# part 3
		test_acc = perceptron_test(training_test, weights, bias)
		print(test_acc)

	if do_dynamic:
		# 2. Dynamic Perceptron
		print('\n# 2. Dynamic Perceptron')
		(lr1, lr2, lr3) = cross_validate(training_cross_val, dynamic_perceptron_wrapper, perceptron_test, 10)
		print((lr1, lr2, lr3))
		# part 2
		weights = np.array([0.001 for j in range(num_feats)])
		bias = 0.001
		ptd = partial(perceptron_test, test_data=training_dev)
		(weights, bias) = dynamic_perceptron(training_train, weights, bias, 0.1, 20, ptd)
		# part 3
		test_acc = perceptron_test(training_test, weights, bias)
		print(test_acc)


	if do_margin:
		# 3. Margin Perceptron
		print('\n# 3. Margin Perceptron')
		(lr1_m1, lr1_m2, lr1_m3, lr2_m1, lr2_m2, lr2_m3, lr3_m1, lr3_m2, lr3_m3) = cross_validate(training_cross_val, margin_perceptron_wrapper, perceptron_test, 10, margins=True)
		print((lr1_m1, lr1_m2, lr1_m3, lr2_m1, lr2_m2, lr2_m3, lr3_m1, lr3_m2, lr3_m3))
		# part 2
		weights = np.array([0.001 for j in range(num_feats)])
		bias = 0.001
		ptd = partial(perceptron_test, test_data=training_dev)
		#best_epoch = 16
		(weights, bias) = margin_perceptron(training_train, weights, bias, 1, 0.01, 20, ptd)
		# part 3
		test_acc = perceptron_test(training_test, weights, bias)
		print(test_acc)


	if do_averaged:
		# 4. Averaged Perceptron
		print('\n# 4. Averaged Perceptron')
		(lr1, lr2, lr3) = cross_validate(training_cross_val, avgd_perceptron_wrapper, perceptron_test, 10)
		print(lr1, lr2, lr3)
		# part 2
		weights = np.array([0.001 for j in range(num_feats)])
		bias = 0.001
		ptd = partial(perceptron_test, test_data=training_dev)
		(weights, bias) = avgd_perceptron(training_train, [weights, weights], [bias, bias], 1, 20, ptd)
		# part 3
		test_acc = perceptron_test(training_test, weights, bias)
		print(test_acc)

	if do_aggr_avg:
		# 5. Aggressive Averaged Perceptron
		print('\n# 5. Aggressive Perceptron')
		(lr1_m1, lr1_m2, lr1_m3, lr2_m1, lr2_m2, lr2_m3, lr3_m1, lr3_m2, lr3_m3) = cross_validate(training_cross_val, aggr_perceptron_wrapper, perceptron_test, 10, margins=True)
		print(lr1_m1, lr1_m2, lr1_m3, lr2_m1, lr2_m2, lr2_m3, lr3_m1, lr3_m2, lr3_m3)
		# part 2
		weights = np.array([0.001 for j in range(num_feats+1)])
		ptd = partial(perceptron_test, test_data=training_dev)
		(weights, bias) = aggr_perceptron(training_train, weights, learning_rate=1, margin=1, epochs=20, pc_test=ptd)
		# part 3
		test_acc = perceptron_test(training_test, weights, bias)
		print(test_acc)