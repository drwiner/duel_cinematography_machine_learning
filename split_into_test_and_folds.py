import random



if __name__ == '__main__':
	total_examples = []
	with open("observation_features_cleaned_action-pruned_full.txt", 'r') as ofca:
		for line in ofca:
			total_examples.append(eval(line))

	random.shuffle(total_examples)

	train = total_examples[:2300]
	test = total_examples[2300:]

	num_per_fold = int(2300/5)
	train_splits = [total_examples[i:i+num_per_fold] for i in range(0, len(total_examples)+1, num_per_fold)]

	# with open("data/all_train.txt", 'w') as att:
	# 	for example in train:
	# 		att.write(str(example))
	# 		att.write('\n')

	for fold in range(5):
		with open("data/CV_Splits/train_{}".format(fold), 'w') as cv_fold:
			for example in train_splits[fold]:
				cv_fold.write(str(example))
				cv_fold.write("\n")

	# with open("data/test.txt", 'w') as ftw:
	# 	for example in test:
	# 		ftw.write(str(example))
	# 		ftw.write("\n")

