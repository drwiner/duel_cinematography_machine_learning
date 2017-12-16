import subprocess


def train_i(task, i, d):

	model = 'model.' + task + "_" + str(i) + "_" + str(d)
	train = base_cvsplits.format(task, i, d)
	command = 'train -s 0 ' + train + ' ' + model
	print(command)
	subprocess.call(command, shell=True)

def predict_i(task,i,d):

	test_features = base_cvsplits.format(task, i, d)
	model = 'model.' + task + "_" + str(i) + "_" + str(d)
	predict = 'predict.' + task + "_" + str(i) + "_" + str(d)
	acc = 'accuracy_' + task + "_" + str(i) + "_" + str(d) + '.txt'
	command = 'predict ' + test_features + ' ' + model + ' ' + predict + ' > ' + acc
	print(command)
	subprocess.call(command, shell=True)


def train(task, d):

	model = 'model.' + task + "_" + str(d)
	train = train_file.format(task, d)
	command = 'train -s 0 ' + train + ' ' + model
	print(command)
	subprocess.call(command, shell=True)

def predict(task, d):

	test_features = test_file.format(task, d)
	model = 'model.' + task + "_" +   str(d)
	predict = 'predict.' + task + "_"   + str(d)
	acc = 'accuracy_' + task + "_"  + str(d) + '.txt'
	command = 'predict ' + test_features + ' ' + model + ' ' + predict + ' > ' + acc
	print(command)
	subprocess.call(command, shell=True)


def accumulate_averages():
	base_cvsplits = "CV_Splits/training.{}_{}_{}"

	train_file = "../bagged_forest_training_{}_{}.txt"
	test_file = "../bagged_forest_training_{}_{}.txt"
	# test_file = "../test_{}_{}.txt"

	accuracy_file = "accuracy_{}_{}_{}"



	for task in ["scale", "xpos"]:
		print(task)
		# example_sets = [load_examples_from_file(base_cvsplits.format(task, i)) for i in range(5)]

		for d in range(5, 16, 3):
			print(d)

			total_acc = 0
			for i in range(5):
				for j in range(5):

					if i == j:
						continue

					accuracy_file.format(task, i, d)
					total_acc += 1

if __name__ == "__main__":

	# from collections import defaultdict
	# from generate_features import load_examples_from_file


	base_cvsplits = "CV_Splits/training.{}_{}_{}"

	train_file = "../bagged_forest_training_{}_{}.txt"
	# test_file = "../test_{}_{}.txt"
	test_file = "../bagged_forest_test_{}_{}.txt"


	for task in ["scale", "xpos"]:
		print(task)
		# example_sets = [load_examples_from_file(base_cvsplits.format(task, i)) for i in range(5)]

		for d in range(5, 16, 3):
			print(d)

			train(task, d)
			predict(task, d)

			# for i in range(5):
			# 	for j in range(5):
			#
			# 		if i == j:
			# 			continue
			#
			# 		with open(base_cvsplits.format(task, i, d)

					# train(task, i, d)
					# predict(task, i, d)