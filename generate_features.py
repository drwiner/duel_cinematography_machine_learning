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


def get_num_feats(action_file="action_types.txt", scene_file="scenes.txt"):
	"""
	FEATURES:
	"""
	with open(action_file, 'r') as att:
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

	with open(scene_file, 'r') as stt:
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
	test = "../data/test.txt"
	training_whole = "../data/all_train.txt"
	base_cvsplits = "../data/CV_Splits/train_{}"

	get_num_feats()


	training_examples_xpos = [parse_featlist(base_cvsplits.format(i), "xpos") for i in range(5)]
	training_examples_scale = [parse_featlist(base_cvsplits.format(i), "scale") for i in range(5)]


	whole_training_scale = parse_featlist(training_whole, "scale")

	with open("training.scale", 'w') as tsw:
		for tscale in whole_training_scale:
			tsw.write(str(tscale.label))
			tsw.write(" ")
			for item in tscale.feats:
				tsw.write("{}:{} ".format(str(item+1), str(1)))
			tsw.write("\n")

	scale_labels = [0, 1, 2, 3]
	whole_training_xpos = parse_featlist(training_whole, "xpos")

	with open("training.xpos", 'w') as tsw:
		for txpos in whole_training_xpos:
			tsw.write(str(txpos.label))
			tsw.write(" ")
			for item in txpos.feats:
				tsw.write("{}:{} ".format(str(item+1), str(1)))
			tsw.write("\n")

	xpos_labels = [0, 1, 2, 3, 4]

	test_scale = parse_featlist(test, "scale")
	test_xpos = parse_featlist(test, "xpos")

	with open("test.scale", 'w') as tsw:
		for txpos in test_scale:
			tsw.write(str(txpos.label))
			tsw.write(" ")
			for item in txpos.feats:
				tsw.write("{}:{} ".format(str(item+1), str(1)))
			tsw.write("\n")


	with open("test.xpos", 'w') as tsw:
		for txpos in test_xpos:
			tsw.write(str(txpos.label))
			tsw.write(" ")
			for item in txpos.feats:
				tsw.write("{}:{} ".format(str(item+1), str(1)))
			tsw.write("\n")