# check of any labels are inconsistent

from collections import defaultdict

if __name__ == '__main__':

	obs_name = 'observation_features_cleaned_action-pruned.txt'

	feat_dict_scale = defaultdict(list)
	feat_dict_pos = defaultdict(list)

	with open(obs_name, 'r') as file_name:
		for line in file_name:
			feats = eval(line)
			feat_dict_scale[(feats[1], feats[2], feats[3])].append(feats[-3])
			feat_dict_pos[(feats[1], feats[2], feats[3])].append(feats[-1])


	print('stop')