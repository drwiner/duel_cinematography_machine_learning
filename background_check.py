from collections import defaultdict

if __name__ == '__main__':

	obs_name = 'observation_features_cleaned_action-pruned.txt'

	foreground = 0
	background = 0
	with open(obs_name, 'r') as file_name:
		for line in file_name:
			feats = eval(line)
			zeta = feats[-2]
			if zeta == 0: foreground+=1
			elif zeta==1: background+=1

	print('foreground: {}'.format(foreground))
	print('background: {}'.format(background))