"""
The purpose of this script is to calculate the observation features (and labels)

"""
import SceneDataStructs as SDS
from collections import defaultdict
from collections import Counter


# basic scales
CU = ['cu', 'ecu', 'mcu', 'closeup', 'extremecloseup', 'extremecu', 'mediumcu', 'cutoecu', 'tightecu', 'mcutocu', 'cutomcu', 'tightcu', 'cujohnny', 'cudoc', 'ms-cu', 'ecu/profile', 'cuprofile', 'cu/profile', 'mcudoc', 'culockettoface']
WAIST = ['waist', 'waistshot', 'waistshot(\"cowboy\")', 'waistshot-monacoexitslleavingcolonelaloneinframe', 'waistshotslightlybelowbeltline', 'waisttoms', 'wast', '3/4figurewaist', 'waist-3.25', 'waistshot/2characters', 'waiste', 'waist/widebg', '3/4figuretowaist', '3.4-waist', 'waist/reverse', '2-shot/waist']
FIGURE = ['figure', '3/4figure', '0.75', 'fullfigure', 'full', '3.25', 'full-figure', 'ful', 'fullshot', '3/4figureshot', '3/4fig', '0.75-ff', 'full-3.25', '3.25shot']
WIDE = ['wide','w', 'wideshot', 'mediumwide', 'extremewide', 'ew', 'wide/full', 'extremewideshot', 'extremewideshot3characters', 'medwide\"minimaster\"', 'extremewideshotrevealinghandinfg',
'full-wide', 'wideangle', 'medwide']
SCALES = [CU, WAIST, FIGURE, WIDE]

# different scales for foreground and background entities
CU_WIDE = ['overshoulder/extremewide', 'closeup/wide', 'extremecloseup/wide', 'wide/ovs', 'cu/widebg', 'wideovershouldertobandits','w-cuforeground', 'cu-widebackground', 'cu-wide', 'closeovershoulder/wide', 'cu-w']
CU_FIGURE = ['closeup/3.25', 'closeup/0.75bg']
CU_WAIST = ['waist/closeup']
WAIST_WIDE = ['0.75/wide', '0.75/w', 'wideshotwithjoeinnearfg', '0.75ovs/wide', 'waist/widebg', ]
FIGURE_WIDE = ['0,75/w', 'ff/w', 'full-wide', 'wide/full']
TWO_SCALES = [CU_WIDE, CU_FIGURE, CU_WAIST, WAIST_WIDE, FIGURE_WIDE]
TWO_SCALE_LABELS = [('cu','wide'), ('cu', 'figure'), ('cu', 'waist'), ('waist', 'wide'), ('figure', 'wide')]


def extract_scale(scale, zeta):
	new_scale = None
	for scale_type in SCALES:
		if scale in scale_type:
			return scale_type[0]
	if new_scale is None:
		for scale_type in TWO_SCALES:
			for j, scale in enumerate(scale_type):
				scale_tuple = TWO_SCALE_LABELS[j]
				if zeta == 1:
					# background
					return scale_tuple[1]
				else:
					return scale_tuple[0]
	return new_scale


BG = ['background', 'ackground', 'backgrooud']
FG = ['foreground', 'foeground', 'foregorund', 'forground']
ONE_WORD_POS = ['left', 'right', 'center', 'full']
POS = ['left', 'right', 'center-right', 'center-left', 'left-center', 'full', 'center', 'right-center']


def extract_zeta(pos, zeta):
	sp = pos.split('-')
	collected_parts = []
	_zeta = None
	_pos = None

	for part in sp:
		if part in ONE_WORD_POS: collected_parts.append(part)
		if part in BG: _zeta = 1
		elif part in FG: _zeta = 0

	if 'left' in collected_parts:
		if 'center' in collected_parts: _pos = 'center-left'
		elif 'full' in collected_parts: _pos = 'full'
		else: _pos = 'left'
	elif 'right' in collected_parts:
		if 'center' in collected_parts: _pos = 'center-right'
		elif 'full' in collected_parts: _pos = 'full'
		else: _pos = 'right'
	else:
		if 'full' in collected_parts: _pos = 'full'
		elif 'center' in collected_parts: _pos = 'center'
		else: raise ValueError('Should be a left-right-center position')


	if _zeta is None: _zeta = zeta
	if _pos is None: _pos = pos

	return _pos, _zeta

def extract_composition(_str):
	"""
	:param _str: a figureobjs(shotstart) string
	:return: one of (left, center-left, center, center-right, right), plus forground/background
	"""

	ents = []
	if _str is None or _str == 'None' or _str == 'none':
		return ''
	str_list = _str.split(',')
	for i in str_list:
		if i == '':
			continue
		z = i.split('(')
		new_ent = []
		for k in z:
			if k == '':
				continue
			m = k.split(')')
			for q in m:
				if q == '':
					continue
				new_ent.append(q)
		# new_ent[1]
		ents.append(new_ent)
	return ents


def get_actions_in_shot(shot, entity, none_town):
	actions_in_shot_w_ent = []
	actions_in_shot = []
	for action in shot.actions:
		if action in none_town or action._type in none_town:
			continue

		if type(action._type) is str:
			action_name = action._type

		elif action._type.type_name in none_town or action._type.type_name in none_town:
			action_name = action._type._orig_name
		else:
			action_name = action._type.type_name

		if action_name in none_town or str(action_name) in none_town:
			continue

		for ent in action._args:
			if entity == str(ent):
				actions_in_shot_w_ent.append(action_name)
			else:
				actions_in_shot.append(action_name)

	return actions_in_shot, actions_in_shot_w_ent



def time_to_seconds(shotlength):
	return shotlength.second + shotlength.microsecond / 1000000


def extract_labeled_features(scene_lib):
	none_town = {'None', 'none', None}
	# gOAL: create tuple (entity, action, starts, finishes, shot_num, scale, fore/back, horizon_pos)

	labeled_observations_per_scene = dict()
	for scene_name, scene in scene_lib.items():
		if scene_name in none_town:
			continue
		labeled_observations_per_scene[scene_name] = []

		preceding_shot = None
		time_into_scene = 0
		total_scene_length = sum(time_to_seconds(shot.shotlength) for shot in scene if shot is not None)

		for shot_num, shot in enumerate(scene):

			if shot.scale in none_town:
				continue

			current_scale = str(shot.scale)
			shot_duration = time_to_seconds(shot.shotlength)

			fground_dict = {str(ent[0]): str(ent[1]) for ent in shot.foreground if len(ent)>1}
			bground_dict = {str(ent[0]): str(ent[1]) for ent in shot.background if len(ent)>1}

			for action in shot.actions:
				if action in none_town or action._type in none_town:
					continue

				if type(action._type) is str:
					action_name = action._type

				elif action._type.type_name in none_town or action._type.type_name in none_town:
					action_name = action._type._orig_name
				else:
					action_name = action._type.type_name

				if action_name in none_town or str(action_name) in none_town:
					continue


				# check for each argument if any of them are in the shot.
				arg_2_pos = dict()
				for arg in action._args:

					if str(arg) in fground_dict.keys():
						pos = (0, fground_dict[str(arg)])
					elif str(arg) in bground_dict.keys():
						pos = (1, bground_dict[str(arg)])
					else:
						pos = (None, None)
					arg_2_pos[str(arg)] = pos


				# if action_name

				# create observation for each entity
				for arg, pos in arg_2_pos.items():
					acts_in_shot, acts_with_ent = get_actions_in_shot(shot, str(arg), none_town)
					if preceding_shot is not None:
						acts_in_last_shot, acts_in_last_shot_with_ent = get_actions_in_shot(preceding_shot, str(arg), none_town)
					else:
						acts_in_last_shot, acts_in_last_shot_with_ent = ([], [])

					#### THE OBSERVATION ####
					labeled_observation = (
						arg, action_name, action.starts, action.finishes,
						acts_with_ent, acts_in_last_shot_with_ent, acts_in_shot, acts_in_last_shot,
						scene_name, shot_duration, time_into_scene/total_scene_length, shot_num, len(scene),
						current_scale, pos[0], pos[1]
					)

					labeled_observations_per_scene[scene_name].append(labeled_observation)

				preceding_shot = shot

			time_into_scene += shot_duration

	return labeled_observations_per_scene


def find_interval_span(a, action_instances):

	if a.starts == 'phi':
		a.starts = max_o_same_arg(a, action_instances) + 1
	if a.finishes == 'psi':
		a.finishes = min_o_same_arg(a, action_instances) - 1



def extract_and_print_original_observations(scene_lib, file_name):
	labeled_features = extract_labeled_features(scene_lib)

	with open(file_name, 'w') as asd:
		for scene_name, observation_list in labeled_features.items():
			for obs in observation_list:
				asd.write(str(obs))
				asd.write('\n')
		asd.write('\n')


def clean_observations(obs_file, obs_file_cleaned):
	cleaned_list = []
	with open(obs_file, 'r') as asb:
		for line in asb:
			if line == '\n':
				continue

			feat = eval(line)
			if feat[-1] is None:
				continue
			if feat[-2] is None:
				continue

			zeta = feat[-2]
			pos = feat[-1]
			scale = feat[-3]

			if feat[4] == 'bc':
				print('stop')

			# first, make sure zeta is accurate
			(new_pos, new_zeta) = extract_zeta(pos, zeta)
			new_scale = extract_scale(scale, zeta)
			# new scale commandeered

			old_list = list(feat)[0:-3]
			old_list.extend([new_scale, new_zeta, new_pos])
			cleaned_list.append(tuple(old_list))


	with open(obs_file_cleaned, 'w') as asd:
		for obs in cleaned_list:
			asd.write(str(obs))
			asd.write('\n')


def get_most_frequent_actions(obs_file):
	actions = []

	with open(obs_file, 'r') as asb:
		for line in asb:
			if line == '\n':
				continue
			feat = eval(line)
			actions.append(feat[1])

	c_a = Counter(actions)
	x_list = c_a.most_common(39)
	acts = [x[0] for x in x_list]
	return acts


def clean_actions(obs_file, obs_file_pruned):
	top_acts = get_most_frequent_actions(obs_file)
	broken_actions = {'firegun': 'fire-gun', 'aimgun': 'aim-gun', 'getshot': 'get-shot', 'drawgun': 'draw-gun', 'drawguns': 'draw-gun'}
	new_obs = []
	with open(obs_file, 'r') as asd:
		for line in asd:
			if line == '\n':
				continue
			feat = eval(line)
			if feat[1] in broken_actions.keys():
				new_line_list = list(feat)
				new_line_list[1] = broken_actions[feat[1]]
				new_line = str(tuple(new_line_list)) + '\n'
			elif feat[1] not in top_acts:
				continue
			else:
				new_line = line

			# fix some minor problems with actions; if number of items on feature changes, need more clever strategy

			new_obs.append(new_line)

	with open(obs_file_pruned, 'w') as asb:
		for obs in new_obs:
			asb.write(obs)

def eval_feats(obs_file):
	actions = []
	scales = []
	comps = []
	with open(obs_file, 'r') as asb:
		for line in asb:
			if line == '\n':
				continue
			feat = eval(line)

			actions.append(feat[1])
			scales.append(feat[-3])
			comps.append(feat[-1])

	c_a = Counter(actions)
	c_s = Counter(scales)
	c_c = Counter(comps)
	print('stop')

if __name__ == '__main__':
	# if no SDS to load, go to Narrative Understanding Pipeline and load it up
	# scene_lib = SDS.load('WESTERN_DUEL_CORPUS_MASTER_update..pkl')

	obs_file = 'observation_features_full.txt'
	obs_file_cleaned = 'observation_features_cleaned_full.txt'
	obs_file_cleaned_pruned = 'observation_features_cleaned_action-pruned_full.txt'

	train_file = "data/all_train.txt"

	# extract_and_print_original_observations(scene_lib, obs_file)
	# clean_observations(obs_file, obs_file_cleaned)
	# clean_actions(obs_file_cleaned, obs_file_cleaned_pruned)
	eval_feats(obs_file_cleaned_pruned)

	actions = []


	with open(train_file, 'r') as asb:
		for line in asb:
			if line == '\n':
				continue
			feat = eval(line)
			actions.append(feat[1])

	actions = set(actions)

	with open("durations.txt", 'w') as atw:
		for dur in actions:
			atw.write(str(dur))
			atw.write("\n")


	# get_most_frequent_actions("observation_features_full.txt")

	#
	# clean_actions(obs_file_cleaned)


	print('end')