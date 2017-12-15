import SceneDataStructs


entity_path = 'entity_folder\\'

# a class for an entity object
class Entity:
	def __init__(self, entity_name, types=None, role=None):
		self.name = entity_name
		self.types = types
		self.role = role
		#
		# # keys are shot numbers, value is pair of spatial positions (shotstart, shotend)
		# self.spatial_dict = dict()

	def asDict(self):
		return self.__dict__

	def __hash__(self):
		return hash(str(self.name) + str(self.role))

	def __eq__(self, other):
		if hasattr(other, name):
			if self.name == other.name:
				return True
			return False
		else:
			if self.name == other:
				return True
			return False

	def __str__(self):
		return str(self.name) + '_' + str(self.role)

	def __repr__(self):
		return str(self.name) + '_' + str(self.role)

def generateEntities(scene_lib):
	print('writing entities:')
	for sc_name, scene in scene_lib.items():
		if sc_name is None:
			continue
		print(sc_name)
		scene_entity_file = open(entity_path + 'scene' + sc_name + '_entities.txt', 'w')
		for entity in scene.entities:
			scene_entity_file.write(entity)
			scene_entity_file.write('\n')

def readEntityRoles(scene_file):
	role_dict = dict()
	subs = False
	for line in scene_file:
		split_line = line.split()
		if len(split_line) == 0:
			continue
		if not subs:
			if len(split_line) > 1:
				role_dict[split_line[0]] = [Entity(split_line[0], role=split_line[-1])]
			else:
				role_dict[split_line[0]] = [Entity(split_line[0])]
			if split_line[-1] == '_':
				subs = True
				continue
		if subs:
			role_dict[split_line[0]] = [wrd.lower() for wrd in split_line[2:]]
	return role_dict


def assignRoles(scene_lib):
	print('assigning entities to roles')

	for sc_name, scene in scene_lib.items():
		if sc_name in SceneDataStructs.EXCLUDE_SCENES:
			continue
		# print(sc_name)
		scene_entity_file = open(entity_path + 'scene' + sc_name + '_entities_coded.txt')
		rd = readEntityRoles(scene_entity_file)
		scene.substituteEntities(rd)
		scene_entity_file.close()

	print(scene_lib)



if __name__ == '__main__':
	from SceneDataStructs import Scene, SceneLib, Shot, Action, ActionType
	print('loading scene library')
	scene_lib = SceneDataStructs.load()
	# print(scene_lib)

	assignRoles(scene_lib)
	SceneDataStructs.save_scenes(scene_lib)