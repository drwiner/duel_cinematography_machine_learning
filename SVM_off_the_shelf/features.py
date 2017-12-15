
UNKPOS = 'UNKPOS'
UNKWORD = 'UNKWORD'
PHI = 'PHI'
PHIPOS = 'PHIPOS'
OMEGA = 'OMEGA'
OMEGAPOS = 'OMEGAPOS'
PSEUDO_POS = {OMEGAPOS, PHIPOS, UNKPOS}
PSEUDO_WORDS = {OMEGA, PHI, UNKWORD}


LABELS = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
LABEL_DICT = dict(zip(LABELS, list(range(0, len(LABELS)))))

CAP_ID = None
WORD_DICT = None
PREV_POS_DICT = None
NEXT_POS_DICT = None
PREV_WORD_DICT = None
NEXT_WORD_DICT = None

class Row:

	def __init__(self, label, pos, word, prev_pos, prev_word, cap=None):
		self.label = label
		self.pos = pos
		self.word = word
		self.prev_pos = prev_pos
		self.prev_word = prev_word
		self.cap = self.word[0].isupper()
		if cap is not None:
			self.cap = cap

	def next(self, next_pos, next_word):
		self.next_pos = next_pos
		self.next_word = next_word


def extract_training(train):
	line_array = []
	new_sentence = True
	beg = True

	for line in train:
		row = line.split()
		if row:

			if new_sentence:
				# then new sentence
				new_sentence = False
				prev_pos, prev_wrd = OMEGAPOS, OMEGA
				if not beg:
					# update end of last sentence
					line_array[-1].next(OMEGAPOS, OMEGA)

				else:
					beg = False

			else:
				# middle/last of sentence, update end of last
				line_array[-1].next(row[1], row[2])
				prev_pos, prev_wrd = line_array[-1].pos, line_array[-1].word

			line_array.append(Row(label=row[0], pos=row[1], word=row[2], prev_pos=prev_pos, prev_word=prev_wrd))
		else:
			# last is end of sentence, there is no row here
			new_sentence = True

	line_array[-1].next(OMEGAPOS, OMEGA)
	return line_array


def extract_testing(test_text, line_array):
	prev_pos_set = {row.prev_pos for row in line_array}.union(PSEUDO_POS)
	next_pos_set = {row.next_pos for row in line_array}.union(PSEUDO_POS)
	prev_word_set = {row.prev_word for row in line_array}.union(PSEUDO_WORDS)
	next_word_set = {row.next_word for row in line_array}.union(PSEUDO_WORDS)

	train_words = {line.word for line in line_array}
	train_pos = {line.pos for line in line_array}

	global UNKWORD, PHI, OMEGA, UNKPOS, PHIPOS, OMEGAPOS
	# protect against these words from actually being the words/pos
	if UNKWORD in train_words:
		UNKWORD += 'davidrosswiner'
	if PHI in train_words:
		PHI += 'davidrosswiner'
	if OMEGA in train_words:
		OMEGA += 'davidrosswiner'
	if UNKPOS in train_pos:
		UNKPOS += 'davidrosswiner'
	if PHIPOS in train_pos:
		PHIPOS += 'davidrosswiner'
	if OMEGAPOS in train_pos:
		OMEGAPOS += 'davidrosswiner'

	line_array_test = []
	new_sentence = True
	beg = True
	for line in test_text:
		row = line.split()
		if row:
			cap = None

			if row[1] not in train_pos:
				row[1] = UNKPOS
			if row[2] not in train_words:
				if row[2][0].isupper():
					cap = True
				else:
					cap = False
				row[2] = UNKWORD

			if new_sentence:
				# then new sentence
				new_sentence = False
				prev_pos, prev_wrd = OMEGAPOS, OMEGA
				if not beg:
					# update end of last sentence
					line_array_test[-1].next(OMEGAPOS, OMEGA)

				else:
					beg = False

			else:
				# middle/last of sentence

				# update last's next_pos, next_word
				r_pos, r_word = row[1], row[2]
				if r_pos not in next_pos_set:
					r_pos = UNKPOS
				if r_word not in next_word_set:
					r_word = UNKWORD
				line_array_test[-1].next(r_pos, r_word)

				# update last's prev_pos, prev_word
				prev_pos, prev_wrd = line_array_test[-1].pos, line_array_test[-1].word
				if prev_pos not in prev_pos_set:
					prev_pos = UNKPOS
				if prev_wrd not in prev_word_set:
					prev_wrd = UNKWORD

			line_array_test.append(
				Row(label=row[0], pos=row[1], word=row[2], prev_pos=prev_pos, prev_word=prev_wrd, cap=cap))

		else:
			# last is end of sentence, there is no row here
			new_sentence = True
	line_array_test[-1].next(OMEGAPOS, OMEGA)
	return line_array_test, train_words, train_pos

def makeIDs(items, last, unkpos=False, unkword=False):
	start = last
	end = len(items) + start + 1
	idd = dict(zip(items, list(range(start, end))))
	if unkpos:
		idd.update({UNKPOS: end, PHIPOS: end+1, OMEGAPOS: end+2}); end += 3
	if unkword:
		idd.update({UNKWORD: end, PHI: end+1, OMEGA: end+2}); end += 3

	return idd, end

def setupIDs(train_words, line_array):
	##### ADD psuedo word/pos for unkown ####
	train_words.update({UNKWORD})

	# word ids
	index = len(train_words) + 1

	global WORD_DICT
	WORD_DICT = dict(zip(train_words, list(range(1, index))))

	# cap id
	global CAP_ID
	CAP_ID = max(WORD_DICT.values()) + 1
	last = CAP_ID + 1

	# prev_pos ids
	global PREV_POS_DICT, NEXT_POS_DICT
	PREV_POS_DICT, last = makeIDs({row.prev_pos for row in line_array}, last, unkpos=True)
	# next_pos ids
	NEXT_POS_DICT, last = makeIDs({row.next_pos for row in line_array}, last, unkpos=True)

	# prev_word ids
	global PREV_WORD_DICT, NEXT_WORD_DICT
	PREV_WORD_DICT, last = makeIDs({row.word for row in line_array}, last, unkword=True)
	# next_word ids
	NEXT_WORD_DICT, last = makeIDs({row.next_word for row in line_array}, last, unkword=True)

	return

def process(train_file, test_file, ftypes):
	#print('process received +' + str(ftypes))
	train_text = open(train_file)
	test_text  = open(test_file)

	line_array = extract_training(train_text)
	line_array_test, train_words, train_pos = extract_testing(test_text, line_array)

	# debug message
	print('Found {} training instances with {} distinct words and {} distinct POS tags'.format(len(line_array), len(train_words), len(train_pos)))
	print('Found {} test instances'.format(len(line_array_test)))

	setupIDs(train_words, line_array)

	for ftype in ftypes:
		print('generating features for ' + ftype)
		generateFeatures(ftype, line_array, line_array_test)

def toFeat(row, feat_type):

	feats = []
	feats.append(WORD_DICT[row.word])

	if str(feat_type) != 'word':

		if row.cap:
			feats.append(CAP_ID)

		if str(feat_type) == 'poscon' or str(feat_type) == 'bothcon':
			feats.append(PREV_POS_DICT[row.prev_pos])
			feats.append(NEXT_POS_DICT[row.next_pos])

		if str(feat_type) == 'lexcon' or str(feat_type) == 'bothcon':
			feats.append(PREV_WORD_DICT[row.prev_word])
			feats.append(NEXT_WORD_DICT[row.next_word])

	feature_str = str(LABEL_DICT[row.label]) + ' '
	for f in feats:
		feature_str += str(f) + ':1 '

	return feature_str + '\n'



def generateFeatures(ftype, line_array, line_array_test):
	print(ftype)
	train = open('train.' + str(ftype), 'w')
	test = open('test.' + str(ftype), 'w')

	for line in line_array:
		train.write(toFeat(line, ftype))
	train.close()
	for line in line_array_test:
		test.write(toFeat(line, ftype))
	test.close()

	
import sys
if __name__ == '__main__':
	words = 'word wordcap poscon lexcon bothcon'.split()
	if len(sys.argv) > 3:
		ftypes = []
		if str(sys.argv[3]) == 'all':
			ftypes = words
		else:
			ftypes = [str(sys.argv[3])]

		process(sys.argv[1], sys.argv[2], ftypes)
	else:
		process('train.txt', 'test.txt', words)

# generateFeatures('word')
# generateFeatures('wordcap')
# generateFeatures('poscon')
# generateFeatures('lexcon')
#generateFeatures('bothcon')
#process('train.txt', "test.txt", ['bothcon'])