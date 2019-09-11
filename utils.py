import os
import csv
import numpy as np
import copy
from hoeffdingtree import *
import arff
import math

CUR_DIR = os.getcwd()
ROOT_DIR = CUR_DIR[:CUR_DIR.index('/codes')]
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def get_filepath(dataname, key):
	file_dir = os.path.join(DATA_DIR, dataname)
	for filename in os.listdir(file_dir):
		if key in filename:
			return os.path.join(file_dir, filename)
	return None

def parse_attributes(data):
	class_index = 0
	data_types = []
	header = []
	for i, attribute in enumerate(data['attributes']):
		if attribute[1] == ['0','1']:
			class_index = i
			break
		header.append(attribute[0])
		data_types.append(attribute[1].capitalize())
	return class_index, header, data_types

def open_dataset(filepath):
	data = arff.load(open(filepath, 'rb'))
	class_index, header, data_types = parse_attributes(data)
	nominal_indices = []
	for i, data_type in enumerate(data_types):
		if data_type != 'Numeric':
			nominal_indices.append(i)
	class_values = []
	instances = []
	for row in data['data']:
		inst = row[:class_index]
		label = reduce(lambda x,y:x+y, row[class_index:])
		class_values.append(label)
		inst.insert(0, label)
		instances.append(inst)

	class_values = list(set(class_values))
	attributes = []
	attributes.append(Attribute('Class', class_values, att_type='Nominal'))
	for i, h in enumerate(header):
		data_type = data_types[i]
		if data_type == 'Numeric':
			attributes.append(Attribute(h, att_type='Numeric'))
		else:
			att_values = data['attributes'][i][1]
			attributes.append(Attribute(h, att_values, att_type='Nominal'))

	dataset = Dataset(attributes, 0)
	for inst in instances:
		inst[0] = int(attributes[0].index_of_value(str(inst[0])))
		for i in nominal_indices:
			inst[i+1] = int(attributes[i+1].index_of_value(str(inst[i+1])))
		dataset.add(Instance(att_values=inst))

	return dataset

def open_slc_dataset(filepath, num_covs=None):
	data = arff.load(open(filepath, 'rb'))
	class_index, header, data_types = parse_attributes(data)
	if num_covs is None or num_covs > class_index:
		num_covs = class_index
	nominal_indices = []
	for i, data_type in enumerate(data_types):
		if data_type != 'Numeric':
			nominal_indices.append(i)
	instances = []
	for row in data['data']:
		inst = row[:class_index]
		label = '0'
		inst.insert(0, label)
		inst = inst[:num_covs+1]
		instances.append(inst)

	num_classes = len(data['data'][0]) - class_index
	class_values = map(str, range(num_classes))
	attributes = []
	attributes.append(Attribute('Class', class_values, att_type='Nominal'))
	for i, h in enumerate(header):
		data_type = data_types[i]
		if data_type == 'Numeric':
			attributes.append(Attribute(h, att_type='Numeric'))
		else:
			att_values = data['attributes'][i][1]
			attributes.append(Attribute(h, att_values, att_type='Nominal'))
	attributes = attributes[:num_covs+1]
	dataset = Dataset(attributes, 0)
	for inst in instances:
		inst[0] = int(attributes[0].index_of_value(str(inst[0])))
		for i in nominal_indices:
			if i > num_covs:
				break
			inst[i+1] = int(attributes[i+1].index_of_value(str(inst[i+1])))
		dataset.add(Instance(att_values=inst))

	return dataset

def convert_to_slc_dataset(dataset):
	n = dataset.num_attributes()
	attributes = []
	for i in xrange(n):
		attributes.append(dataset.attribute(i))
	sample_label = dataset.class_attribute().value(0)
	att_values = map(str, xrange(len(sample_label)))
	attributes[0] = Attribute('Class', att_values, att_type='Nominal')
	new_dataset = Dataset(attributes, 0)
	return new_dataset

def unnormalized_rank_loss(s, Y):
	k = len(s)
	cnt = 0
	Y_complement = set(range(k)).difference(Y)
	if len(Y) == 0 or len(Y_complement) == 0:
		return 0
	for l in Y:
		for r in Y_complement:
			if s[r] > s[l]:
				cnt += 1
			elif s[r] == s[l]:
				cnt += 0.5
	return cnt

def rank_loss(s, Y):
	k = len(s)
	normalize_const = float(len(Y)*(k-len(Y)))
	if normalize_const == 0:
		return 0
	return unnormalized_rank_loss(s, Y) / normalize_const

# this isn't real rank loss. DO NOT MODIFY OMGLSDJFKJ
def pair_hinge_loss(rel_votes, irr_votes):
	if irr_votes >= rel_votes:
		return float(irr_votes - rel_votes)
	return 0.0

def pair_ind_loss(rel_votes, irr_votes):
	if irr_votes >= rel_votes:
		return 1.0
	return 0.0

def pair_dind_loss(rel_votes, irr_votes):
	if irr_votes - rel_votes >= 2:
		return 2.0
	elif irr_votes - rel_votes >= 1:
		return 1.0
	return 0.0

def pair_logloss_gradient_scalar(rel_votes, irr_votes):
	return 1.0 / (1.0 + np.exp(rel_votes - irr_votes))

def pair_logloss_gradient(rel_votes, irr_votes, rel_or_irr):
	scalar = pair_logloss_gradient_scalar(rel_votes, irr_votes)
	if rel_or_irr is True:
		return -scalar
	return scalar

def pair_square_loss(rel_votes, irr_votes):
	if irr_votes >= rel_votes:
		return float(rel_votes - irr_votes)**2
	return 0.0

def pair_exp_loss(rel_votes, irr_votes):
	return np.exp(irr_votes - rel_votes)

def pair_softmax_loss(rel_votes, irr_votes):
	return 1.0 / (1 + np.exp(rel_votes - irr_votes))

def hinge_loss(s, Y):
	k = len(s)
	normalize_const = float(len(Y)*(k-len(Y)))
	_sum = 0
	Y_complement = set(range(k)).difference(Y)
	if normalize_const == 0:
		return 0
	for l in Y:
		for r in Y_complement:
			if s[r] >= s[l]:
				_sum += s[r]-s[l]
	# return _sum/normalize_const
	return _sum

def exp_loss(s, Y):
	k = len(s)
	normalize_const = float(len(Y)*(k-len(Y)))
	Y_complement = set(range(k)).difference(Y)
	_sum = 0.0
	if normalize_const == 0:
		return 0
	for l in Y:
		for r in Y_complement:
			_sum += np.exp(s[r] - s[l])
	return _sum/normalize_const

def univ_exp_loss(s, Y):
	k = len(s)
	normalize_const = float(len(Y)*(k-len(Y)))
	if normalize_const == 0:
		return 0
	s_ndarray = np.array(s)
	s_ndarray[list(Y)] *= -1
	ret = np.exp(s_ndarray).sum()
	return ret/normalize_const

def univ_logistic_loss(s, Y):
	k = len(s)
	normalize_const = float(len(Y)*(k-len(Y)))
	if normalize_const == 0:
		return 0
	s_ndarray = np.array(s)
	s_ndarray[list(Y)] *= -1
	ret = np.log(np.exp(s_ndarray)+1).sum()
	return ret/normalize_const

def mc_potential(t, b, Y=[0], M=10000, s=None, loss=rank_loss):
	k = len(b)
	r = 0
	if s is None:
		s = np.zeros(k)
	_sum = 0.0
	for _ in xrange(M):
		x = np.random.multinomial(t, b)
		x = x + s
		_sum += loss(x, Y)
	return _sum / M

def topk(votes, k):
	return set(votes.argsort()[-k:])

def combinations(total, choose_size):
	assert total >= choose_size
	result = float(math.factorial(total))
	result /= float(math.factorial(choose_size))
	result /= float(math.factorial(total - choose_size))
	return result

def pairwise_potential(num_samples, init_votes, prob, pair_loss, M):
	k = 3
	sum = 0.0
	for _ in range(M):
		sample = np.random.multinomial(num_samples, prob)
		random_final_vote = (sample + init_votes)[:-1] #ignore 'do nothing' votes
		sum += pair_loss(random_final_vote[0], random_final_vote[1])
	return sum / float(M)

def pair_pot_surr_OLD(num_samples, rel_votes, irr_votes, edge, num_classes, pair_loss, M):
	# we add in the extra indices for the case where the adversary does nothing
	init_votes = np.asarray([rel_votes, irr_votes, 0])

	# compute case with max
	base_mass_max = (1-edge)/num_classes
	assert base_mass_max >= 0
	dist_max = np.asarray([edge + base_mass_max, base_mass_max, 0])
	assert np.sum(dist_max) <= 1, dist_max
	dist_max[-1] = 1 - np.sum(dist_max) # place remaining mass on 'do nothing'
	assert np.sum(dist_max) == 1, dist_max
	pot_max = pairwise_potential(num_samples, init_votes, dist_max, pair_loss, M)

	# compute case with min
	base_mass_min = max((1-(num_classes-1)*edge)/num_classes, 0.0)
	dist_min = np.asarray([edge + base_mass_min, base_mass_min, 0])
	assert np.sum(dist_min) <= 1, dist_min
	dist_min[-1] = 1 - np.sum(dist_min)
	assert np.sum(dist_min) == 1, dist_min
	pot_min = pairwise_potential(num_samples, init_votes, dist_min, pair_loss, M)

	return max(pot_max, pot_min)

# TODO: try this guy
def pair_pot_surr(num_samples, rel_votes, irr_votes, edge, num_classes, pair_loss, M, verbose=False):
	init_votes = np.asarray([rel_votes, irr_votes, 0])

	# 1 .. m-1
	max_pot_value = 0.0
	max_index = -1
	for i in range(1, num_classes):
		irr_label_mass = (1.0 - edge * i) / num_classes
		if irr_label_mass < 0.0:
			break
		assert irr_label_mass >= 0
		dist_i = np.asarray([edge + irr_label_mass, irr_label_mass, 0])
		assert np.sum(dist_i) <= 1, dist_i
		dist_i[-1] = 1 - np.sum(dist_i)
		assert np.sum(dist_i) == 1, dist_i
		pot_i = pairwise_potential(num_samples, init_votes, dist_i, pair_loss, M)
		if pot_i > max_pot_value:
			max_pot_value = pot_i
			max_index = i

	if verbose:
		return (max_pot_value, max_index)
	return max_pot_value

# Use this function to turn the labels in the multilabel setting to sets
def label_array_to_set(label_array):
	Ystr = reduce(lambda x,y:x+y, label_array)
	Yset = str_to_set(Ystr)
	return Yset

def str_to_set(_str):
	ret = []
	for i, letter in enumerate(_str):
		if letter == '1':
			ret.append(i)
	return set(ret)

def expit_diff(x, y):
	'''Calculates the logistic (expit) difference between two numbers
	Args:
		x (float): positive value
		y (float): negative value
	Returns:
		value (float): the expit difference
	'''
	value = 1/(1 + np.exp(x - y))
	return value

def read_params():
	filename = 'params.csv'
	filepath = os.path.join(DATA_DIR, filename)
	ret = {}
	with open(filepath, 'rb') as f:
		r = csv.reader(f)
		for row in r:
			key = row[0]
			val = row[1]
			try:
				val = float(val)
			except:
				pass
			ret[key] = val
	return ret
