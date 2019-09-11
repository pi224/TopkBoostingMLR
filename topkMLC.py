import csv
import numpy as np
import copy
from hoeffdingtree import *
import utils
from batchMLC import AdaBoostMM

class TopkAdaBoostOLM:
	'''
	Main class for Online Multiclass AdaBoost algorithm using VFDT.

	Notation conversion table:

	v = expert_weights
	alpha =  wl_weights
	sVec = expert_votes
	yHat_t = expert_preds
	C = cost_mat

	'''

	# TODO: add in rho as a parameter
	def __init__(self, data_source, k, rho, d, loss='logistic',
					num_covs=20, gamma=0.1):
		# Initializing computational elements of the algorithm
		self.num_wls = None
		self.num_classes = None
		self.num_data = 0
		self.dataset = None
		self.class_index = 0
		self.cum_error = 0
		self.exp_step_size = 1
		self.loss = loss
		self.gamma = gamma
		self.M = 100
		self.num_covs = num_covs

		self.wl_edges = None
		self.weaklearners = None
		self.expert_weights = None
		self.wl_weights = None
		self.wl_preds = None
		self.expert_preds = None
		self.cached_potentials = {}
		self.cache_hits = 0
		self.potential_calls = 0

		# Initializing data states
		self.X = None
		self.Yhat_index = None
		self.Y_index = None
		self.Yhat = None
		self.Y = None
		self.pred_conf = None

		# top-k feedback variables
		self.k = k
		self.rho = rho
		self.d = d

		self.initialize_dataset(data_source)

	########################################################################

	# Helper functions
	def make_cov_instance(self, X):
		'''Turns a list of covariates into an Instance set to self.datset
		with None in the location of the class of interest. This is required to
		pass to a HoeffdingTree so it can make predictions.

		Args:
			X (list): A list of the covariates of the current data point.
					  Float for numerical, string for categorical. Categorical
					  data must have been in initial dataset

		Returns:
			pred_instance (Instance): An Instance with the covariates X and
					  None in the correct locations

		'''
		inst_values = list(copy.deepcopy(X))

		inst_values = map(float, inst_values)

		inst_values.insert(self.class_index, None)

		# indices = range(len(inst_values))
		# del indices[self.class_index]
		# for i in indices:
		#     if self.dataset.attribute(index=i).type() == 'Nominal':
		#         inst_values[i] = int(self.dataset.attribute(index=i)
		#             .index_of_value(str(inst_values[i])))
		#     else:
		#         inst_values[i] = float(inst_values[i])


		pred_instance = Instance(att_values = inst_values)
		pred_instance.set_dataset(self.slc_dataset)
		return pred_instance

	def make_full_instance(self, X, Y):
		'''Makes a complete Instance set to self.dataset with
		class of interest in correct place

		Args:
			X (list): A list of the covariates of the current data point.
					  Float for numerical, string for categorical. Categorical
					  data must have been in initial dataset
			Y (string): the class of interest corresponding to these covariates.

		Returns:
			full_instance (Instance): An Instance with the covariates X and Y
							in the correct locations

		'''

		inst_values = list(copy.deepcopy(X))

		inst_values = map(float, inst_values)

		inst_values.insert(self.class_index, Y)
		i = self.class_index
		inst_values[i] = int(self.slc_dataset.attribute(index=i)
					.index_of_value(str(inst_values[i])))
		# for i in range(len(inst_values)):
		#     if self.dataset.attribute(index=i).type() == 'Nominal':
		#         inst_values[i] = int(self.slc_dataset.attribute(index=i)
		#             .index_of_value(str(inst_values[i])))
		#     else:
		#         inst_values[i] = float(inst_values[i])


		full_instance = Instance(att_values=inst_values)
		full_instance.set_dataset(self.slc_dataset)
		return full_instance

	def find_Y(self, Y_index):
		'''Get class string from its index
		Args:
			Y_index (int): The index of Y
		Returns:
			Y (string): The class of Y
		'''

		Y = self.dataset.attribute(index=self.class_index).value(Y_index)
		return Y

	def find_Y_index(self, Y):
		'''Get class index from its string
		Args:
			Y (string): The class of Y
		Returns:
			Y_index (int): The index of Y
		'''

		Y_index = int(self.dataset.attribute(index=self.class_index)
					.index_of_value(Y))
		return Y_index

	########################################################################


	# TODO: test this guy
	def generate_random_ranking(self, Yhat, topkYhat):
		# with probability 1-rho, do nothing
		if np.random.random() > self.rho:
			return Yhat

		# deep copy Yhat
		Ytilde = np.array([x for x in Yhat])

		# the members of the topk and not topk set are the indices
		NOTtopkYhat = set(range(self.num_classes)) - topkYhat
		assert len(NOTtopkYhat.intersection(topkYhat)) == 0

		swapping_NOTtopk_indices = np.random.choice(list(NOTtopkYhat), self.d, False)
		swapping_topk_indices  = np.random.choice(list(topkYhat), self.d, False)

		for i, j in zip(swapping_topk_indices, swapping_NOTtopk_indices):
			Ytilde[i], Ytilde[j] = Ytilde[j], Ytilde[i]

		return Ytilde

	# TODO: need to debug this
	def pair_prob(self, rel_label, irr_label):
		# Actually, it doesn't matter if rel_label or irr_label is relevant
		m = self.num_classes
		k = self.k
		d = self.d
		rho = self.rho
		topkYhat = self.topkYhat
		if rel_label in topkYhat and irr_label in topkYhat:
			if d > k - 2:
				return 1 - rho
			assert d <= k - 2
			numerator = utils.combinations(k-2, d)
			denominator = utils.combinations(k, d)
			return 1 - rho + rho * numerator / denominator
		elif rel_label in topkYhat or irr_label in topkYhat:
			if d == k:
				return rho
			assert d < k
			numerator_topk = utils.combinations(k-1, d)
			denominator_topk = utils.combinations(k, d)
			result = numerator_topk / denominator_topk
			numerator_Ntopk = utils.combinations(m - k, d - 1)
			denominator_Ntopk = utils.combinations(m - k, d)
			result *= numerator_Ntopk / denominator_Ntopk
			return rho * result
		else:
			numerator = utils.combinations(m-k, d-2)
			denominator = utils.combinations(m-k, d)
			return rho * numerator / denominator

	def pair_pot_surr_manager(self, num_samples, rel_votes, irr_votes):
		self.potential_calls += 1
		# subtract minimum of the two votes, because it doesn't affect the loss function
		min_votes = min(rel_votes, irr_votes)
		irr_votes -= min_votes
		irr_votes = np.round(irr_votes, 2)

		rel_votes -= min_votes
		rel_votes = np.round(rel_votes, 2)


		key = (num_samples, rel_votes, irr_votes)
		if key not in self.cached_potentials:
			new_pot_result = utils.pair_pot_surr(num_samples, rel_votes, irr_votes,
						self.gamma, self.num_classes, utils.pair_hinge_loss, self.M)
			self.cached_potentials[key] = new_pot_result
		else:
			self.cache_hits += 1
		return self.cached_potentials[key]

	def compute_rankloss_estimate(self, s):
		rel_found = self.topkRel
		irr_found = self.topkNOTRel
		estimate = 0.0
		for a in rel_found:
			for b in irr_found:
				estimate += utils.pair_ind_loss(s[a], s[b]) / self.pair_prob(a, b)
		return estimate

	def compute_logloss_gradient_estimate(self, s, index):
		rel_found = self.topkRel
		irr_found = self.topkNOTRel
		gradient = 0.0
		# true or false dictates if the gradient is wrt to relevant or irrelevant label
		# print('rel_found:', rel_found)
		# print('irr_found:', irr_found)
		if index in rel_found:
			for b in irr_found:
				gradient += utils.pair_logloss_gradient(s[index], s[b], True) / self.pair_prob(index, b)
		elif index in irr_found:
			for a in rel_found:
				gradient += utils.pair_logloss_gradient(s[a], s[index], False) / self.pair_prob(a, index)
		# if gradient is not in rel_found or irr_found, gradient estimate is zero
		return gradient

	def compute_pot_surr(self, s, i):
		# i is the weak learner index
		rel_found = self.topkRel
		irr_found = self.topkNOTRel
		potential = 0.0
		for a in rel_found:
			for b in irr_found:
				pair_pot_val = self.pair_pot_surr_manager(self.num_wls-i, s[a], s[b])
				potential += pair_pot_val / self.pair_prob(a, b)
		return potential

	def compute_cost(self, s, i):
		''' Compute cost matrix
		Args:
			s (list): Current state
			i (int): Weak learner index
		Return:
			(numpy.ndarray) Cost matrix
		'''
		m = self.num_classes
		s = np.array(s)
		if self.loss == 'logistic':
			return np.asarray([
				self.compute_logloss_gradient_estimate(s, index)
										for index in range(m)
								])
		elif self.loss =='zero_one':
			ret = np.zeros(m)
			for l in xrange(m):
				e = np.zeros(m)
				e[l] = 1
				ret[l] = self.compute_pot_surr(s+e, i+1)
			ret = ret - min(ret)
			return ret
		else:
			raise ValueError()

	def get_logistic_grad(self, i):
		''' Compute gradient vector for logistic loss
		'''
		s = self.expert_votes_mat[i, :]
		x = self.wl_preds[i, :]
		rel_found = self.topkRel
		irr_found = self.topkNOTRel
		gradient = 0.0
		for a in rel_found:
			for b in irr_found:
				pairwise_score = x[b] - x[a]
				gradient += pairwise_score * utils.pair_logloss_gradient_scalar(s[a], s[b]) / self.pair_prob(a, b)
		return gradient

	def get_lr(self):
		''' Get learning rate
		Args:
			i (int): Weak learner index
		Return:
			(float): Learning rate
		'''
		t = self.num_data
		m = self.num_classes
		return 8 * self.rho * np.sqrt(2) / (m**2 * np.sqrt(t))

	def update_alpha(self, alpha, i):
		''' Update the weight alpha
		Args:
			c (ndarray): Cost vector
			i (int): Weak learner index
			alpha (float): Weight
		Return:
			(float): updated alpha
		'''
		if self.loss == 'zero_one':
			return 1
		else:
			grad = self.get_logistic_grad(i)
			lr = self.get_lr()
			return max(-2, min(2, alpha - lr*grad))

	def predict(self, X, verbose=False):
		'''Runs the entire prediction procedure, updating internal tracking
		of wl_preds and Yhat, and returns the randomly chosen Yhat

		Args:
			X (list): A list of the covariates of the current data point.
					  Float for numerical, string for categorical. Categorical
					  data must have been in initial dataset

		Returns:
			Yhat (string): The final class prediction
		'''

		self.X = np.array(X)

		# Initialize values

		expert_votes = np.zeros(self.num_classes)
		expert_votes_mat = np.empty([self.num_wls, self.num_classes])

		for i in xrange(self.num_wls):
			data_indices = self.data_indices[i]
			pred_inst = self.make_cov_instance(self.X[data_indices])
			# Get our new week learner prediction and our new expert prediction
			pred_probs = \
				self.weaklearners[i].distribution_for_instance(pred_inst)
			pred_probs = np.array(pred_probs)
			# if self.loss == 'zero_one':
			#     _max = max(pred_probs)
			#     tmp = [i for i in range(self.num_classes) if \
			#                 pred_probs[i] > _max-0.001]
			#     label = np.random.choice(tmp, 1)[0]
			#     pred_probs = np.zeros(self.num_classes)
			#     pred_probs[label] = 1
			self.wl_preds[i,:] = pred_probs
			if verbose is True:
				print i, pred_probs
			expert_votes += self.wl_weights[i] * pred_probs
			expert_votes_mat[i,:] = expert_votes

		if self.loss == 'zero_one':
			pred_index = -1
		else:
			tmp = self.expert_weights/sum(self.expert_weights)
			pred_index = np.random.choice(range(self.num_wls), p=tmp)
		self.Yhat = expert_votes_mat[pred_index,:]
		self.topkYhat = utils.topk(self.Yhat, self.k)
		self.expert_votes_mat = expert_votes_mat

		self.Ytilde = self.generate_random_ranking(self.Yhat, self.topkYhat)
		self.topkYtilde = utils.topk(self.Ytilde, self.k)
		return self.Ytilde

	def update(self, topkRel, X=None, verbose=False):
		# TODO: change this to use topkY feedback set
		'''Runs the entire updating procedure, updating interal
		tracking of wl_weights and expert_weights
		Args:
			X (list): A list of the covariates of the current data point.
					  Float for numerical, string for categorical. Categorical
					  data must have been in initial dataset. If not given
					  the last X used for prediction will be used.
			Y (string): The true class
		'''

		assert(topkRel.issubset(self.topkYtilde))

		if X is None:
			X = self.X

		self.X = np.array(X)
		self.topkRel = topkRel
		self.topkNOTRel = self.topkYtilde.difference(topkRel)
		if len(self.topkRel) == 0 or len(self.topkNOTRel) == 0:
			# without relevant and irrelevant labels, pass
			return
		self.num_data +=1
		expert_votes = np.zeros(self.num_classes)
		cost_vec = self.compute_cost(expert_votes, 0)

		for i in xrange(self.num_wls):
			alpha = self.wl_weights[i]
			w = self.weight_consts[i]
			# if self.loss == 'zero_one':
			#     w *= 5
			data_indices = self.data_indices[i]

			# if self.num_data >= 500:
			# 	print(i)
			# 	# print(np.round(cost_vec, 3))
			# 	print(np.round(self.wl_preds[i], 2))

			_max = max(cost_vec)
			for l in topkRel:
				full_inst = self.make_full_instance(self.X[data_indices], l)
				full_inst.set_weight(w*(_max - cost_vec[l]))
				self.weaklearners[i].update_classifier(full_inst)
			if False:
				print i, _max

			# updating the quality weights and weighted vote vector
			expert_votes = self.expert_votes_mat[i,:]
			cost_vec = self.compute_cost(expert_votes, i+1)

			# adaptive algorithm stuff
			self.wl_weights[i] = \
								self.update_alpha(alpha, i)
			if self.loss == 'logistic':
				self.expert_weights[i] *= \
								np.exp(-self.compute_rankloss_estimate(expert_votes) \
								* self.exp_step_size)
		# assert self.num_data <= 50

		self.expert_weights = self.expert_weights/sum(self.expert_weights)

	def initialize_dataset(self, data_source):
		filepath = utils.get_filepath(data_source, 'train')
		self.dataset = utils.open_dataset(filepath)
		self.slc_dataset = utils.open_slc_dataset(filepath, self.num_covs)

		self.num_classes = self.slc_dataset.num_classes()

	def gen_weaklearners(self, num_wls, min_conf = 0.00001, max_conf = 0.9,
											min_grace = 1, max_grace = 10,
											min_tie = 0.001, max_tie = 1,
											min_weight = 10, max_weight = 200,
											seed = 1):
		''' Generate weak learners.
		Args:
			num_wls (int): Number of weak learners
			Other args (float): Range to randomly generate parameters
			seed (int): Random seed
		Returns:
			It does not return anything. Generated weak learners are stored in
			internal variables.
		'''
		np.random.seed(seed)
		self.num_wls = num_wls
		self.weaklearners = [HoeffdingTree() for _ in range(num_wls)]

		min_conf = np.log10(min_conf)
		max_conf = np.log10(max_conf)
		min_tie = np.log10(min_tie)
		max_tie = np.log10(max_tie)

		for wl in self.weaklearners:
			wl._header = self.slc_dataset
			conf = 10 ** np.random.uniform(low=min_conf, high=max_conf)
			grace = np.random.uniform(low=min_grace, high=max_grace)
			tie = 10**np.random.uniform(low=min_tie, high=max_tie)
			wl.set_split_confidence(conf)
			wl.set_grace_period(grace)
			wl.set_hoeffding_tie_threshold(tie)

		self.wl_edges = np.zeros(num_wls)
		self.expert_weights = np.ones(num_wls)/num_wls
		if self.loss == 'zero_one':
			self.wl_weights = np.ones(num_wls)
		else:
			self.wl_weights = np.zeros(num_wls)
		self.wl_preds = np.zeros((num_wls, self.num_classes))
		self.expert_preds = np.zeros(num_wls)

		self.weight_consts = [np.random.uniform(low=min_weight, high=max_weight)
								for _ in range(num_wls)]

		self.data_indices = []
		data_len = self.dataset.num_attributes() - 1
		if data_len <= self.num_covs:
			for _ in xrange(num_wls):
				self.data_indices.append(range(data_len))
		else:
			for _ in xrange(num_wls):
				data_indices = np.random.choice(range(data_len),
													self.num_covs,
													replace=False)
				self.data_indices.append(data_indices)


	def get_cum_error(self):
		return self.cum_error

	def get_dataset(self):
		return self.dataset

	def set_dataset(self, dataset):
		self.dataset = dataset

	def set_num_wls(self, n):
		self.num_wls = n

	def set_class_index(self, class_index):
		self.class_index = class_index

	def set_num_classes(self, num_classes):
		self.num_classes = num_classes

	def set_exp_step_size(self, exp_step_size):
		self.exp_step_size = exp_step_size
