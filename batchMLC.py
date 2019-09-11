import csv
import numpy as np
import copy
from hoeffdingtree import *
import utils
import time
import arff
from sklearn import svm, tree

class AdaBoostMM:
    
    def __init__(self, data_source):
        self.initialize_dataset(data_source)
        self.class_index = 0

    ########################################################################

    def gen_weaklearners(self, num_wls, min_conf = 0.00001, max_conf = 0.9, 
                                              min_grace = 1, max_grace = 10,
                                              min_tie = 0.001, max_tie = 1,
                                              min_weight = 1, max_weight = 100, 
                                              seed = 123):
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
        self.weaklearners = [HoeffdingTree() for _ in xrange(num_wls)]

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
            
        self.weight_consts = [np.random.uniform(low=min_weight, high=max_weight)
                                for _ in range(num_wls)]

    def compute_cost(self, i, cum_votes):
        ret = np.zeros((self.num_data, self.num_classes))
        for t in xrange(self.num_data):
            s = cum_votes[t]
            Y = self.class_sets[t]
            Y_complement = set(range(self.num_classes)).difference(Y)
            normalize_const = len(Y)*len(Y_complement)
            sum_correct = sum([np.exp(-s[l]) for l in Y])
            sum_wrong = sum([np.exp(s[r]) for r in Y_complement])
            if normalize_const != 0:
                sum_correct /= normalize_const
                sum_wrong /= normalize_const
            for l in xrange(self.num_classes):
                if l in Y:
                    ret[t, l] = -sum_wrong * np.exp(-s[l])
                else:
                    ret[t, l] = sum_correct * np.exp(s[r]) 
        return ret

    def get_weights(self, cost_mat):
        ret = []
        for t in xrange(self.num_data):
            Y = self.class_sets[t]
            weight = -sum(cost_mat[t, list(Y)])
            ret.append(weight)
        ret = np.array(ret)
        ret /= max(ret)
        return ret

    def get_slc_labels(self, cost_mat):
        ret = []
        for t in xrange(self.num_data):
            cost_vec = cost_mat[t]
            _min = min(cost_vec)
            tmp = [l for l in xrange(self.num_classes) if cost_vec[l] == _min]
            ret.append(str(np.random.choice(tmp)))
        return ret

    def insert_slc_label(self, t, label):
        inst = self.slc_dataset.instance(t)
        ret = copy.deepcopy(inst)
        ret.set_value(self.class_index, int(label))
        return ret

    def train_weaklearner(self, i, cost_mat):
        const = max(-cost_mat[cost_mat<0])
        const = self.weight_consts[i] / const
        for t in xrange(self.num_data):
            for l in self.class_sets[t]:
                inst = self.insert_slc_label(t, l)
                w = -cost_mat[t,l] * const
                inst.set_weight(w)
                self.weaklearners[i].update_classifier(inst)
    
    def wl_predict(self, i, cost_mat):
        ret = []
        for t in xrange(self.num_data):
            inst = self.dataset.instance(t)
            pred_probs = self.weaklearners[i].distribution_for_instance(inst)
            _max = max(pred_probs)
            tmp = [l for l in xrange(self.num_classes) if pred_probs[l] == _max]
            ret.append(str(np.random.choice(tmp)))
        return ret

    def update_cum_votes(self, i, cum_votes, cost_mat):
        preds = self.wl_predict(i, cost_mat)
        _sum = 0.0
        for t, l in enumerate(preds):
            _sum += cost_mat[t, l]
        if i == 0:
            cum_exp_loss = self.num_data
        else:
            cum_exp_loss = self.num_data * self.exp_losses[i-1]
        gamma = -_sum / cum_exp_loss
        alpha = 0.5 * np.log((1+gamma)/(1-gamma))
        self.wl_weights.append(alpha)
        for t, l in enumerate(preds):
            cum_votes[t, l] += alpha
        return cum_votes

    def record_losses(self, cum_votes):
        exp_sum = 0.0
        rank_sum = 0.0
        for t in xrange(self.num_data):
            s = cum_votes[t]
            Y = self.class_sets[t]
            exp_sum += utils.exp_loss(s, Y)
            rank_sum += utils.rank_loss(s, Y)
        self.exp_losses.append(exp_sum/self.num_data)
        self.rank_losses.append(rank_sum/self.num_data)

    def save_class_sets(self):
        self.class_sets = []
        for t in xrange(self.num_data):
            Y_str = self.dataset.instance(t).class_value()
            Y_set = utils.str_to_set(Y_str)
            self.class_sets.append(Y_set)
        self.num_classes = len(Y_str)

    def fit(self):
        self.exp_losses = []
        self.rank_losses = []
        self.num_data = self.dataset.num_instances()
        self.wl_weights = []
        self.save_class_sets()
        cum_votes = np.zeros((self.num_data, self.num_classes))
        for i in xrange(self.num_wls):
            cost_mat = self.compute_cost(i, cum_votes)
            self.train_weaklearner(i, cost_mat)
            cum_votes = self.update_cum_votes(i, cum_votes, cost_mat)
            self.record_losses(cum_votes)
            print 'Iteration', i, 'Exp:', round(self.exp_losses[-1], 4), 'Rank:', round(self.rank_losses[-1],4)
    
    def get_num_wls(self):
        return self.num_wls

    def initialize_dataset(self, data_source):
        filepath = utils.get_filepath(data_source, 'train')
        self.dataset = utils.open_dataset(filepath)
        self.slc_dataset = utils.open_slc_dataset(filepath)

    def get_dataset(self):
        return self.dataset

    def get_slc_dataset(self):
        return self.slc_dataset

    def get_exp_losses(self):
        return self.exp_losses

    def get_rank_losses(self):
        return self.rank_losses

class AdaBoostMMsvm(AdaBoostMM):
    

    def __init__(self, data_source, num_covs=20):
        self.class_index = 0
        self.process_data(data_source)
        self.data_indices = []
        self.num_covs=num_covs

    ########################################################################

    def process_data(self, data_source):
        fp = utils.get_filepath(data_source, 'train')
        data = arff.load(open(fp, 'rb'))
        class_index, header, data_types = utils.parse_attributes(data)
        self.data = []
        self.class_sets = []
        for row in data['data']:
            self.data.append(row[:class_index])
            label = reduce(lambda x,y:x+y, row[class_index:])
            self.class_sets.append(utils.str_to_set(label))
        self.num_data = len(self.class_sets)
        self.num_classes = len(label)
        self.data = np.array(self.data)

        fp = utils.get_filepath(data_source, 'test')
        test_data = arff.load(open(fp, 'rb'))
        self.test_data = []
        self.test_class_sets = []
        for row in test_data['data']:
            self.test_data.append(row[:class_index])
            label = reduce(lambda x,y:x+y, row[class_index:])
            self.test_class_sets.append(utils.str_to_set(label))
        self.test_num_data = len(self.test_class_sets)
        self.test_data = np.array(self.test_data)


    def gen_weaklearners(self, num_wls, min_C=0.01, max_C=100, 
                                    min_gamma=0.01, max_gamma=100, seed=123):
        np.random.seed(seed)
        self.num_wls = num_wls
        min_C = np.log10(min_C)
        max_C = np.log10(max_C)
        min_gamma = np.log10(min_gamma)
        max_gamma = np.log10(max_gamma)
        self.weaklearners = []
        for i in xrange(num_wls):
            kernel = str(np.random.choice(['rbf', 'linear']))
            C = 10 ** np.random.uniform(min_C, max_C)
            gamma = 10 ** np.random.uniform(min_gamma, max_gamma)
            wl = svm.SVC(kernel=kernel, C=C, gamma=gamma)
            self.weaklearners.append(wl)

    def train_weaklearner(self, i, cost_mat):
        # data = []
        # labels = []
        # weights = []
        # for t in xrange(self.num_data):
        #     for l in self.class_sets[t]:
        #         data.append(self.data[t])
        #         labels.append(l)
        #         weights.append(-cost_mat[t,l])
        # self.sample_weights = weights
        # self.weaklearners[i].fit(data, labels, weights)
        if len(self.data[0]) < self.num_covs:
            data_indices = np.arange(len(self.data[0]))
        else:
            data_indices = np.random.choice(range(len(self.data[0])),
                                            self.num_covs, replace=False)
        self.data_indices.append(data_indices)

        data = []
        labels = []
        weights = []
        for t in xrange(self.num_data):
            _max = np.max(cost_mat[t])
            for l in xrange(self.num_classes):
                data.append(self.data[t, data_indices])
                labels.append(l)
                weights.append(-cost_mat[t,l]+_max)
        self.sample_weights = weights
        self.weaklearners[i].fit(data, labels, weights)
    
    def update_cum_votes(self, i, cum_votes, cost_mat):
        preds = self.weaklearners[i].predict(self.data[:, self.data_indices[i]])
        _sum = 0.0
        for t, l in enumerate(preds):
            _sum += cost_mat[t, l]
        if i == 0:
            cum_exp_loss = self.num_data
        else:
            cum_exp_loss = self.num_data * self.exp_losses[i-1]
        gamma = -_sum / cum_exp_loss
        alpha = 0.5 * np.log((1+gamma)/(1-gamma))
        self.wl_weights.append(alpha)
        for t, l in enumerate(preds):
            cum_votes[t, l] += alpha
        return cum_votes, gamma

    def fit(self):
        self.exp_losses = []
        self.rank_losses = []
        self.wl_weights = []
        cum_votes = np.zeros((self.num_data, self.num_classes))
        for i in xrange(self.num_wls):
            cost_mat = self.compute_cost(i, cum_votes)
            self.train_weaklearner(i, cost_mat)
            cum_votes, edge = self.update_cum_votes(i, cum_votes, cost_mat)
            self.record_losses(cum_votes)
            print 'Iteration', i, 'Exp:', round(self.exp_losses[-1], 4), 'Rank:', round(self.rank_losses[-1],4), 'Edge:', edge


    def get_test_results(self, num_wls=0):
        if num_wls == 0:
            num_wls = self.num_wls
        cum_votes = np.zeros((self.test_num_data, self.num_classes))
        for i in xrange(num_wls):
            alpha = self.wl_weights[i]
            wl = self.weaklearners[i]
            preds = wl.predict(self.test_data[:,self.data_indices[i]])
            for t, l in enumerate(preds):
                cum_votes[t, l] += alpha

        exp_sum = 0.0
        rank_sum = 0.0
        for t in xrange(self.test_num_data):
            s = cum_votes[t]
            Y = self.test_class_sets[t]
            exp_sum += utils.exp_loss(s, Y)
            rank_sum += utils.rank_loss(s, Y)
        exp_sum /= self.test_num_data
        rank_sum /= self.test_num_data
        return exp_sum, rank_sum

class AdaBoostMMtree(AdaBoostMMsvm):
    def gen_weaklearners(self, num_wls, min_depth=2, max_depth=20, 
                                        min_split=5, max_split=20,
                                        min_leaf=1, max_leaf=10,
                                        seed=123):
        np.random.seed(seed)
        self.num_wls = num_wls
        self.weaklearners = []
        for i in xrange(num_wls):
            d = np.random.choice(range(min_depth, max_depth+1))
            split = np.random.choice(range(min_split, max_split+1))
            leaf = np.random.choice(range(min_leaf, max_leaf+1))
            wl = tree.DecisionTreeClassifier(max_depth=d,
                                                min_samples_split=split,
                                                min_samples_leaf=leaf)
            self.weaklearners.append(wl)

class AdaBoostMMtreeBinary(AdaBoostMMtree):
    def train_weaklearner(self, i, cost_mat):
        if len(self.data[0]) < self.num_covs:
            data_indices = np.arange(len(self.data[0]))
        else:
            data_indices = np.random.choice(range(len(self.data[0])),
                                            self.num_covs, replace=False)
        self.data_indices.append(data_indices)

        base_class = i % self.num_classes
        data = self.data[:,data_indices]
        labels = []
        weights = []
        for t in xrange(self.num_data):
            c = cost_mat[t,base_class]
            weights.append(abs(c))
            if c < 0:
                labels.append(1)
            else:
                labels.append(0)
        self.sample_weights = weights
        self.weaklearners[i].fit(data, labels, weights)

    def transform_binary_preds(self, i, bin_preds):
        base_class = i % self.num_classes
        preds = []
        pos = np.zeros(self.num_classes)
        pos[base_class] = 1
        neg = np.ones(self.num_classes) / (self.num_classes-1)
        neg[base_class] = 0
        for l in bin_preds:
            if l == 1:
                preds.append(pos)
            else:
                preds.append(neg)
        return preds

    def update_cum_votes(self, i, cum_votes, cost_mat):
        bin_preds = self.weaklearners[i].predict(
                            self.data[:,self.data_indices[i]])
        preds = self.transform_binary_preds(i, bin_preds)
        _sum = 0.0
        for t in xrange(self.num_data):
            _sum += np.dot(cost_mat[t], preds[t])
        if i == 0:
            cum_exp_loss = self.num_data
        else:
            cum_exp_loss = self.num_data * self.exp_losses[i-1]
        gamma = -_sum / cum_exp_loss
        alpha = 0.5 * np.log((1+gamma)/(1-gamma))
        self.wl_weights.append(alpha)
        for t in xrange(self.num_data):
            cum_votes[t] += alpha*preds[t]
        return cum_votes, gamma    

    def get_test_results(self, num_wls=0):
        if num_wls == 0:
            num_wls = self.num_wls
        cum_votes = np.zeros((self.test_num_data, self.num_classes))
        for i in xrange(num_wls):
            alpha = self.wl_weights[i]
            wl = self.weaklearners[i]
            bin_preds = wl.predict(self.test_data[:,self.data_indices[i]])
            preds = self.transform_binary_preds(i, bin_preds)
            for t in xrange(self.test_num_data):
                cum_votes[t] += alpha*preds[t]

        exp_sum = 0.0
        rank_sum = 0.0
        for t in xrange(self.test_num_data):
            s = cum_votes[t]
            Y = self.test_class_sets[t]
            exp_sum += utils.exp_loss(s, Y)
            rank_sum += utils.rank_loss(s, Y)
        exp_sum /= self.test_num_data
        rank_sum /= self.test_num_data
        return exp_sum, rank_sum

class AdaBoostMMtreeProba(AdaBoostMMtree):
    def update_cum_votes(self, i, cum_votes, cost_mat):
        preds = self.weaklearners[i].predict_proba(
                            self.data[:,self.data_indices[i]])
        _sum = 0.0
        for t in xrange(self.num_data):
            _sum += np.dot(cost_mat[t], preds[t])
        if i == 0:
            cum_exp_loss = self.num_data
        else:
            cum_exp_loss = self.num_data * self.exp_losses[i-1]
        gamma = -_sum / cum_exp_loss
        alpha = 0.5 * np.log((1+gamma)/(1-gamma))
        self.wl_weights.append(alpha)
        for t in xrange(self.num_data):
            cum_votes[t] += alpha*preds[t]
        return cum_votes, gamma

    def get_test_results(self, num_wls=0):
        if num_wls == 0:
            num_wls = self.num_wls
        cum_votes = np.zeros((self.test_num_data, self.num_classes))
        for i in xrange(num_wls):
            alpha = self.wl_weights[i]
            wl = self.weaklearners[i]
            preds = wl.predict_proba(self.test_data[:,self.data_indices[i]])
            for t in xrange(self.test_num_data):
                cum_votes[t] += alpha*preds[t]

        exp_sum = 0.0
        rank_sum = 0.0
        for t in xrange(self.test_num_data):
            s = cum_votes[t]
            Y = self.test_class_sets[t]
            exp_sum += utils.exp_loss(s, Y)
            rank_sum += utils.rank_loss(s, Y)
        exp_sum /= self.test_num_data
        rank_sum /= self.test_num_data
        return exp_sum, rank_sum

class OStree(AdaBoostMMtree):
    def __init__(self, data_source, gamma=0.01, M=100, num_covs=20):
        self.class_index = 0
        self.process_data(data_source)
        self.gamma = gamma
        self.M = M
        self.set_baseline()
        self.num_covs = num_covs
        self.data_indices = []

    def set_baseline(self):
        self.baseline = []
        for t in xrange(self.num_data):
            Y = self.class_sets[t]
            x = (1 - self.gamma*len(Y))/float(self.num_classes)
            b = x * np.ones(self.num_classes)
            b[list(Y)] += self.gamma
            self.baseline.append(b)
        self.baseline = np.array(self.baseline)
        if self.baseline.min() < 0 or self.baseline.max() > 1:
            raise NameError('Baseline got ineligible values.')
        
    def compute_cost(self, i, cum_votes):
        ret = np.zeros((self.num_data, self.num_classes))
        for t in xrange(self.num_data):
            s = cum_votes[t]
            b = self.baseline[t]
            Y = self.class_sets[t]
            for l in xrange(self.num_classes):
                s[l] += 1
                ret[t, l] = utils.mc_potential(self.num_wls-i, b, Y, self.M, s)
                s[l] -= 1
            ret[t] -= sum(ret[t])/float(self.num_classes)
        return ret

    def update_cum_votes(self, i, cum_votes, cost_mat):
        preds = self.weaklearners[i].predict(self.data[:, self.data_indices[i]])
        alpha = 1
        self.wl_weights.append(alpha)
        _sum = 0.0
        for t, l in enumerate(preds):
            cum_votes[t, l] += alpha
            _sum += cost_mat[t, l]
        return cum_votes, 2*_sum/abs(cost_mat).sum()

class AdaBoostMMUnivExp(AdaBoostMMtree):
    def record_losses(self, cum_votes):
        exp_sum = 0.0
        rank_sum = 0.0
        for t in xrange(self.num_data):
            s = cum_votes[t]
            Y = self.class_sets[t]
            exp_sum += utils.univ_exp_loss(s, Y)
            rank_sum += utils.rank_loss(s, Y)
        self.exp_losses.append(exp_sum/self.num_data)
        self.rank_losses.append(rank_sum/self.num_data)

    def compute_cost(self, i, cum_votes):
        ret = np.zeros((self.num_data, self.num_classes))
        for t in xrange(self.num_data):
            s = np.array(cum_votes[t])
            Y = self.class_sets[t]
            Y_complement = set(range(self.num_classes)).difference(Y)
            normalize_const = len(Y)*len(Y_complement)
            s[list(Y)] *= -1
            ret[t] = np.exp(s)
            ret[t, list(Y)] *= -1
            if normalize_const != 0:
                ret[t] /= normalize_const
            # ret[t] -= sum(np.abs(ret[t]))/float(self.num_classes)
        return ret

class AdaBoostMMUnivLogistic(AdaBoostMMtree):
    def record_losses(self, cum_votes):
        exp_sum = 0.0
        rank_sum = 0.0
        for t in xrange(self.num_data):
            s = cum_votes[t]
            Y = self.class_sets[t]
            exp_sum += utils.univ_logistic_loss(s, Y)
            rank_sum += utils.rank_loss(s, Y)
        self.exp_losses.append(exp_sum/self.num_data)
        self.rank_losses.append(rank_sum/self.num_data)

    def compute_cost(self, i, cum_votes):
        ret = np.zeros((self.num_data, self.num_classes))
        for t in xrange(self.num_data):
            s = np.array(cum_votes[t])
            Y = self.class_sets[t]
            Y_complement = set(range(self.num_classes)).difference(Y)
            normalize_const = len(Y)*len(Y_complement)
            s[list(Y_complement)] *= -1
            ret[t] = 1/(1+np.exp(s))
            ret[t, list(Y)] *= -1
            if normalize_const != 0:
                ret[t] /= normalize_const
            # ret[t] -= sum(np.abs(ret[t]))/float(self.num_classes)
        return ret