
# coding: utf-8

# In[1]:


import os
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random
import copy
import arff
from tqdm import tqdm
import time

# Change current working directory for imports
cwd = os.getcwd()
FOLDER_NAME = 'MLRBoostingWithVFDT'
if FOLDER_NAME not in cwd:
    new_wd = os.path.join(cwd, FOLDER_NAME)
    os.chdir(new_wd)

from hoeffdingtree import *
from topkMLC import TopkAdaBoostOLM
import utils

HOME_DIR = os.getcwd()
DATA_DIR = HOME_DIR.replace('codes/'+FOLDER_NAME, 'data')

full_information_history = {'yeast': 0.1728, 'emotions': 0.1657, 'mediamill': 0.0472, 'scene':0.0720}
# --------------------------------------------------------------
# USING OLD SURROGATE POTENTIAL FUNCTION
# topk_history = {'emotions': 0.198, 'scene': 0.1267, 'yeast': 0.23}

# topk_N = {'emotions': 50, 'scene': 70, 'yeast': 30, 'mediamill': 1}
# topk_gamma = {'emotions': 0.3, 'scene': 0.3, 'yeast': 0.3, 'mediamill':0.3}
# topk_rho = {'emotions': .02, 'scene': .02, 'yeast': .04}
# topk_d = {'emotions': 2, 'scene': 2, 'yeast': 2}
# topk_loops = {'emotions': 6, 'scene': 6, 'yeast': 6}


# ----------------------------------------------------------------
# USING NEW SURROGATE POTENTIAL FUNCTION
new_topk_best =  {'emotions': 0.2152, 'scene': 0.1106, 'yeast':  0.2244}

new_topk_N =     {'emotions': 20, 'scene': 50, 'yeast': 30}
new_topk_gamma = {'emotions': 0.3, 'scene': 0.2, 'yeast': 0.3}
new_topk_k =     {'emotions': 3, 'scene': 3, 'yeast': 3}
new_topk_rho =   {'emotions': .02, 'scene': .02, 'yeast': .02}
new_topk_d =     {'emotions': 2, 'scene': 2, 'yeast': 2}
new_topk_loops = {'emotions': 10, 'scene': 10, 'yeast': 10}

# ================================================================
# ADAPTIVE ALGORITHM
ada_topk_best = {'emotions': 0.2427, 'scene': 0.1216}

ada_topk_N = {'emotions': 70, 'scene': 50}
# all other parameters, we'll inherit from optimal hyperparameters


# In[3]:


data_source = 'yeast'
fp = utils.get_filepath(data_source, 'train')
data = arff.load(open(fp, 'rb'))
class_index, header, data_types = utils.parse_attributes(data)
train_rows = data['data']
fp = utils.get_filepath(data_source, 'test')
data = arff.load(open(fp, 'rb'))
test_rows = data['data']

if data_source == 'mediamill_reduced':
    train_rows = train_rows[0:400]
    test_rows = test_rows[0:100]

loss = 'logistic'
num_wls = 5
num_covs = 20
gamma = 0.3
M = 100
k=3
d=2
rho=0.02
loops = 10

model = TopkAdaBoostOLM(data_source, k, rho, d, num_covs=num_covs, loss=loss, gamma=gamma)
model.M = M
model.gen_weaklearners(num_wls,
                       min_grace=5, max_grace=20,
                       min_tie=0.01, max_tie=0.9,
                       min_conf=0.01, max_conf=0.9,
                       min_weight=3, max_weight=10)
model.verbose = False
start = time.time()

error_history = []
history_counter = 0
total_unnorm_error = 0.0
for row in (train_rows * loops):
    X = row[:class_index]
    Y = row[class_index:]
    pred = model.predict(X)

    topk = utils.topk(pred, k)
    Yset = utils.label_array_to_set(Y)
    topkRel = topk.intersection(Yset)
    model.update(topkRel)

    total_unnorm_error += utils.unnormalized_rank_loss(pred, Yset)
    history_counter += 1.0
    error_history.append(total_unnorm_error / history_counter)

mid = time.time()
cum_error = 0
cum_unnorm_error = 0

model.verbose = True

for row in test_rows:
    X = row[:class_index]
    Y = row[class_index:]
    pred = model.predict(X)

    topk = utils.topk(pred, k)
    Yset = utils.label_array_to_set(Y)
    topkRel = topk.intersection(Yset)
    model.update(topkRel)

    cum_error += utils.rank_loss(pred, Yset)
    cum_unnorm_error += utils.unnormalized_rank_loss(pred, Yset)
    total_unnorm_error += utils.unnormalized_rank_loss(pred, Yset)
    history_counter += 1.0
    error_history.append(total_unnorm_error / history_counter)

end = time.time()
print 'Training time', mid - start
print 'Test time', end - mid
if loss == 'zero_one':
    print 'cache hit percentage:', float(model.cache_hits) / float(model.potential_calls)
print 'Average rank loss', round(cum_error / float(len(test_rows)), 4)
print 'Average unnormalized rank loss', round(cum_unnorm_error / float(len(test_rows)), 4)

plt.plot(error_history)
plt.show()
