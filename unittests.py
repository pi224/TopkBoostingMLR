import utils
from topkMLC import TopkAdaBoostOLM
import numpy as np

def test_pair_prob():
	data_source = 'mediamill_reduced'
	k = 97
	rho = 0.5
	d = 2
	A = TopkAdaBoostOLM(data_source=data_source, k=k, rho=rho, d=d)
	A.topkYhat = set(range(k))
	A.Yhat = np.flip(np.arange(101))

	label_a = 98
	label_b = 99

	M = 5000000
	counts = 0.0
	expected = A.pair_prob(label_a, label_b)
	print('beginning monte-carlo')
	for i in np.arange(M):
		rand_ranking = A.generate_random_ranking(A.Yhat, A.topkYhat)
		topk_rand_ranking = utils.topk(rand_ranking, k)
		if label_a in topk_rand_ranking and label_b in topk_rand_ranking:
			counts += 1.0
	result = counts / M
	print('num_classes: ', A.num_classes)
	print('estimation: ',  result)
	print('calculated: ', expected)

if __name__ == '__main__':
	test_pair_prob()
