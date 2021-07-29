import numpy as np
from math import log2
from scipy.special import rel_entr
import scipy.io

# calculate the kl divergence
def kl_divergence(p, q):
	vec = rel_entr(p, q)
	kl_div = np.sum(np.ma.masked_invalid(vec).compressed())

	return kl_div

if __name__ == '__main__':
	perturb_base = "test_rot_of_"
	rot_angle = 45

	origin = np.load("test0.npy")
	origin_pert = np.load(perturb_base + str(rot_angle) + "0.npy")
	perturb = np.zeros([180, 512], dtype=float)
	DLpert = np.zeros([1, 60], dtype=float)
	for i in range(60):
		p = origin[i]
		q = origin_pert[i]
		DLpert[:,i] = min(kl_divergence(p,q), kl_divergence(q, p))

	clean0 = np.load("test1.npy")
	clean0_pert = np.load(perturb_base + str(rot_angle) + "1.npy")
	clear0 = np.zeros([180, 512], dtype=float)
	DLclean = np.zeros([1, 60], dtype=float)
	for i in range(60):
		p = clean0[i]
		q = clean0_pert[i]
		DLclean[:,i] = min(kl_divergence(p,q), kl_divergence(q, p))

	clean1 = np.load("test2.npy")
	clean1_pert = np.load(perturb_base + str(rot_angle) + "2.npy")
	clear1 = np.zeros([180, 512], dtype=float)
	DLclean_cloudy = np.zeros([1, 60], dtype=float)
	for i in range(60):
		p = clean1[i]
		q = clean1_pert[i]
		DLclean_cloudy[:,i] = min(kl_divergence(p,q), kl_divergence(q, p))

	print(clear1.shape)
	scipy.io.savemat('kl_clear' +perturb_base + str(rot_angle) + '.mat', dict(DLpert=DLpert, DLclean=DLclean))
	scipy.io.savemat('kl_cloudy'+perturb_base + str(rot_angle) +'.mat', dict(DLpert=DLpert, DLclean=DLclean_cloudy))

