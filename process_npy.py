import numpy as np
import matplotlib.pyplot as plt

for r_alpha in [0.0]:

	# Read in and get data for outliers (bad images from Site A)
	sample_out_dir = "./samples/50ep/out/50_" + str(r_alpha) + "/"
	out_d = np.squeeze(np.load(sample_out_dir + "d_values.npy"))
	out_d_of_g = np.squeeze(np.load(sample_out_dir + "d_of_g_values.npy"))

	# Read in and get data for inliers (good images from Site A)
	sample_in_dir = "./samples/50ep/in/50_" + str(r_alpha) + "/"
	in_d = np.squeeze(np.load(sample_in_dir + "d_values.npy"))
	in_d_of_g = np.squeeze(np.load(sample_in_dir + "d_of_g_values.npy"))

	# Normalize the straight D data between [0, 1]
	max_val = max(np.max(out_d), np.max(in_d))
	min_val = min(np.min(out_d), np.min(in_d))
	#out_d = np.array([(x - min_val)/(max_val - min_val) for x in out_d])
	#in_d = np.array([(x - min_val)/(max_val - min_val) for x in in_d])

	bins = np.linspace(min_val, max_val, 100)
	plt.hist(out_d, bins, alpha=0.5, label="outliers", density=True)
	plt.hist(in_d, bins, alpha=0.5, label="inliers", density=True)
	plt.legend(loc="upper right")
	plt.savefig(str(r_alpha) + "sample_d.jpg")
	plt.close()

	# Normalize the D of G data between [0, 1]
	max_val = max(np.max(out_d_of_g), np.max(in_d_of_g))
	min_val = min(np.min(out_d_of_g), np.min(in_d_of_g))
	#out_d_of_g = np.array([(x - min_val)/(max_val - min_val) for x in out_d_of_g])
	#in_d_of_g = np.array([(x - min_val)/(max_val - min_val) for x in in_d_of_g])

	bins = np.linspace(min_val, max_val, 100)
	plt.hist(out_d_of_g, bins, alpha=0.5, label="outliers", density=True)
	plt.hist(in_d_of_g, bins, alpha=0.5, label="inliers", density=True)
	plt.legend(loc="upper right")
	plt.savefig(str(r_alpha) + "sample_d_of_g.jpg")
	plt.close()

	threshs = np.linspace(min_val, max_val, 100)
	for thresh in threshs:

		TP = (in_d_of_g >= thresh).sum()
		FN = (in_d_of_g < thresh).sum()

		TN = (out_d_of_g < thresh).sum()
		FP = (out_d_of_g >= thresh).sum()

		if TP/(TP+FN) >= 0.85 and TN/(FP+TN) >= 0.85:
			
			print(thresh)
			print(TP/(TP+FN), FN/(TP+FN))
			print(FP/(FP+TN), TN/(FP+TN))
			print()
			print(TP, FN)
			print(FP, TN)
			print()