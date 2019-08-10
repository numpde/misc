
# RA, 2019-08-05

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.random import RandomState
from joblib import Parallel, delayed
from math import sqrt


PARAM = {
	'data_file': "../dbeaver/crhisto1_201908051439.csv",
	'output_img': "OUTPUT/crhisto1/img/histo.{ext}",
}


def fenumerate(arr):
	arr = list(arr)
	return zip(np.linspace(0, 1, 1 + len(arr))[1:], arr)

def plot_histo(df):
	#df = df.loc[df['ab_slot1_variant'] == '']

	(fig, ax) = plt.subplots()
	fig: plt.Figure
	ax: plt.Axes

	nsamples = 100000

	ov: pd.DataFrame
	ov = df.loc[df['fp'] == True, ['o', 'v']]

	std_o = np.std(df['o'])
	avg_v = np.mean(df['v'])

	def sample_qoi(size):
		[o, v] = ov.sample(size, replace=True, random_state=RandomState()).sum(axis=0).values
		return o / v

	for (f, samplesize) in fenumerate(np.power(4, [3, 4, 5, 6])):
		samples = Parallel(n_jobs=7)(delayed(sample_qoi)(samplesize) for _ in range(nsamples))
		(a, b) = (np.std(samples), (std_o / avg_v / sqrt(samplesize)))
		print(samplesize, np.mean(samples), a, b, (b - a) / a)
		ax.hist(samples, bins='fd', color=(1 - f, 1 - f, 1), label=("Sample size: {}".format(samplesize)))

	ax.legend()

	# fig.savefig(
	# 	PARAM['output_img'].format(ext='png'),
	# 	bbox_inches='tight', pad_inches=0,
	# 	dpi=300
	# )

	plt.show()
	plt.close(fig)


def prep_df(filename):
	df = pd.read_csv(filename, sep=',').drop(
		columns=['domain_userid']
	).rename(
		columns={'ab_slot1_variant': 'fp', 'viewed_product': 'v', 'ordered_variant': 'o'}
	).replace(
		{'fp': {'Control': False, 'Test': True}}
	)

	return df


def main():
	os.makedirs(os.path.dirname(PARAM['output_img']).format(), exist_ok=True)

	df = prep_df(PARAM['data_file'])
	plot_histo(df)


if __name__ == "__main__":
	main()
