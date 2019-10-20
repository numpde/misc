
# RA, 2019-08-05

import pandas as pd
import numpy as np
import os
import time
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

def plot_histo(df: pd.DataFrame):
	#df = df.loc[df['ab_slot1_variant'] == '']

	(fig, ax) = plt.subplots()
	fig: plt.Figure
	ax: plt.Axes

	nsamples = 100000

	ov: pd.DataFrame
	ov = df.loc[df['fp'] == True, ['o', 'v']]

	df = df.loc[df['v'] > 0].dropna(axis=1).sort_values(by='v')
	c = df['o']
	v = df['v']
	bins = np.logspace(0, np.log10(max(v)), 15)
	binned_c = [c[(a < v) & (v <= b)] for (a, b) in zip(bins, bins[1:])]
	binned_v = [v[(a < v) & (v <= b)] for (a, b) in zip(bins, bins[1:])]

	# ax.hist(binned_c[2] > 0, bins=20, density=True, alpha=1)

	ax.set_xscale('log')
	ax.set_yscale('log')

	import xgboost as xgb
	# df['v'] = np.random.permutation(df['v']) # Destroys the pattern
	df = df.sort_values(by='v')
	model0 = xgb.XGBRegressor(n_estimators=100, objective='count:poisson', n_jobs=6).fit(df[['v']], df[['o']])
	o_pred = model0.predict(df[['v']])
	ax.scatter(v, o_pred, s=2, alpha=0.2)
	xx = np.linspace(min(v), max(v), 100)
	ax.plot(xx, 1 - np.exp(-xx / 66))
	ax.grid()
	plt.show()
	exit(39)
	y_rate_pred = pd.Series(model0.predict(X), index=y.index, name='rate')

	for (cbin, vbin) in zip(binned_c, binned_v):
		if not len(cbin):
			continue
		[vmin, vmax] = [np.min(vbin), np.max(vbin)]
		theta = sum(cbin > 0) / len(cbin)
		lam = np.mean(cbin) / sum(cbin)
		print("{} <= Views <= {},  #Orders = {},  E[CR] = {},  E[Volume / TotVol] = {}".format(vmin, vmax, len(cbin), theta, lam))
		ax.plot([vmin, vmax], [theta, theta], c='r')
		ax.plot([vmin, vmax], [lam, lam], c='b')

	plt.show()
	exit(39)


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
