
# RA, 2019-08-08

# Caused by:

# [1] https://booking.ai/understanding-mechanisms-of-change-in-online-experiments-at-booking-com-629201ec74ee

# Further references:

# [2]
# Kosuke Imai, Luke Keele, Dustin Tingley, and Teppei Yamamoto
# "Unpacking the Black Box of Causality: Learning about Causal Mechanisms from Experimental and Observational Studies"
# American Political Science Review, 105 (4), 2011, pp. 765-789
# https://doi.org/10.1017/S0003055411000414

# [3]
# Talk slides for [2]
# https://pdfs.semanticscholar.org/196d/8969cf726739b3b0a7b6021e7e2bf5f0862d.pdf

# [4]
# K. Imai, L. Keele, D. Tingley
# "A general approach to causal mediation analysis"
# Psychol Methods., 15 (4), 2010, pp. 309-334
# https://dx.doi.org/10.1037/a0020761
# https://www.researchgate.net/publication/47457932_A_General_Approach_to_Causal_Mediation_Analysis
# https://imai.fas.harvard.edu/research/files/BaronKenny.pdf -- right paper under wrong name



# Quantities of interest in the dataframe

# Treatment:  df['treat']    -- treatment flag
# Mediatah:   df['booked']   -- number of bookings
# Outcome:    df['canceled'] -- cancellations per visitor
# Confoundah: df['business'] -- business traveler flag


import pandas as pd
import numpy as np
from scipy import stats
from numpy.random import RandomState
from itertools import combinations, product, groupby
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import xgboost as xgb

import matplotlib.pyplot as plt


PARAM = {
	'simdatasize': 100000,
	'rs': RandomState(42),

	# Treatment split
	'treat': 0.5,

	# Traveler type split
	'business': 0.4,

	# Booking rates (Poisson lambda)
	'booked': (
		lambda row:
		3 if (row['treat'] and row['business'])
		else 1
	),

	# Cancellation rates (binomial p)
	'is_cancel': (
		lambda row:
		(
			0.14 if row['business'] else 0.07
		)
		if row['booked']
		else np.nan
	),

	# Units for consistent display
	'visitors_per_day': (1e5 / 30),
}


# == HELPERS ==

# https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python/48812729#48812729
def sigdig(x: float, n: int = 5) -> str:
	return '{:g}'.format(float('{:.{n}g}'.format(x, n=n)))


def speak(b: bool):
	return {True: "Yes", False: "No"}[b]


def compose(f, g):
	return (lambda x: f(g(x)))


# Group series, count group size in percent
def pred_perc(y: pd.Series):
	return ", ".join([
		"rate {} occurs {}% of the time".format(sigdig(rate), sigdig(len(list(group)) * 100 / len(y)))
		for (rate, group) in groupby(sorted(round(y, 2)))
	])


# Return a random vector of True/False
def nan_binomial(p, size=None):
	if (0 <= p <= 1):
		return (1 == PARAM['rs'].binomial(n=1, p=p, size=size))
	else:
		return np.nan


# All sublists of list
def powerset(s: list):
	for n in range(len(s) + 1):
		yield from map(list, combinations(s, n))


# == SIMULATED DATA GENERATION ==

def get_simulated_data() -> pd.DataFrame:

	size = PARAM['simdatasize']

	df: pd.DataFrame
	df = pd.DataFrame(data=0, index=range(0, size), columns=['treat', 'business', 'booked', 'is_cancel', 'canceled'])

	for var in ['treat', 'business']:
		df[var] = nan_binomial(p=PARAM[var], size=size)

	for (var, rand) in zip(['booked', 'is_cancel'], [PARAM['rs'].poisson, nan_binomial]):
		df[var] = df.apply(compose(rand, PARAM[var]), axis=1)

	var = 'canceled'
	df[var] = (df['booked'] * df['is_cancel']).fillna(0)

	return df


def show_data_summary(df: pd.DataFrame) -> pd.DataFrame:
	# Summarize by treatment / business flag
	df_reduced = df.groupby(['treat', 'business']).agg(lambda df: df.mean(skipna=True))

	print("Data summary -- averages:")
	print(df_reduced)

	return df


# == LEAST SQUARES PART ==

def fit_lsq(X, y):

	model = LinearRegression(fit_intercept=True).fit(X, y)
	model_param = pd.Series(index=['const', *X.columns], data=[model.intercept_, *model.coef_])

	return (model, model_param)


# "Naive" one-stage linear regression
def fit_1stage_lsqr(df: pd.DataFrame):

	for vars in powerset(['booked', 'business']):
		vars = ['treat'] + vars

		print("--")

		X = df[vars]
		y = df['canceled']

		print("LSQR: {} => {}".format(", ".join(X.columns), y.name))

		(model, model_param) = fit_lsq(X, y)

		print("Model parameters:", dict(model_param))
		print("Average treatment effect (cancellations per day):", (model_param['treat'] * PARAM['visitors_per_day']))


# "Linear structural equation model" -- "stacked" linear regression
def fit_2stage_lsem(df: pd.DataFrame):

	print("--")

	# STAGE 1

	X = df[['treat', 'business']]
	M = df['booked']  # Mediator

	(model1, param1) = fit_lsq(X, M)

	print("Stage 1 model parameters:", dict(param1))

	M1 = model1.predict(X)

	# show_data_summary(X.assign(**{M.name: model1.predict(X), ("({})".format(M.name)): M}))

	# STAGE 2

	X = df[['treat']].assign(**{M.name: model1.predict(X)})
	y = df['canceled']

	(model2, param2) = fit_lsq(X, y)

	print("Stage 2 model parameters:", dict(param2))

	y2 = model2.predict(X)

	# Stacked model

	def ev(t, b):
		return param2['const'] + (param2['treat'] * t) + param2['booked'] * (param1['const'] + (param1['treat'] * t) + (param1['business'] * b))

	# Sample correlation between errors between stage 1 and stage 2

	print("Sample correlation between errors:")
	# Correlation of errors
	r = np.corrcoef(M - M1, y - y2)[1, 0]
	# Significance via t-statistic, two-sided
	p = 2 * stats.t.cdf(-abs(r / sqrt(1 - (r * r)) * sqrt(len(X) - 2)), df=(len(X) - 2))
	print("r = {} (p = {})".format(r, round(p, 4)))

	# Effect of treatment

	# For stacked linear regression,
	# the (predicted) average treatment effect
	# is just param1['treat'] * param2['booked']
	# (cf. p. 21 in [3])

	# The number is exactly the same as in the direct model "treat, business => canceled"

	for b in [1, 0]:
		print("ATE on 'business = {}' traveler (canceled / day):".format(b), ((ev(t=1, b=b) - ev(t=0, b=b)) * PARAM['visitors_per_day']))


# == MORE COMPLEX REGRESSION ==

def fit_1stage_rate(df: pd.DataFrame):

	for vars in powerset(['booked', 'business']):
		vars = ['treat'] + vars

		print("--")

		X = df[vars]
		y = df['canceled']

		print("XGB: {} => {}".format(", ".join(X.columns), y.name))

		# This classifies by the Poisson rate
		model0 = xgb.XGBRegressor(objective='count:poisson').fit(X, pd.DataFrame(y))
		y_rate_pred = pd.Series(model0.predict(X), index=y.index, name='rate')
		print("Full model rounded-rate percentages:", pred_perc(y_rate_pred))

		ate = model0.predict(X.loc[X['treat']]).mean() - model0.predict(X.loc[~X['treat']]).mean()
		print("Average treatment effect (cancellations per day):", (ate * PARAM['visitors_per_day']))


def fit_2stage_rate(df: pd.DataFrame):

	print("--")

	# STAGE 1

	stage1_in = ['treat', 'business']
	mediator = 'booked'

	X = df[stage1_in]
	M = df[mediator]

	# This classifies by the Poisson rate
	model1 = xgb.XGBRegressor(objective='count:poisson').fit(X, pd.DataFrame(M))
	rate_M_pred = pd.Series(model1.predict(X), index=X.index, name='rate')
	# print(pd.DataFrame(data=[df['business'], M, M_rate_pred]).T)
	print("Stage 1 model rounded-rate percentages:", pred_perc(rate_M_pred))

	# STAGE 2

	stage2_in = ['treat'] # besides the mediator
	# stage2_in += ['booked'] # ?
	stage2_out = 'canceled'

	X = df[stage2_in].assign(mediator=model1.predict(X))
	y = df[stage2_out]

	model2 = xgb.XGBRegressor(objective='count:poisson').fit(X, pd.DataFrame(y))
	rate_y_pred = pd.Series(model2.predict(X), index=X.index, name='rate')
	print("Stage 2 model rounded-rate percentages:", pred_perc(rate_y_pred))

	# Combined predictor
	# Avoid the sklearn "pipeline" hassle
	def rate_predictor(df: pd.DataFrame):
		return pd.Series(model2.predict(df[stage2_in].assign(mediator=(model1.predict(df[stage1_in])))), index=df.index, name='rate')

	# Causal mediation effect, [2, Eqn. (1)]
	def delta(df: pd.DataFrame) -> pd.DataFrame:
		M0 = model1.predict(df[stage1_in].assign(treat=0))
		M1 = model1.predict(df[stage1_in].assign(treat=1))
		Y0 = model2.predict(df[stage2_in].assign(mediator=M0))
		Y1 = model2.predict(df[stage2_in].assign(mediator=M1))

		return (Y1 - Y0)

	# Direct effect calculator, [2, Eqn. (2)]
	def zeta(df: pd.DataFrame) -> pd.DataFrame:
		Mt = model1.predict(df[stage1_in])
		Y0 = model2.predict(df[stage2_in].assign(treat=0, mediator=Mt))
		Y1 = model2.predict(df[stage2_in].assign(treat=1, mediator=Mt))

		return (Y1 - Y0)

	# What is the treatment effect on cancellations per day?

	print("Effects on cancellations per day")
	print("ATE:  Average treatment effect")
	print("ACME: Average causal mediation effect")
	print("ADTE: Average direct treatment effect")

	for b in [0, 1]:
		print("Business traveler: ", speak(bool(b)))

		sub_df = df.loc[df['business'] == b]

		mean_rate0 = rate_predictor(sub_df.loc[~sub_df['treat']]).mean()
		mean_rate1 = rate_predictor(sub_df.loc[sub_df['treat']]).mean()

		[ate, acme, adte] = map(
			lambda x: (x * PARAM['visitors_per_day']),
			[
				(mean_rate1 - mean_rate0), # ATE
				delta(sub_df).mean(),      # ACME
				zeta(sub_df).mean(),       # ADTE
			]
		)

		print("{} (ATE)  =  {} (ACME)  +  {} (ADTE)".format(*map(sigdig, (ate, acme, adte))))

	# print([
	# 	zeta(df.sample(100, replace=True)).mean()
	# 	for _ in range(100)
	# ])


	# # Direct effect calculator, [2, Eqn. (2)]
	# # Average over a dataset, with extra correlation (of error between model1 and model2)
	# def zeta_corr(df: pd.DataFrame, theta0: float, theta1: float) -> pd.DataFrame:
	# 	Mt = model1.predict(df[stage1_in])
	# 	Y0 = model2.predict(df[stage2_in].assign(treat=0, mediator=(Mt * theta0)))
	# 	Y1 = model2.predict(df[stage2_in].assign(treat=1, mediator=(Mt * theta0)))
	#
	# 	e = (Y1 - Y0).mean()
	# 	s = (Y1 - Y0).std()
	#
	#
	# # Sensitivity computation
	# gammas = np.logspace(-2, 2, 50)
	# zetas = np.array([zeta(df, mediator_infactor=gamma).mean() for gamma in gammas])
	# zetas_std = np.array([zeta(df, mediator_infactor=gamma).std() for gamma in gammas])
	#
	# fig: plt.Figure
	# ax: plt.Axes
	# (fig, ax) = plt.subplots()
	#
	# ax.plot(gammas, zetas)
	# ax.plot(gammas, zetas + zetas_std)
	# ax.plot(gammas, zetas - zetas_std)
	# ax.set_xscale('log')
	# plt.show()



# == MASTER ==

def main():
	df = show_data_summary(get_simulated_data())

	# fit_1stage_lsqr(df)
	# fit_2stage_lsem(df)

	# fit_1stage_rate(df)
	fit_2stage_rate(df)


if (__name__ == "__main__"):
	main()
