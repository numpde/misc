
# RA, 2019-08-09


# Exploration of the "Fraud detection" dataset
# https://data-flair.training/blogs/data-science-machine-learning-project-credit-card-fraud-detection/

# Original (?) publication:
# A. D. Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi,
# "Calibrating Probability with Undersampling for Unbalanced Classification"
# 2015 IEEE Symposium Series on Computational Intelligence, Cape Town, 2015, pp. 159-166.
#
# Paper:
# https://dx.doi.org/0.1109/SSCI.2015.33
# https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf
#
# Dataset cited in the paper:
# http://www.ulb.ac.be/di/map/adalpozz/data/creditcard.Rdata
#
# Cf. "loading rdata into python":
# https://stackoverflow.com/questions/21288133/

# Further references
#
# "[..] Precision/Recall curve should be preferred in highly imbalanced situations
# https://www.kaggle.com/lct14558/imbalanced-data-why-you-should-not-use-roc-curve


TESTMODE = True  # Fast turnaround
# TESTMODE = False # Production


# (!) Import does not commute with helpers.commons
from openTSNE.sklearn import TSNE

from helpers.commons import logger
from helpers.commons import makedirs
from helpers import commons

import re
import pandas as pd
import numpy as np
from numpy.random import RandomState
from collections import Counter

# from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

from contextlib import contextmanager

from typing import Iterator, Generator, Tuple
import matplotlib.pyplot as plt


# ~~~~ PROGRAM PARAMETERS REPO ~~~~ #

PARAM = {
	'data': "ORIGINALS/UV/creditcard.csv.zip",

	'do_train': True,

	'do_first_summary': False,
	'first_summary': makedirs("OUTPUT/summary/summary_{of}.{ext}"),

	'do_first_tsne': False,
	'first_tsne_rs': RandomState(0),
	'first_tsne_fig': makedirs("OUTPUT/tsne/{run:04}.{ext}"),

	# Holdout set
	'3way_holdout': 0.1,
	'3way_rs': RandomState(41), # SACRED

	# Train / validation split
	'2way_train': 0.7,

	'train_rs': RandomState(40),

	# ROC curves plot filename
	'roc_fig': makedirs("OUTPUT/roc/{dataset}.{ext}"),
	# Prevision vs Recall curves
	'pvr_fig': makedirs("OUTPUT/pvr/{dataset}.{ext}"),

	# Feature columns (regex with re.match)
	'data_features_regex': ["Time*", "V*", "Amount"],
	# Class labels column
	'data_label_column': "Class",

	# Options for fig.savefig
	'savefig_kwargs': dict(bbox_inches='tight', pad_inches=0, dpi=300),
}


# ~~~~ HELPERS ~~~~ #

# Verbose open
open = commons.logged_open


# Use this to make sure the (holdout) split is the same every time
def partition_2way(df, p: float, random_state: RandomState):
	assert(0 <= p <= 1)
	i = random_state.choice([True, False], size=len(df), replace=True, p=[p, 1 - p])
	logger.debug("partition_2way: {} (hash: {})".format(dict(Counter(i).most_common()), hash(tuple(i))))
	return (df.loc[i], df.loc[~i])


@contextmanager
def subplots(filename=None) -> Generator[Tuple[plt.Figure, plt.Axes], None, None]:
	# Create a new figure
	(fig, ax) = plt.subplots()
	try:
		yield (fig, ax)

		# Save if filename provided, otherwise just show
		if filename:
			with open(filename, 'wb') as fd:
				fig.savefig(fd, **PARAM['savefig_kwargs'])
		else:
			plt.show()
	finally:
		# Close the figure, if possible
		try:
			plt.close(fig)
		except:
			pass


def df2xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
	# Select features columns
	X = df[[col for col in df.columns if any(re.match(feat_like, col) for feat_like in PARAM['data_features_regex'])]]

	# Class labels column
	y = df[PARAM['data_label_column']]

	return (X, y)


# ~~~~ PREDICTOR METRICS CURVES ~~~~ #

# ROC curve
def plot_roc(y, p, filename=None):
	(fpr, tpr, thresholds) = metrics.roc_curve(y, p)

	with subplots(filename) as (fig, ax):
		ax.plot(fpr, tpr)

		ax.set_xlabel("False positives rate")
		ax.set_ylabel("True positives rate")


# Prevision vs Recall curve
def plot_pvr(y, p, filename=None):
	(pre, rec, thresholds) = metrics.precision_recall_curve(y, p)

	with subplots(filename) as (fig, ax):
		ax: plt.Axes
		ax.plot(pre, rec)

		ax.set_xlabel("Precision = P(is fraud | called)")
		ax.set_ylabel("Recall = P(called | is fraud)")


# ~~~~ DATA SOURCE ~~~~ #

def get_data() -> pd.DataFrame:
	with open(PARAM['data'], 'rb') as fd:
		df = pd.read_csv(fd, compression='zip')

	# Add a time-of day feature
	SECONDS_IN_A_DAY = (60 * 60 * 24)
	df = df.assign(TimeH=(df['Time'].mod(SECONDS_IN_A_DAY)))

	return df


# ~~~~ EXPLORATORY ~~~~ #

def show_data_summary(df: pd.DataFrame):

	# Basic dataframe summary
	with open(PARAM['first_summary'].format(of="df", ext="txt"), 'w') as fd:
		# Basic
		print(df.head(), file=fd)
		print(df.describe(), file=fd)

		# Category frequencies
		print("Class frequencies:", dict(Counter(df[PARAM['data_label_column']]).most_common(10)), file=fd)

		# Missing data
		print("Null values:", df.isnull().sum().sum(), file=fd)

	# Class histograms by column
	for column in df.columns:

		with subplots(PARAM['first_summary'].format(of=commons.safe_filename(column), ext="png")) as (fig, ax):

			for label in set(df[PARAM['data_label_column']]):
				df.loc[df.Class == label, column].hist(ax=ax, bins='fd', alpha=0.5, label=label)

			ax.set_xlabel("Column: {}".format(column))
			ax.set_ylabel("Count")

			ax.set_yscale('log')
			ax.grid(zorder=-10)
			ax.legend()


def tsne(df):
	(X, y) = df2xy(df)

	rs = PARAM['first_tsne_rs']

	for run in range(3):
		logger.debug("Computing TSNE embedding")
		x = TSNE(random_state=rs, n_jobs=commons.PARALLEL_MAP_CPUS).fit_transform(X.values)

		logger.debug("Plotting TSNE embedding")

		with subplots(PARAM['first_tsne_fig'].format(run=run, ext="png")) as (fig, ax):

			for label in set(y):
				i = (y == label)
				ax.scatter(x[i, 0], x[i, 1], s=1)

			ax.set_xticks([])
			ax.set_yticks([])


# ~~~~ TRAINING ~~~~ #

def train(df):
	rs: RandomState
	rs = PARAM['train_rs']

	# Train/Test split

	(df_train, df_test) = partition_2way(df, PARAM['2way_train'], random_state=rs)
	del df

	# Set up classification model

	usual_model_params = {'random_state': rs, 'class_weight': "balanced"}

	# # Random forest
	# core_model = RandomForestClassifier(n_estimators=1111, max_depth=4, n_jobs=commons.PARALLEL_MAP_CPUS, **usual_model_params)

	# # Logistic regression
	# core_model = LogisticRegression(solver='lbfgs', max_iter=1000, **usual_model_params)

	# RBF with SVM
	# https://www.kaggle.com/c/home-credit-default-risk/discussion/63499
	core_model = SVC(probability=True, **usual_model_params)

	# Set up the full model pipeline
	pipeline = Pipeline([
		("normalize_features", StandardScaler()),
		("core_model", core_model),
	])

	# Train the model

	logger.debug("Starting training")
	pipeline.fit(*df2xy(df_train))

	# Extract the class labels from the predictor
	# Relevant, e.g, for predict_proba

	class_labels = list(pipeline.classes_)
	logger.debug("Predictor class labels: {}".format(class_labels))

	# Basic reports

	print("Accuracy reports:")

	for (subdf, meta) in zip([df_train, df_test], ["train", "test"]):

		(X, y) = df2xy(subdf)
		fraud_proba = pipeline.predict_proba(X)[:, class_labels.index(1)]

		# Confusion matrix

		print("Classification report ({} set):".format(meta))
		print(metrics.classification_report(y, pipeline.predict(X), digits=3))

		# Precision-vs-recall curve, ROC curve

		plot_pvr(y, fraud_proba, filename=(PARAM['pvr_fig'].format(dataset=meta, ext="png")))
		plot_roc(y, fraud_proba, filename=(PARAM['roc_fig'].format(dataset=meta, ext="png")))


# ~~ MASTER ~~ #

def main():
	# Holdout + training dataset
	logger.info("Master: load and partition data (holdout subset)")
	(df_holdout, df) = partition_2way(get_data(), PARAM['3way_holdout'], PARAM['3way_rs'])

	if PARAM['do_first_summary']:
		logger.info("Master: calling show_data_summary(...)")
		show_data_summary(df)

	if TESTMODE:
		df = df.sample(10000)
		logger.warning("Master: Subsampled the train/test data to size {}".format(len(df)))

	if PARAM['do_first_tsne']:
		logger.info("Master: calling tsne(...)")
		tsne(df)

	if PARAM['do_train']:
		logger.info("Master: calling train(...)")
		train(df)


# ~~~~ ENTRY ~~~~ #

if (__name__ == "__main__"):
	main()
