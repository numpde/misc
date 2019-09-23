
# RA, 2019-09-04

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
import os
import calendar

PARAM = {
	# Hourly driver activity
	'data_hda': "ORIGINALS/UV/hda.csv",
	# Replace missing values by ...
	'hda_fillna': 0,

	# Hourly overview search
	'data_hos': "ORIGINALS/UV/hos.csv",

	# Finished ride average value estimate (EUR)
	'ride_fare': 10,
	# Fraction of fare as company revenue
	'ride_fare_revenue_frac': 0.2,

	# Date format in the input CSV files
	'date_format': "%Y-%m-%d %H",

	# Output files
	'out_exploration_recbyhour': "OUTPUT/exploration/recbyhour.{ext}",

	'out_task1': "OUTPUT/task1/undersupp_{type}.{ext}",
	'out_task2': "OUTPUT/task2/supply-demand.{ext}",
	'out_task3': "OUTPUT/task3/undersupp_heat.{ext}",
	'out_task4': "OUTPUT/task4/extra_hours.{ext}",

	# Mapping 0 -> 'Mon', 1 -> 'Tue, etc.
	'7': (lambda i: calendar.day_abbr[i]),

	'n_undersupplied': 36, # Task 1
	'n_mostpeakhours': 36, # Task 4
	'n_highestdemand': 36, # Task 5
}



# ~~~ DATA SOURCE AND PREP ~~~

# Hourly driver activity
COL_DATE = "Date"
COL_ACTIVE_DRIVERS = "Active drivers"
COL_ONLINE = "Online (h)"
COL_HAS_BOOKING = "Has booking (h)"
COL_WAITING = "Waiting for booking (h)"
COL_BUSY = "Busy (h)"
COL_FINISHED_RIDES = "Finished Rides"

# Hourly overview search
# COL_DATE = "Date"
COL_SAW0 = "Taxify: People saw 0 cars (unique)"
COL_SAW1 = "Taxify: People saw +1 cars (unique)"
COL_COV_RATIO = "Taxify: Coverage Ratio (unique)"


def set_data_date(df: pd.DataFrame):
	# Parse date/time
	df[COL_DATE] = pd.to_datetime(df[COL_DATE], format=PARAM['date_format'])

	df = df.assign(**{
		'7': df[COL_DATE].apply(lambda x: x.weekday()),
		'24': df[COL_DATE].apply(lambda x: x.hour),
	})

	return df


# Load datasets from disk:
# Hourly driver activity
def get_data_hda():
	df = set_data_date(pd.read_csv(PARAM['data_hda'])).fillna(PARAM['hda_fillna'])
	return df


# Load datasets from disk:
# Hourly overview search
def get_data_hos():
	df = set_data_date(pd.read_csv(PARAM['data_hos']))
	return df


# ~~~ HELPERS ~~~

# Prepare output directories
def makedirs():
	for (k, v) in PARAM.items():
		if k.startswith('fig_') or k.startswith('out_'):
			os.makedirs(os.path.dirname(v), exist_ok=True)



# Format as (weekday)-(hour)
def get_weekhour(d: dt.datetime):
	return "{}-{:02}".format(d.strftime("%w[%a]"), d.hour)


# ~~~ EXPLORATION ~~~

# Basic dataset properties
def exploration(hda: pd.DataFrame):
	ax: plt.Axes
	fig: plt.Figure

	dates = hda['Date']

	# (hours, counts) = zip(*sorted(dates.groupby(lambda i: dates[i].hour).count().items()))
	# ax.plot(hours, counts)

	# 1. Histogram of entries by hour of the day

	(fig, ax) = plt.subplots()
	ax.hist(dates.apply(lambda x: x.hour), bins=range(0, 25), rwidth=0.8)
	ax.set_xticks(range(25))
	ax.set_xlabel("Time of day")
	ax.set_ylabel("Number of records")
	fig.savefig(PARAM['out_exploration_recbyhour'].format(ext="png"))
	plt.close(fig)

	# 2. ...


# ~~~ TASK 1 ~~~

def task1(hos):
	fig: plt.Figure
	ax: plt.Axes

	# Weekday-Hour
	WH = hos[COL_DATE].apply(get_weekhour)

	# People saw 0 cars in the app
	saw0 = hos[COL_SAW0].groupby(WH).mean()
	# People saw some cars in the app
	saw1 = hos[COL_SAW1].groupby(WH).mean()

	for missed_type in ['abs', 'rel']:
		missed = {
			'rel': saw0 / (saw0 + saw1),
			'abs': saw0
		}[missed_type]

		# Most undersupplied moments
		undersupplied = missed.nlargest(PARAM['n_undersupplied'])

		with open(PARAM['out_task1'].format(type=missed_type, ext="txt"), 'w') as fd:
			print("Most undersupplied week-hours (in decreasing order):", file=fd)
			print("\n".join(undersupplied.index), file=fd)

		(fig, ax) = plt.subplots()
		ax.plot(missed, '--.', label=("Missed clients (av. hourly, {type})".format(type=missed_type)))
		ax.plot(undersupplied, '*', label="Most undersupplied")
		ax.xaxis.set_visible(False)
		ax.legend(loc='upper right')
		fig.savefig(PARAM['out_task1'].format(type=missed_type, ext="png"))
		plt.close(fig)


# ~~~ TASK 2 ~~~

def task2(hos: pd.DataFrame):
	fig: plt.Figure
	ax: plt.Axes

	# Hour of the day
	H = hos[COL_DATE].apply(lambda x: x.hour)

	# People saw zero cars / some cars in the app
	saw0 = hos[COL_SAW0].groupby(H).sum()
	saw1 = hos[COL_SAW1].groupby(H).sum()

	demand = saw0 + saw1
	supply = saw1

	(fig, ax) = plt.subplots()
	param = dict(align='edge', width=1, edgecolor='k')
	ax.bar(demand.index, demand, label="Demand", **param)
	ax.bar(supply.index, supply, label="Supply", **param)
	ax.legend(loc='lower right')
	ax.set_xticks(range(25))
	fig.savefig(PARAM['out_task2'].format(ext="png"))
	plt.close(fig)


# ~~~ TASK 3 ~~~

def task3(hos):
	fig: plt.Figure
	ax: plt.Axes

	hos_24x7: pd.DataFrame
	hos_24x7 = hos[['24', '7', COL_SAW0]] \
		.groupby(['24', '7'], as_index=False) \
		.mean() \
		.pivot(index='7', columns='24', values=COL_SAW0) \
		.sort_index(ascending=False)

	(fig, ax) = plt.subplots()
	im = ax.imshow(hos_24x7.values, cmap=plt.cm.get_cmap('Blues'), extent=(0, 24, -0.5, 6.5), origin='lower')
	ax.tick_params(axis='y', length=0)
	ax.set_yticks(range(7))
	ax.set_yticklabels(map(PARAM['7'], hos_24x7.index))
	ax.set_xticks(range(25))

	# cb = fig.colorbar(im, fraction=2/24/7)
	# cb.set_ticks(im.get_clim())
	# cb.ax.tick_params(length=0)

	fig.savefig(PARAM['out_task3'].format(ext="png"), bbox_inches='tight', dpi=300)
	plt.close(fig)


# ~~~ TASK 4 ~~~

def task4(hda: pd.DataFrame, hos: pd.DataFrame):
	fig: plt.Figure
	ax: plt.Axes

	hos = hos[['24', '7', COL_SAW0, COL_SAW1]].groupby(['7', '24']).mean()
	hda = hda[['24', '7', COL_ONLINE]].groupby(['7', '24']).mean()

	# Most undersupplied moments
	missed_idx = hos[COL_SAW0].nlargest(PARAM['n_undersupplied']).index

	hos = hos.loc[missed_idx, :]
	hda = hda.loc[missed_idx, :]

	# Extrapolate to estimate extra online hours wanted
	extra_hours = hda[COL_ONLINE] * (hos[COL_SAW0] / hos[COL_SAW1])

	COL_HAVE = 'av. online, h (have)'
	COL_WANT = 'av. online, h (want)'

	df = pd.DataFrame(data={COL_HAVE: hda[COL_ONLINE], COL_WANT: extra_hours}).sort_index()

	# Days with undersupply
	days = sorted(set(df.index.get_level_values(0)))

	(fig, axs) = plt.subplots(
		1, len(days),
		figsize=[10, 3],
		squeeze=True, sharey='row',
		gridspec_kw=dict(
			width_ratios=[(len(df.loc[i]) or 1) for i in days]
		),
	)

	axs[0].set_ylabel("Extra online hours wanted")

	for i in days:
		ax = axs[i]
		df_day = df.loc[i]

		ax.bar(list(map(str, df_day.index)), df_day[COL_WANT], width=0.9, color="C1")
		ax.tick_params(axis='x', labelsize=5)

		ax.set_title(PARAM['7'](i))

	fig.savefig(PARAM['out_task4'].format(ext="png"), bbox_inches='tight', dpi=300)
	plt.close(fig)


# ~~~ TASK 5 ~~~

# UNFINISHED
def task5(hda: pd.DataFrame, hos: pd.DataFrame):
	fig: plt.Figure
	ax: plt.Axes

	hos = hos[['24', '7', COL_SAW0, COL_SAW1]].groupby(['7', '24']).mean()

	high_demand_idx = (hos[COL_SAW0] + hos[COL_SAW1]).nlargest(PARAM['n_highestdemand']).index

	# High-demand week-hours
	hda_hd: pd.DataFrame
	hda_hd = (hda.groupby(['7', '24']).mean()).loc[high_demand_idx, :]

	# Weekly revenue, averaged over weeks
	weekly_revenue = (hda_hd[COL_FINISHED_RIDES].sum()) * PARAM['ride_fare'] * PARAM['ride_fare_revenue_frac']
	print(weekly_revenue)

	# Average number of finished rides / hour
	rides_per_hour = hda_hd[COL_FINISHED_RIDES].mean()
	print(rides_per_hour)

	pass


# ~~~ MAIN ~~~

def main():
	# Data source
	hda = get_data_hda()
	hos = get_data_hos()

	# Prepare output directories
	makedirs()

	# Data exploration
	exploration(hda)

	# Tasks
	task5(hda, hos)
	task4(hda, hos)
	task3(hos)
	task2(hos)
	task1(hos)


if __name__ == "__main__":
	main()
