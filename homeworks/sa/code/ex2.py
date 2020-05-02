# RA, 2020-04-30
# Python 3.7

# SA HWK Ex 2
# https://gist.github.com/numpde/ea33f6de0cbb908f12789c8286d98959

from pathlib import Path

import numpy as np
import pandas as pd

from numpy import isclose as equalish

from scipy.optimize import fsolve

from datetime import datetime as dt
from datetime import timedelta

# Note: some computations are performed using Decimal
from decimal import Decimal

# Provides the money unit
# Note: (2 + Money(1, 'USD')) is OK but (2.0 + ...) is not
from money import Money

import sys, os

# # Redirect "print" (avoid logger for this task)
# sys.stdout = open("output-ex2.txt", "w")


"""
			Constants
"""

PARAM = {
	'Report src': Path(__file__).parent.parent / "data/UV/2/SamplePortfolio.xlsx",
	'Report trg': Path(__file__).parent / "UV/SamplePortfolio_Filled.xlsx",

}

"""
			Helpers
"""


def relpath(path):
	"""
	The `path` relative to the current directory
	"""
	return os.path.relpath(str(path), os.path.curdir)


def singleton(s: pd.Series):
	"""
	If the series has only one distinct element
	return it, otherwise raise an AssertionError
	"""
	assert (1 == len(set(s))), "Series is not a singleton set"
	return set(s).pop()


# Days in year as `timedelta`
days_in_year = (lambda year: timedelta(days=(365 + (not (year % 4)))))

# Last day as `datetime`
last_day_of_year = (lambda year: dt(year=year, month=12, day=31))

"""
			Load the portfolio data
"""


def load_data(filepath):
	df = pd.read_excel(filepath, sheet_name=None)
	print(F"Loaded data from: {relpath(filepath)}")
	return df


# Where to look for the data file
attempt_filepaths = [
	Path(__file__).parent / PARAM['Report src'].name,
	Path.cwd() / PARAM['Report src'].name,
	PARAM['Report src'],
]

for filepath in attempt_filepaths:
	try:
		df = load_data(PARAM['Report src'], )
		break
	except FileNotFoundError:
		continue

# Sheets
df_ptf = df['Portfolio']
df_ir = df['IR Curves'].set_index("Tenor (yrs)", verify_integrity=True)
df_shocks = df['Partial shocks'].set_index('Asset type', verify_integrity=True)

# Relation between sheets "Portfolio" and "IR Curves"
ccy_to_zone = {
	'EUR': "Euro",
	'USD': "United States",
	'JPY': "Japan",
	'CHF': "Switzerland",
	'AUD': "Australia",
	'GBP': "United Kingdom",
}

assert set(df_ptf['Quote ccy']).issubset(ccy_to_zone.keys())
assert (set(df_ir.columns) == set(ccy_to_zone.values()))

"""
			Extract currency exchange rates
"""

# Extract the '4_Portfolio-PortfolioCurrency'
portfolio_ccy = singleton(df_ptf['Portfolio ccy'])

# Currency unit attacher
to_portfolio_ccy = (lambda x: Money(x, portfolio_ccy))

# Attempt to extract the relevant exchange rates
# For simplicity, average the available guesses
exchange_rates = pd.DataFrame({
	'Q': df_ptf['Quote ccy'],
	'P': df_ptf['Portfolio ccy'],
	'Q2P': df_ptf['Mkt val pccy'] / df_ptf['Mkt val qccy'],
}).dropna().groupby(['Q', 'P']).mean().squeeze()


# Currency converter
def convert_ccy(m: Money, to: str, xrate=None):
	xrate = xrate or exchange_rates[m.currency, to]
	return Money(m.amount * Decimal(xrate), to)


# Now attach the money unit to certain columns
# Market value in quote currency
df_ptf['Mkt val qccy'] = df_ptf['Mkt val qccy'].apply(Decimal) * (
	df_ptf['Quote ccy'].apply(lambda ccy: Money(1, ccy))
)
# Market value in portfolio currency
df_ptf['Mkt val pccy'] = df_ptf['Mkt val pccy'].apply(to_portfolio_ccy)

"""
			Cash assets / FX shock
"""

"""
Cash position
The market value in quote ccy is the notional.
The market value in portfolio ccy is the former but
converted to portfolio ccy.
Thus, this position is sensitive to 
	- FX up
	- FX down

In fact, all assets in Foreign CCY 
respond in the same way to FX up/down. 
"""

"""
Stress scenarios
FX up   : more USD for the same Foreign CCY
FX down : less USD for the same Foreign CCY
"""

for scenario in ["FX up", "FX down"]:
	for (i, pos) in df_ptf.iterrows():
		if (pos['Quote ccy'] == portfolio_ccy):
			mkt_val_pccy = pos['Mkt val qccy']
		else:
			q2p_xrate = exchange_rates[pos['Quote ccy'], portfolio_ccy]

			try:
				q2p_xrate *= (1 + df_shocks['Shock size'][scenario])
			except KeyError as offender:
				print(F"Warning: scenario not found ({offender})")

			mkt_val_pccy = convert_ccy(pos['Mkt val qccy'], portfolio_ccy, q2p_xrate)

		# Record the effect of the scenario
		df_ptf.loc[i, scenario] = mkt_val_pccy

"""
			Bonds
"""

"""
Assumption:
If a bond has yield y,
interpret the "Spread up" shock x
as new yield (y + x)
"""


def value_pos_bond(cpn_apr: float, cpn_freq: int, date_now: dt, next_cpn: dt, maturity: dt, y: float, face_value=1):
	"""
	Fixed-rate bond valuation

	Parameters with examples:

	# Coupon freq/year
	cpn_freq = 2
	# Coupon annual percentage rate (in percent)
	cpn_apr = 1.9

	# Coupon dates
	parse_date = (lambda s: dt.strptime(s, "%m/%d/%Y"))
	date_now = parse_date("12/31/2019")
	maturity = parse_date("4/15/2025")
	next_cpn = parse_date("4/15/2020")

	# The yield (fraction, not percent)
	# as the effective yearly forward discount/interest rate
	y = 0.1

	# Bond face value
	face_value = 100
	"""

	# Gives the one-year forward interest rate for that year
	fw_ir = (lambda year: y)
	# Gives the one-year discount for that year
	fwd_disc = (lambda year: 1 / (1 + fw_ir(year)))

	# Coupon rate per payout
	cpn_pay_rate = cpn_apr / 100 / cpn_freq

	# How many cash flows?
	ncf = round(1 + (maturity - next_cpn) / timedelta(days=365.25) * 2)

	# Prepare the array of cash flows (normalized by the face value)
	cash_flows = pd.Series(
		index=[date_now, *pd.date_range(next_cpn, maturity, ncf).round('D')],
		data=0,
	)
	cash_flows[cash_flows.index >= next_cpn] = cpn_pay_rate
	cash_flows[maturity] += 1

	# # Insert all intermediate days (optional)
	# cash_flows = cash_flows.resample('D').sum().fillna(0)

	# Introduce end-of-year checkpoints
	cash_flows = cash_flows.append(
		pd.Series(index=map(last_day_of_year, set(cash_flows.index.year)), data=0)
	)

	# Collapse possible date duplicates
	cash_flows = cash_flows.groupby(cash_flows.index).sum()

	assert equalish(float(cash_flows.sum()), (1 + ncf * cpn_pay_rate))

	# Important: sort by date before next block
	cash_flows = cash_flows.sort_index()

	# The year of each period
	years = pd.Series(index=cash_flows.index, data=cash_flows.index.year)

	# Discounts to present-value for each period (note cumprod)
	discounts = pd.Series(
		index=cash_flows.index,
		data=[1] + list(
			years[1:].apply(fwd_disc)
			**
			(np.diff(cash_flows.index) / years[1:].apply(days_in_year))
		)
	).cumprod()

	# Present value (normalized)
	pv = sum(cash_flows * discounts)

	# Macaulay duration (in days)
	macd_days = sum((cash_flows * discounts) * (cash_flows.index - cash_flows.index[0]).days) / pv
	# Modified duration (in days)
	# TODO: is this valid for variable-period cash flows?
	modd_days = macd_days / (1 + y / cpn_freq)

	# Unnormalize (may introduce the money unit)
	pv = face_value * pv

	return pv


try:
	# Test value_pos_bond on an example
	# computed "by hand" in a spread sheet
	bond = {
		'cpn_freq': 2,
		'cpn_apr': 1.9,
		'face_value': 100,

		'date_now': dt(year=2019, month=12, day=31),
		'maturity': dt(year=2025, month=4, day=15),
		'next_cpn': dt(year=2020, month=4, day=15),

		'y': 1.7777 / 100,
	}
	pv = 101.056329644344
	assert equalish(pv, value_pos_bond(**bond))
except AssertionError:
	raise

#

ipos_bond_fixed = (df_ptf['Asset type'] == "Bonds") & (df_ptf['Cpn type'] == "Fixed")
assert any(ipos_bond_fixed)


def value_pos_bond_kwargs(pos: pd.Series, **kwargs):
	"""
	Rename entries in `pos` as suitable for `value_pos_bond`
	"""
	bond = pd.Series({
		'cpn_freq': pos['Cpn freq'],
		'cpn_apr': pos['Coupon'],

		# Note: May have to convert to float
		'face_value': pos['Notional/Quantity'],

		'date_now': pos['Valuation date'],
		'maturity': pos['Maturity'],
		'next_cpn': pos['Next cpn'],

		# Wild guess for the yield
		'y': 0.01,

		**kwargs
	})
	return bond


# Record bond yields
bond_yields = pd.Series(index=ipos_bond_fixed[ipos_bond_fixed].index, data=0)

# Find the yield to maturity for each fixed-rate bond
for (i, pos) in df_ptf[ipos_bond_fixed].iterrows():
	bond = value_pos_bond_kwargs(pos)

	# The given present value in native currency
	pv = float(pos['Mkt val qccy'])

	# Function to solve (i.e. find root)
	f = lambda y: (pv - value_pos_bond(**{**bond, 'y': y}))

	# Find the yield to maturity
	[bond.y] = fsolve(f, x0=bond.y, xtol=1e-10, maxfev=111)

	assert (-2 / 100 <= bond.y <= 10 / 100), F"Unexpected bond yield: {bond.y}"

	assert equalish(pv, value_pos_bond(**bond))

	# mkt_val_qccy = Money(pv, pos['Quote ccy'])
	# mkt_val_pccy = convert_ccy(mkt_val_qccy, to=pos['Portfolio ccy'])

	bond_yields[i] = bond.y

# Implement the "CCY Spread up" scenario
for (i, pos) in df_ptf[ipos_bond_fixed].iterrows():
	bond = value_pos_bond_kwargs(pos)
	scenario = "Spread up"
	shock = df_shocks['Shock size'][F"{pos['Quote ccy']} {scenario}"]
	bond.y = bond_yields[i] + shock
	pv = value_pos_bond(**bond)
	mkt_val_qccy = Money(pv, pos['Quote ccy'])
	mkt_val_pccy = convert_ccy(mkt_val_qccy, to=pos['Portfolio ccy'])

	# Record the effect of the scenario
	df_ptf.loc[i, scenario] = mkt_val_pccy

"""
			FX forwards
"""

"""
To value an `FX forward` position,
we take the notional N (in the quote ccy) and 
discount it by the IR (of the quote ccy) using time to maturity.
The market value in portfolio currency
is obtained by conversion using the exchange spot rate. 

Thus, in addition to FX shock, this position
is sensitive to interest rates.
"""

ipos_fx_fwd = (df_ptf['Asset type'] == "FX forward")
assert any(ipos_fx_fwd)


def value_pos_fx_fwd(pos, scenario=None):
	qccy = pos['Quote ccy']
	pccy = pos['Portfolio ccy']
	notional = pos['Notional/Quantity']

	# Assume valuation takes place at end of year
	assert (pos['Valuation date'].month == 12) and (pos['Valuation date'].day == 31)
	# Assume the maturity is within a year
	assert (pos['Maturity'].year == pos['Valuation date'].year + 1)

	# Interest rate for next year (fraction, not percent)
	r = df_ir[ccy_to_zone[qccy]][1]

	if (scenario in {"IR up", "IR down"}):
		try:
			r += df_shocks['Shock size'][F"{qccy} {scenario}"]
		except KeyError as offender:
			print(F"Warning: scenario not found ({offender})")
	else:
		assert not scenario

	discount = 1 / (1 + r) ** ((pos['Maturity'] - pos['Valuation date']) / days_in_year(pos['Maturity'].year))
	mkt_val_qccy = Money(notional * discount, qccy)
	mkt_val_pccy = convert_ccy(mkt_val_qccy, to=pccy)

	return mkt_val_pccy


# Implement the "IR" scenario
for scenario in {"IR up", "IR down"}:
	for (i, pos) in df_ptf[ipos_fx_fwd].iterrows():
		# Record the effect of the scenario
		df_ptf.loc[i, scenario] = value_pos_fx_fwd(pos, scenario)

"""
			Printout of the shock columns
"""

print(
	df_ptf[
		["Asset type", "Asset ID", "FX up", "FX down", "IR up", "IR down", "EQ down", "Spread up"]
	].to_markdown(
		# https://pypi.org/project/tabulate/
		showindex=False, headers='keys', tablefmt='psql'
	)
)

"""
			Save to file
"""

# Remove the currency unit
for col in df_ptf.columns:
	if any(isinstance(x, Money) for x in df_ptf[col]):
		df_ptf[col] = df_ptf[col].apply(float)

# https://xlsxwriter.readthedocs.io/example_pandas_datetime.html
writer = pd.ExcelWriter(
	PARAM['Report trg'],
	datetime_format="MM/DD/YYYY",
	engine='openpyxl',
)

df_ptf.to_excel(
	writer,
	index=False, na_rep="",
	sheet_name="Portfolio",
)

writer.save()

print("Saved to:", os.path.relpath(str(PARAM['Report trg']), os.path.curdir))
