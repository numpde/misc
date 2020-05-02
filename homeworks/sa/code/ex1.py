# RA, 2020-04-29
# Python 3.7

# SA HWK Ex 1
# https://gist.github.com/numpde/dcb92e394b61e14a0c62168b5958ac45

from pathlib import Path

import pandas as pd
from datetime import datetime as dt

# Note: some computations are performed using Decimal
from decimal import Decimal

# Provides the money unit and automatic conversion (once set up)
# Note!: (2 + XMoney(1, 'USD')) does not raise an exception
# However, (2.0 + ...) does
from money import XMoney as Money
from money import xrates as money_rates

import sys, os

# # Redirect "print" (avoid logger for this task)
# sys.stdout = open("output-ex1.txt", "w")


"""
			Constants
"""

PARAM = {
	# Whichever works first
	'Report srcs': [
		Path(__file__).parent.parent / "data/UV/1/Sample TPT V3 Report.xlsx",
		Path(__file__).parent / "Sample TPT V3 Report.xlsx",
		Path.cwd() / "Sample TPT V3 Report.xlsx",
	],

	'Report trg': Path(__file__).parent / "UV/TPT_restored.xlsx",

	# Cash positions can be identified by their CIC code:
	'Cash CIC': "XT71",
}

"""
			Helpers
"""


def singleton(s: pd.Series):
	"""
	If the series has only one distinct element
	return it, otherwise raise an AssertionError
	"""
	assert (1 == len(set(s))), "Series is not a singleton set"
	return set(s).pop()


"""
			Load the TPT report
			Extract some columns of interest
"""

for filepath in PARAM['Report srcs']:
	try:
		df_report = pd.read_excel(filepath)
		print(F"Loaded report template from: {os.path.relpath(str(filepath), os.path.curdir)}")
		break
	except FileNotFoundError:
		continue

# # Need to fix date parsing
# # Don't do this in view of Task 5, use pd.to_datetime instead
# for c in df_report:
# 	if c.lower().endswith("date"):
# 		df_report[c] = pd.to_datetime(df_report[c])

# The column numbers are standardized
# in the Solvency II Tripartite Template
# http://sst-fundreporting.solvencyanalytics.com/20151012-solvencyiitptversion3.pdf
col4_por_ccy = df_report['4_Portfolio-PortfolioCurrency']
col5_totnet = df_report['5_Portfolio-TotalNetAssets']
col8_shareprice = df_report['8_Portfolio-ShareClass-SharePrice']
col8b_totshares = df_report['8b_Portfolio-ShareClass-TotalNumberOfShares']
col12_cic = df_report['12_Position-InstrumentCIC']
col14_instcode = df_report['14_Position-InstrumentCode-Code']
col39_maturity = df_report['39_Position-IntRateInst-Redemption-MaturityDate']
col22_value_qc = df_report['22_Position-Valuation-MarketValueQC']
col24_value_pc = df_report['24_Position-Valuation-MarketValuePC']
col21_pos_ccy = df_report['21_Position-Valuation-QuotationCurrency']
col26_posweight = df_report['26_Position-Valuation-PositionWeight']
col97_scr_U = df_report['97_Position-ContributionToSCR-MktIntUp']
col98_scr_D = df_report['98_Position-ContributionToSCR-MktintDown']  # Inconsistency

col39_maturity: pd.Series
col39_maturity = pd.to_datetime(col39_maturity)

""" 
			A few observations and assumptions about the data
"""

# There only one portfolio in the table (nonzero)
assert singleton(df_report['1_Portfolio-PortfolioID-Code'])

# It has a uniquely defined price (nonzero)
assert singleton(col8_shareprice)

# These columns have no missing data
assert all(col22_value_qc.notnull())
assert all(col24_value_pc.notnull())

# '4_Portfolio-PortfolioCurrency' is consistent (and nonempty)
assert singleton(col4_por_ccy)

# '26_Position-Valuation-PositionWeight' adds up to 1
assert (1e-6 > abs(1 - col26_posweight.sum()))

"""
			Setup the money units and conversion
"""

# Extract the '4_Portfolio-PortfolioCurrency'
portfolio_ccy = singleton(col4_por_ccy)

# Currency unit attacher
to_portfolio_ccy = (lambda x: Money(x, portfolio_ccy))

# Attempt to extract the relevant exchange rates
# For simplicity, average the available guesses
exchange_rates = pd.DataFrame({
	'P': col4_por_ccy,
	'Q': col21_pos_ccy,
	'P2Q': col22_value_qc / col24_value_pc,
}).dropna().groupby(['P', 'Q']).mean()

# Allow the Money class to auto-convert
# See https://pypi.org/project/money/#currency-exchange
money_rates.install('money.exchange.SimpleBackend')
money_rates.base = portfolio_ccy
for ((p, q), c) in exchange_rates['P2Q'].items():
	assert (p == portfolio_ccy)
	money_rates.setrate(q, 1 / Decimal(c))

# Sanity check, specific to the current task
assert (Money(1, 'CHF').to('USD') < Money(1, 'USD'))

"""
	Task 1a (and 1b)
	Compute the total net assets and total number of shares
	
	Assume that the value of an asset is
		its position market value (in portfolio currency)
			24_Position-Valuation-MarketValuePC
	
	Assume that 
		total net assets = (share price) x (total number of shares)
	[This doesn't give an integer number, but I don't know that it should] 
"""

print("-" * 60)

# Asset values is position value in portfolio currency (possibly zero)
# No further weights
asset_value = df_report['24_Position-Valuation-MarketValuePC'].apply(to_portfolio_ccy)

# Compute '5_Portfolio-TotalNetAssets'
portfolio_tnav: Money
portfolio_tnav = sum(asset_value)

# We can also recover the approx total net from
# '26_Position-Valuation-PositionWeight', which is defined as
# "Market valuation in portfolio currency / portfolio net asset value in %"
# i.e.:
# ((asset_value / df_report['26_Position-Valuation-PositionWeight'].apply(Decimal)).mean())

# Share price in portfolio currency
portfolio_shareprice = Money(singleton(col8_shareprice), portfolio_ccy)

# Hence '8b_Portfolio-ShareClass-TotalNumberOfShares' is
number_of_shares = float(portfolio_tnav / portfolio_shareprice)

# Report the total net asset value (in portfolio currency)
df_report['5_Portfolio-TotalNetAssets'] = float(portfolio_tnav.to(portfolio_ccy).amount)
# and the total number of shares
df_report['8b_Portfolio-ShareClass-TotalNumberOfShares'] = number_of_shares

print(F"Total net asset value: {portfolio_tnav.to('USD')} ~ {portfolio_tnav.to('CHF')}")
print(F"Total number of shares: ~{round(number_of_shares, 3)}")

"""
	Task 2
	Filtering cash positions
"""

print("-" * 60)

# Total cash item value
cash_tnav = asset_value[col12_cic == PARAM['Cash CIC']].sum()
print(F"Total cash item value: {cash_tnav.to('USD')}")

# Distribution of currencies in percent
ccy_dist = 100 * col21_pos_ccy.value_counts(dropna=False, normalize=True)
print(
	F"Currencies:",
	", ".join("{} ({}%)".format(*cp) for cp in ccy_dist.round(2).items())
)

# Amount of cash of the fund / total net asset value of the fund, in %
cash_percentage = float(100 * (cash_tnav / portfolio_tnav))
df_report['9_Portfolio-CashPercentage'] = cash_percentage
print(F"Cash percentage: {cash_percentage:.4}%")

"""
	Task 3a
	Order instrument codes by date
"""

print("-" * 60)

# Size of sublist to display
N = 10

# Select the maturity dates of interest (decreasing order)
maturities_a = col39_maturity.dropna().nlargest(n=N)

# Get the corresponding instrument codes
codes_by_maturity_a = col14_instcode[maturities_a.index]

print(
	"Instrument codes by decreasing maturity date",
	*(
		F"{d.strftime('%Y-%m-%d')}: {c}"
		for (d, c) in zip(maturities_a, codes_by_maturity_a)
	),
	"...", "", sep="\n"
)

"""
	Task 3b
	
	Order maturity dates in increasing order
	but reversing the month
"""


# Converts a date to the tuple (Year, -Month, Day)
ynd = (lambda d: (d.year, -d.month, d.day))
# Sort according to `ynd`
maturities_b = col39_maturity[col39_maturity.dropna().apply(ynd).sort_values().index]
maturities_b = maturities_b[-N:]

# Get the corresponding instrument codes
codes_by_maturity_b = col14_instcode[maturities_b.index]

print(
	"Instrument codes by increasing' maturity date",
	"...",
	*(
		F"{d.strftime('%Y-%m-%d')}: {c}"
		for (d, c) in zip(maturities_b, codes_by_maturity_b)
	),
	"", sep="\n"
)

"""
	Task 4
	Statistics of Contribution/To/SCR -- Up and Down
"""

print("-" * 60)

# "transform the values first to percentage of the position value"
scr_U = 100 * col97_scr_U / col24_value_pc
scr_D = 100 * col98_scr_D / col24_value_pc

# Note: 50% percentile = median
percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# Compile stats
stats = pd.DataFrame(data={
	"^, %": scr_U.describe(percentiles=percentiles),
	"v, %": scr_D.describe(percentiles=percentiles)
})

stats = stats.T.rename(columns={"50%": "median"}).T

print("Descriptive analysis, Contribution/To/SCR")
print(stats.to_markdown())

"""
	Task 5
	Dump to Excel file
"""

print("-" * 60)

# Prefer openpyxl (MIT/Expat) over xlsxwriter (2-clause BSD) ?
df_report.to_excel(PARAM['Report trg'], index=False, na_rep="", sheet_name="TPT", engine='openpyxl')

print("Saved to:", os.path.relpath(str(PARAM['Report trg']), os.path.curdir))

# [I do not have Excel to verify, but in LibreOffice Calc,]
# [the original and the saved file look similar enough    ]
