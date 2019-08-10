
# RA, 2019-07-09

from math import sqrt
from scipy.stats import norm
import pandas as pd
import pyodbc
import logging as logger
import pickle
import datetime as dt
import os

import dotenv
dotenv.load_dotenv()

# PostgreSQL driver:
# sudo apt install odbc-postgresql

PARAM = {
	'DB': [
		"DRIVER={PostgreSQL Unicode}",
		"SERVER=" + os.getenv("SSP_SERVER"),
		"DATABASE=ssp",
		"UID=" + os.getenv("SSP_UID"),
		"PWD=" + os.getenv("SSP_PWD"),
		"PORT=5432",
	],

	'table': "analytics.events",

	'output': "OUTPUT/hypo/hypo.txt",
}


def fetch_hypo(conn):
	sql = (
		"""
			select 
				event_name, 
				sum((ab_slot1_variant = 'Control')::int) as "Control", 
				sum((ab_slot1_variant = 'Test')::int) as "Test" 
			from {table} 
			where 
				(partner_key = 'Partner A') 
				and
				(
					(event_name = 'viewed_product') or 
					(event_name = 'ordered_variant')
				)
			group by event_name;
		"""
	).format(
		table=PARAM['table']
	)

	df = pd.read_sql(sql, conn)

	return {'sql': sql, 'df': df}


def work(conn):
	histo = fetch_hypo(conn)['df'].set_index('event_name')

	p = histo['Control']['ordered_variant'] / histo['Control']['viewed_product']
	q = histo['Test']['ordered_variant'] / histo['Test']['viewed_product']
	n = histo['Test']['viewed_product']

	z = (q - p) / sqrt(p * (1 - p) / n)
	alpha = 1 - norm.cdf(z)

	with open(PARAM['output'], 'w') as fd:
		fd.write("\n".join([
			"p = {}".format(p),
			"p' = {}".format(q),
			"n' = {}".format(n),
			"z = {}".format(z),
			"alpha = {}".format(alpha),
		]))


# Establish a connection to the DB and pass on control
def main():
	os.makedirs(os.path.dirname(PARAM['output']).format(), exist_ok=True)

	with pyodbc.connect(";".join(PARAM['DB'])) as conn:
		# https://github.com/mkleehammer/pyodbc/wiki/Connecting-to-PostgreSQL
		conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
		conn.setencoding(encoding='utf-8')

		try:
			cursor = conn.cursor()
		except:
			logger.exception("Cursor failed")
		else:
			work(conn)


if __name__ == "__main__":
	main()
