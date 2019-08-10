
# RA, 2019-07-09

import pandas as pd
import pyodbc
import logging as logger
import json
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

	'output': "OUTPUT/histo/histo.{ext}",
}


def fetch_histo(conn):
	sql = (
		"""
			select prediction_size, count(*) 
			from {table} 
			where (partner_key = 'Partner A') and (product_domain = 'Dresses') and (event_name = 'completed_profiling') 
			group by prediction_size;
		"""
	).format(
		table=PARAM['table']
	)

	df = pd.read_sql(sql, conn)

	return {'sql': sql, 'df': df}


def work(conn):
	histo = fetch_histo(conn)

	output = {
		'PARAM': PARAM,
		'histo': histo,
		'UTC': dt.datetime.utcnow(),
	}

	with open(PARAM['output'].format(ext='dat'), 'wb') as fd:
		pickle.dump(output, fd)


# Establish a connection to the DB and pass on control
def main():
	os.makedirs(os.path.dirname(PARAM['output']).format(), exist_ok=True)

	print(PARAM)

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
