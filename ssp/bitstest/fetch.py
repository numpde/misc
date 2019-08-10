
# RA, 2019-07-22

import pandas as pd
import pyodbc
import json
import pickle
import datetime as dt
import os

import logging as logger

import dotenv
dotenv.load_dotenv()

PARAM = {
	'DB': [
		"DRIVER={PostgreSQL Unicode}",
		"SERVER=" + os.getenv("SSP_SERVER"),
		"DATABASE=ssp",
		"UID=" + os.getenv("SSP_UID"),
		"PWD=" + os.getenv("SSP_PWD"),
		"PORT=5432",
	],

	'query': open("query.sql", 'r').read(),

	'output': "query_result.{ext}",
}


def fetch(conn):
	df = pd.read_sql(PARAM['query'], conn)
	return df


def save(df):
	df.to_csv(PARAM['output'].format(ext="csv"), sep='\t')
	pickle.dump(df, open(PARAM['output'].format(ext='dat'), 'wb'))


def main():
	logger.info("Establishing DB connection...")

	with pyodbc.connect(";".join(PARAM['DB'])) as conn:
		conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
		conn.setencoding(encoding='utf-8')

		try:
			cursor = conn.cursor()
		except:
			logger.exception("Cursor failed")

		logger.info("Running query...")
		df = fetch(conn)

		logger.info("Saving the result...")
		save(df)


if __name__ == "__main__":
	main()
