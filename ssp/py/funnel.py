
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

	'partner_keys': ["Partner A", "Partner B"],

	'output': "OUTPUT/funnel/funnel.{ext}",
}

def fetch_product_domains(conn, partner_key):
	sql = "select product_domain, count(*) from {table} where (partner_key = '{partner_key}') group by product_domain;"
	sql = sql.format(table=PARAM['table'], partner_key=partner_key)
	df = pd.read_sql(sql, conn)
	return df


def funnel_by_partner_and_pd(conn, partner_key, product_domain):
	logger.info("Processing: {partner_key} -- {product_domain}".format(partner_key=partner_key, product_domain=product_domain))

	assert(product_domain is not None), "Not implemented"

	# Note: match strings using LIKE in order to allow wildcard '%'

	sql = (
		"""
			select event_name, count(event_name) as n
			from {table} 
			where (partner_key = '{partner_key}') and (product_domain LIKE '{product_domain}')
			group by event_name;
		"""
		).format(
			table=PARAM['table'],
			partner_key=partner_key,
			product_domain=product_domain,
		)

	df = pd.read_sql(sql, conn)

	logger.info("Funnel: \n{df}".format(df=df))

	query = {
		'sql': sql,
		'df': df,
	}

	return query


def funnel_by_partner(conn, partner_key):
	product_domains = fetch_product_domains(conn, partner_key)
	product_domains = list(product_domains['product_domain']) + ['%']

	funnel = {}

	for pd in product_domains:
		if pd:
			funnel[pd] = funnel_by_partner_and_pd(conn, partner_key, pd)
		else:
			funnel[pd] = None
			logger.warning("Skipping pd = {pd}".format(pd=pd))

	return funnel


def funnel_all(conn):
	funnel = {}

	for partner_key in PARAM['partner_keys']:
		funnel[partner_key] = funnel_by_partner(conn, partner_key)

	return funnel


def fetch_overview(conn):
	sql = """
		select product_domain, partner_key, count(event_name) as n 
		from {table}
		group by product_domain, partner_key 
		order by product_domain, partner_key;
	""".format(
		table=PARAM['table']
	)

	df = pd.read_sql(sql, conn)

	query = { 'sql': sql, 'df': df, }

	return query


def work(conn):
	overview = fetch_overview(conn)
	funnel = funnel_all(conn)

	output = {
		'PARAM': PARAM,
		'overview': overview,
		'funnel': funnel,
		'UTC': dt.datetime.utcnow(),
	}

	with open(PARAM['output'].format(ext='dat'), 'wb') as fd:
		pickle.dump(output, fd)


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
