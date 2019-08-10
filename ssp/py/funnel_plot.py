
# RA, 2019-07-09

import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt


PARAM = {
	'data_file': "OUTPUT/funnel/funnel.dat",

	'event_name_funnel': [
		#"viewed_product",
		"opened_editor",
		"opened_brand_list",
		"selected_brand",
		"selected_category",
		"selected_size",
		"completed_profiling",
	],

	'event_name_short': [
		#"product",
		"open",
		"list brands",
		"brand",
		"category",
		"size",
		"complete",
	],

	'partner_keys' : ["Partner A", "Partner B"],

	'output_img': "OUTPUT/funnel/img/product={product}.{ext}",
	'output_overview': "OUTPUT/overview.csv",
}


def plot_funnel(product_domain, funnel):

	print(product_domain)

	(fig, ax) = plt.subplots()
	fig: plt.Figure
	ax: plt.Axes

	partner_keys = sorted(funnel.keys())

	styles = dict(zip(
		PARAM['partner_keys'],
		[
			dict(linestyle='--', marker='o', color='C1', linewidth=3, markersize=10),
			dict(linestyle='--', marker='s', color='C2', linewidth=3, markersize=10),
		]
	))

	legend = []

	for partner_key in partner_keys:
		if not product_domain in funnel[partner_key]:
			continue

		query = funnel[partner_key][product_domain]

		if not query:
			continue

		N = query['df'].set_index('event_name')['n']

		print(product_domain, partner_key)

		baseline = N[PARAM['event_name_funnel'][0]]

		rates = [
			(N.get(event_name, 0) / baseline) or None
			for event_name in PARAM['event_name_funnel']
		]

		X = list(range(0, len(rates)))

		# Remove "None"
		(X, rates) = zip(*((x, r) for (x, r) in zip(X, rates) if (r is not None)))

		ax.plot(X, rates, **styles[partner_key])

		ax.set_ylim(0, max(1.5, max(ax.get_ylim())))
		ax.set_xticks(list(range(0, len(PARAM['event_name_short']))))
		ax.set_xticklabels(PARAM['event_name_short'])

		ax.grid(True)

		legend.append(partner_key)

	plt.legend(legend)

	fn = PARAM['output_img'].format(product=product_domain, ext='png').replace(' ', '_')

	fig.savefig(
		fn,
		bbox_inches='tight', pad_inches=0,
		dpi=300
	)

	# plt.show()
	plt.close(fig)


def main_funnel(funnel):

	product_domains = set.union(*(set(funnel[pk].keys()) for pk in funnel.keys()))
	# print(product_domains)

	for product_domain in product_domains:
		plot_funnel(product_domain, funnel)


def main_overview(overview_query):
	df0 = overview_query['df']
	df0: pd.DataFrame

	key = (lambda x: (x is None, x))

	df = pd.DataFrame(index=sorted(set(df0['product_domain']), key=key), columns=sorted(set(df0['partner_key']), key=key))
	df: pd.DataFrame

	for (i, row) in df0.iterrows():
		df.loc[row['product_domain'], row['partner_key']] = row['n']

	df.to_csv(PARAM['output_overview'], sep='\t')


def main():
	os.makedirs(os.path.dirname(PARAM['output_img']).format(), exist_ok=True)
	os.makedirs(os.path.dirname(PARAM['output_overview']).format(), exist_ok=True)

	with open(PARAM['data_file'], 'rb') as fd:
		data = pickle.load(fd)

	main_overview(data['overview'])
	main_funnel(data['funnel'])



if __name__ == "__main__":
	main()
