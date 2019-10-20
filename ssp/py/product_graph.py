
# RA, 2019-08-15

import networkx as nx
import pandas as pd
import numpy as np

from itertools import groupby

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger()


PARAM = {
	'data': "../dbeaver/vo_201908141441.csv.zip",

}


# ~~~ Read data from disk ~~~
def get_data():
	df = pd.read_csv(PARAM['data'], dtype={'product_id': str})
	return df


# ~~~ Collapse a MultiDiGraph to a weighted DiGraph ~~~
def collapse_multi(mdg: nx.MultiDiGraph) -> nx.DiGraph:
	g = nx.DiGraph()
	for ((u, v), group) in groupby(sorted(mdg.edges())):
		g.add_edge(u, v, weight=len(list(group)))
	return g


def fake_graph(size, p):
	g = nx.MultiDiGraph()
	g.add_nodes_from(range(size))

	rs = np.random.RandomState(41)
	for _ in range(5 * size):
		[nodes, degrees] = map(list, zip(*sorted(g.degree, key=(lambda nd: -nd[1]))))
		degrees = 1 + np.asarray(degrees, dtype=float)
		[v, u] = rs.choice(nodes, size=2, p=(degrees / sum(degrees)))
		g.add_edge(u, v)

	freq = nx.degree_histogram(g)
	(fig, ax) = plt.subplots()
	ax.plot(freq, '-*')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.grid()
	plt.show()



#
def make_graph(df):
	mdg = nx.MultiDiGraph()

	for (n, (userid, group)) in enumerate(df.groupby('domain_userid')):
		if (n >= 10000): break
		product_list = list(map(str, group['product_id']))
		for (u, v) in zip(product_list, product_list[1:]):
			mdg.add_edge(u, v)

	g = collapse_multi(mdg)

	# print("Connected components:", (nx.number_connected_components(nx.Graph(g))))
	(edges, weights) = zip(*(nx.get_edge_attributes(g, 'weight').items()))

	# nx.draw(g, edgelist=edges, node_size=0, width=5, edge_color=weights)
	# plt.show()

	freq = nx.degree_histogram(mdg)
	(fig, ax) = plt.subplots()
	ax.plot(freq, '-*')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.grid()
	plt.show()

	# # Adjacency matrix sparsity pattern
	# m = nx.to_numpy_matrix(g)
	# (fig, ax) = plt.subplots()
	# ax: plt.Axes
	# ax.spy(m)
	# plt.show()


# ~~~ Driver ~~~
def main():
	fake_graph(1000, 0.6)
	exit(39)

	print("Loading dataset")
	df = get_data()
	print("Got dataset with {} rows".format(len(df)))

	print("Constructing the product graph")
	make_graph(df)


if (__name__ == "__main__"):
	main()
