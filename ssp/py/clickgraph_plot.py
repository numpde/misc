
# RA, 2019-07-20

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, pow

PARAM = {
	'input_graph': "../dbeaver/clickgraph_201907210233.csv",

	'event_order': [
		'exit',
		'--dummy--',
		'viewed_product',
		'opened_editor',
		'opened_brand_list',
		'selected_brand',
		'selected_category',
		'selected_size',
		'completed_profiling',
		'added_variant_to_cart',
		'ordered_variant',
	],

	'fig': "OUTPUT/clickgraph/clickgraph_sameproduct={sameproduct}_zoom={zoom}.{ext}",
}

df = pd.read_csv(PARAM['input_graph'])
#print(df)

df['product_id_changed'] = df['product_id_changed'].fillna(False)
df['event2'] = df['event2'].fillna('exit')

for sameproduct in [True, False]:
	g = nx.DiGraph()
	for (i, row) in df.iterrows():
		if ((not row['product_id_changed']) == sameproduct):
			g.add_edge(
				"[" + row['event1'] + "]",
				"(" + row['event2'] + ")",
				weight=(row.get('avg') or row.get('rel_freq'))
			)


	nodes1 = [n for n in g.nodes() if n.startswith("[")]
	nodes2 = [n for n in g.nodes() if n.startswith("(")]

	pos1 = {n: (-10, PARAM['event_order'].index(n[1:-1])) for n in nodes1}
	pos2 = {n: (+10, PARAM['event_order'].index(n[1:-1])) for n in nodes2}
	pos = {**pos1, **pos2}

	for zoom in range(0, 10):
		(fig, ax) = plt.subplots()
		#pos = nx.bipartite_layout(g, nodes=[n for n in g.nodes() if not n.startswith("=>")])

		#nx.draw_networkx_nodes(g, ax=ax, pos=pos)
		nx.draw_networkx_labels(g, ax=ax, pos=pos, font_size=6)

		edges = [(u, v) for (u, v, d) in g.edges.data() if (d['weight'] > 0)]
		weights = [g.get_edge_data(u, v)['weight'] * pow(4, zoom) for (u, v) in edges]
		alphas = [max(0, pow(1 - w / 100, 3)) for w in weights]
		colors = ["C{}".format(PARAM['event_order'].index(n[1:-1])) for (n, _) in edges]
		# print(colors)
		arcs = nx.draw_networkx_edges(
			g,
			ax=ax, pos=pos, edgelist=edges,
			arrowsize=0.01, arrowstyle='-',
			width=weights, edge_color=colors, alpha=1,
			connectionstyle='arc3, rad=0.001'
		)
		#
		if alphas:
			for (i, arc) in enumerate(arcs):
				arc.set_alpha(alphas[i])

		# plt.ion()
		# plt.show()

		ax.set_xlim(-15, 15)

		for ext in ['png', 'eps']:
			fig.savefig(
				PARAM['fig'].format(sameproduct=sameproduct, zoom=zoom, ext=ext),
				bbox_inches='tight', pad_inches=0,
				dpi=300
			)

		plt.close(fig)

exit()

#############
#############
#############

# g = nx.MultiDiGraph()
# for (i, row) in df.iterrows():
# 	g.add_edge(
# 		row['event1'], row['event2'],
# 		weight=(row.get('avg') or row.get('rel_freq')),
# 		key=row['product_id_changed']
# 	)
#
#
# (fig, ax) = plt.subplots()
# pos = nx.circular_layout(g)
# nx.draw_networkx_nodes(g, ax=ax, pos=pos)
# nx.draw_networkx_labels(g, ax=ax, pos=pos)
#
# #edges1 = nx.get_edge_attributes(g, 'weight') #g.get_edge_data()
# key = True
# edges = [(u, v) for (u, v, k, d) in g.edges.data(keys=True) if (k == key)]
# weights = [(30 * g.get_edge_data(u, v, key=key)['weight']) for (u, v) in edges]
# nx.draw_networkx_edges(g, ax=ax, pos=pos, edgelist=edges, width=weights, connectionstyle='arc3, rad=0.1')
#
# key = False
# edges = [(u, v) for (u, v, k, d) in g.edges.data(keys=True) if (k == key)]
# weights = [(30 * g.get_edge_data(u, v, key=key)['weight']) for (u, v) in edges]
# nx.draw_networkx_edges(g, ax=ax, pos=pos, edgelist=edges, width=weights, connectionstyle='arc3, rad=0.05')
#
# #nx.draw_networkx_edge_labels(g, ax=ax, pos=pos)
# plt.show()
