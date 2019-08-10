
# RA, 2019-07-09

import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt


PARAM = {
	'data_file': "OUTPUT/histo/histo.dat",
	'bins': ["XS", "S", "M", "L", "XL"],
	'output_img': "OUTPUT/histo/img/histo.{ext}",
}


def plot_histo(histo):

	histo = histo['df'].set_index('prediction_size')

	bins = PARAM['bins']
	values = histo['count'][bins]

	(fig, ax) = plt.subplots()
	fig: plt.Figure
	ax: plt.Axes

	ax.bar(x=range(len(values)), height=values)

	ax.set_xticks(list(range(len(bins))))
	ax.set_xticklabels(bins)

	fn = PARAM['output_img'].format(ext='png').replace(' ', '_')

	fig.savefig(
		fn,
		bbox_inches='tight', pad_inches=0,
		dpi=300
	)

	plt.show()
	plt.close(fig)


def main():
	os.makedirs(os.path.dirname(PARAM['output_img']).format(), exist_ok=True)

	with open(PARAM['data_file'], 'rb') as fd:
		data = pickle.load(fd)

	plot_histo(data['histo'])


if __name__ == "__main__":
	main()
