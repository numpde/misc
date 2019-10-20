
# RA, 2019-10-18

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("agg")

import json
import pandas as pd

PARAM = {
	'default_logs': "OUTPUT/training/last_history.txt",
	'out_history_plot': "OUTPUT/training/last_history.png",
}

def plot(logs: dict):

	logs = pd.DataFrame(logs)

	# with plt.style.context(('dark_background')):
	fig: plt.Figure
	ax1: plt.Axes
	(fig, ax1) = plt.subplots()

	ax1.plot(logs['val_loss'], label="validation")
	ax1.plot(logs['loss'], label="in-sample")
	ax1.plot(0, 0, label=None)
	ax1.plot(0, 1e-1, label=None)
	ax1.grid(True)
	ax1.legend()
	# ax1.set_xscale('log')
	# ax1.set_yscale('log')
	ax1.set_xlabel("Training epoch")

	fig.savefig(PARAM['out_history_plot'], bbox_inches='tight', pad_inches=0, dpi=100)
	plt.close(fig)


if __name__ == "__main__":
	logs = json.load(open(PARAM['default_logs'], 'r'))
	plot(logs)
