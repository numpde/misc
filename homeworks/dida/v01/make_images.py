
# RA, 2019-10-16, CC-BY

# Generate pairs of
#     image = satellite view
#     label = buildings mask
# (quick & dirty)

from helpers import commons, maps

import json, os
from glob import glob
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

from numpy.random import RandomState

from base64 import b32encode

import matplotlib as mpl
import matplotlib.pyplot as plt


PARAM = {
	'buildings': "ORIGINALS/from_josm/rosdorf.geojson",

	'out_figures': "OUTPUT/from_josm/rosdorf/UV/{kind}s/{idx}.png",
	'out_figures_bad': "OUTPUT/from_josm/rosdorf_bad/UV/{kind}s/{idx}.png",

	'mapbox' : {
		'style' : maps.MapBoxStyle.satellite,
		'token' : commons.token_for('mapbox'),
	},
}

makedirs = (lambda file: os.makedirs(os.path.dirname(file), exist_ok=True) or file)


if True:
	images = glob(PARAM['out_figures'].format(kind="image", idx="*"))
	labels = glob(PARAM['out_figures'].format(kind="label", idx="*"))
	assert(len(images) == len(labels))

	for (img, lbl) in sorted(zip(images, labels)):
		assert(os.path.basename(img) == os.path.basename(lbl))
		idx = commons.unformat(PARAM['out_figures'], img)['idx']

		(fig, (ax1, ax2)) = plt.subplots(1, 2)
		ax1.imshow(plt.imread(img))
		ax2.imshow(plt.imread(lbl))
		fig.suptitle("Keep \n" + os.path.basename(img) + "? \n (y/n) ")

		def press(event):
			if (event.key == "y"):
				plt.close(fig)
			if (event.key == "n"):
				for kind in ["image", "label"]:
					os.rename(PARAM['out_figures'].format(kind=kind, idx=idx), makedirs(PARAM['out_figures_bad'].format(kind=kind, idx=idx)))
				plt.close(fig)

		cid = fig.canvas.mpl_connect('key_press_event', press)
		plt.show()
		fig.canvas.mpl_disconnect(cid)

	exit(0)


mpl.use('agg')

random_state = RandomState(2)

data = json.load(open(PARAM['buildings'], 'r'))

BBOX = data['bbox']
[LEFT, BOTTOM, RIGHT, TOP] = BBOX
print("bbox:", BBOX)

# Note: exclude "MultiPolygon" geometry type

buildings = [
	feature['geometry']['coordinates']
	for feature in data['features']
	if feature['properties'] and feature['properties'].get('building') and (feature['geometry']['type'] == ('Polygon'))
]

# Each entry of "buildings" is a list of length 1
assert({1} == set(dict(Counter(len(b) for b in buildings))))

# Pure polygons
buildings = [list(map(tuple, b[0])) for b in buildings]

print("building example:", buildings[0])

def mask(w, h, polygon):
	img = Image.new('L', (w, h), 0)
	print(polygon)
	ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
	return np.array(img)


for i in range(1000):
	# Note: do not change the order of the "random_state" calls
	idx = b32encode(random_state.bytes(30)).decode('ascii')

	# print(mask(10, 20, buildings[0]))
	lat0 = random_state.uniform(BOTTOM, TOP)
	lon0 = random_state.uniform(LEFT, RIGHT)

	(lonw, latw) = (0.0007, 0.0007)
	(left, right) = (lon0 - lonw / 2, lon0 + lonw / 2)
	(bottom, top) = (lat0 - latw / 2, lat0 + latw / 2)

	if not (LEFT < left <= right < RIGHT): continue
	if not (BOTTOM < bottom <= top < TOP): continue

	bbox = [left, bottom, right, top]

	# Buildings in sight
	bb = [
		b
		for b in buildings
		if any((left <= lat <= right) and (bottom <= lon <= top) for (lat, lon) in b)
	]

	if not bb:
		continue

	print(idx, i, bbox, len(bb))

	for kind in ['image', 'label']:
		dpi = 100

		fig: plt.Figure
		ax: plt.Axes
		(fig, ax) = plt.subplots(figsize=(1.28, 1.28), dpi=dpi)

		if (kind == 'image'):
			background_map = maps.get_map_by_bbox(bbox, **PARAM['mapbox'])
			ax.imshow(background_map, interpolation='quadric', extent=maps.mb2ax(*bbox), zorder=-100)

		if (kind == 'label'):
			ax.imshow([[0, 0], [0, 0]], extent=maps.mb2ax(*bbox), zorder=-100, cmap='gray')
			for b in bb:
				(x, y) = zip(*b)
				ax.fill(x, y, color='white')

		ax.set_xlim(left, right)
		ax.set_ylim(bottom, top)

		ax.set_axis_off()
		fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

		savefig_params = dict(bbox_inches='tight', pad_inches=0, dpi=dpi)
		fn = PARAM['out_figures'].format(kind=kind, idx=idx)
		fig.savefig(makedirs(fn), **savefig_params)

		plt.close(fig)

		# https://askubuntu.com/questions/293672/how-can-i-batch-convert-images-to-b-w-while-preserving-the-folder-structure
		# for img in $(find . -iname '*.png'); do echo -n "Converting $img"; convert -colorspace GRAY $img $img && echo ' [Done]'; done

		#