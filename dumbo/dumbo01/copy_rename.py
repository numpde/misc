
# RA, 2019-09-27

import os
import shutil
import pandas as pd, numpy as np
import urllib.request
import zipfile

PARAM = {
	'url': (open("UV/url.txt", 'r').readlines()).pop(0),
	'data': "UV/products.csv",

	'path_src': "UV/src/{name}.JPG",
	'path_trg': "UV/trg/{name}.JPG",
	'path_zip': "UV/zip/{name}.zip",
}

with urllib.request.urlopen(PARAM['url']) as response:
	with open(PARAM['data'], 'wb') as fd:
		fd.write(response.read())

df = pd.read_csv(PARAM['data'])
# print(df.head())

def filename_src(name):
	return PARAM['path_src'].format(name=name)

def filename_trg(sku):
	return PARAM['path_trg'].format(name=sku)

def filename_zip(sku):
	return PARAM['path_zip'].format(name=sku)

for (k, group) in df.groupby(['Product name', 'Color']):
	print("----")

	skus = list(group['Code'])

	# # DEBUG
	# if not ('NEHERASS200701' in codes): continue

	group = group.assign(exists=list(map(os.path.isfile, map(filename_trg, skus))))

	ex = group['exists']

	if all(ex):
		continue

	template_files = set(group['Image'].dropna())

	if not template_files:
		print("No image template for group {}".format(skus))
		continue

	if (len(template_files) > 1):
		print("Multiple templates {} for group {}".format(template_files, skus))
		continue

	assert(len(template_files) == 1)

	template_file = next(iter(template_files))
	del template_files

	print("Taking {} as template for {}".format(template_file, skus))

	template_filename = filename_src(template_file)
	del template_file

	if not os.path.isfile(template_filename):
		print("File not found: {}".format(template_filename))
		continue

	for sku in group['Code'][~ex]:
		(a, b) = (template_filename, filename_trg(sku))
		print("Copying {} to {}".format(a, b))
		shutil.copyfile(a, b)

# ZIP ALL
for (i, row) in df.iterrows():
	print("----")

	if (row['Zip'] == 'nozip'):
		print("Skipping (nozip): {}".format(row['Code']))
		continue

	assert(np.isnan(row['Zip']))

	(a, b) = (filename_trg(row['Code']), filename_zip(row['Code']))

	if os.path.isfile(b):
		print("Zip already exists, skipping: {}".format(b))
		continue

	if not os.path.isfile(a):
		print("File not found: {}".format(a))
		continue

	try:
		print("Zipping {} to {}".format(a, b))
		zipfile.ZipFile(b, mode='w').write(a, os.path.basename(a))
		print("OK")
	except:
		print("FAILED")
