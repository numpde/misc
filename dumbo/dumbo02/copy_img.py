
# RA, 2019-09-28

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

for (i, row) in df.iterrows():

	try:
		src = filename_trg(row['Name'])
		trg = filename_trg(row['Code'])

		print("Copying {} to {}".format(src, trg))

		shutil.copyfile(src, trg)
	except:
		print("FAILED")



# # ZIP ALL
# for (i, row) in df.iterrows():
# 	print("----")
#
# 	if (row['Zip'] == 'nozip'):
# 		print("Skipping (nozip): {}".format(row['Code']))
# 		continue
#
# 	assert(np.isnan(row['Zip']))
#
# 	(a, b) = (filename_trg(row['Code']), filename_zip(row['Code']))
#
# 	if os.path.isfile(b):
# 		print("Zip already exists, skipping: {}".format(b))
# 		continue
#
# 	if not os.path.isfile(a):
# 		print("File not found: {}".format(a))
# 		continue
#
# 	try:
# 		print("Zipping {} to {}".format(a, b))
# 		zipfile.ZipFile(b, mode='w').write(a, os.path.basename(a))
# 		print("OK")
# 	except:
# 		print("FAILED")
