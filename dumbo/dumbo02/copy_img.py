
# RA, 2019-09-28

import os
import shutil
import pandas as pd, numpy as np
import urllib.request
import zipfile

PARAM = {
	'url': (open("UV/url.txt", 'r').readlines()).pop(0),
	'data': "UV/products.csv",

	'color': "UV/src/{color}.JPG",
	'path_src': "UV/src/{name}.JPG",
	'path_trg': "UV/trg/{name}.JPG",
	'path_zip': "UV/zip/{name}.zip",
}

with urllib.request.urlopen(PARAM['url']) as response:
	with open(PARAM['data'], 'wb') as fd:
		fd.write(response.read())

df = pd.read_csv(PARAM['data'])
# print(df.head())

df['Combine'] = df['Combine'].fillna(0)

def filename_src(name):
	return PARAM['path_src'].format(name=name)

def filename_trg(sku):
	return PARAM['path_trg'].format(name=sku)

def filename_zip(sku):
	return PARAM['path_zip'].format(name=sku)

# for (i, row) in df.iterrows():
#
# 	try:
# 		src = filename_src(row['Name'])
# 		trg = filename_trg(row['Code'])
#
# 		print("Copying {} to {}".format(src, trg))
#
# 		shutil.copyfile(src, trg)
# 	except:
# 		print("FAILED")


for (k, group) in df.groupby(['Product']):
	group: pd.DataFrame

	try:
		assert(1 == len(set(group['Combine'])))
	except:
		print("Error for group: {}".format(k))
		print(group)
		raise

	if any(group['Combine']):

		src = list(map(filename_src, set(group['Name'])))

		try:
			src1 = {os.path.isfile(s): s for s in src}[True]
		except:
			print("No source for {}".format(src))
			continue

		for (i, row) in group.iterrows():
			src2 = PARAM['color'].format(color=row['Color'])
			trg = filename_zip(row['Code'])
			name1 = row['Product'] + ".JPG"
			name2 = row['Name'] + ".JPG"
			print("Combine {} and {} to {}".format(src1, src2, trg))

			with zipfile.ZipFile(trg, mode='w') as fd:
				fd.write(src1, name1)
				fd.write(src2, name2)

	else:

		for (i, row) in group.iterrows():
			src = filename_src(row['Name'])
			zip = filename_zip(row['Code'])

			print("Zipping {} to {}".format(src, zip))

			if not os.path.isfile(src):
				print("Not found: {}".format(src))
				continue

			with zipfile.ZipFile(zip, mode='w') as fd:
				fd.write(src, os.path.basename(src))


