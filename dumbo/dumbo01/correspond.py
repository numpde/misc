
import pandas as pd
from jellyfish import levenshtein_distance

PARAM = {
	'data': "UV/products.csv",
}

df = pd.read_csv(PARAM['data'])

def norm(s: str):
	s = s.lower()
	s = s.replace(' ', '_')
	return s

images = df['Image'].dropna().apply(norm).unique()

products = df.apply(lambda r: (r['Product name'] + " " + str(r['Color'])), axis=1).transform(norm).unique()

for p in products:
	i = min(images, key=(lambda i: levenshtein_distance(i, p)))
	print(p, "==({})=>".format(levenshtein_distance(i, p)), i)
