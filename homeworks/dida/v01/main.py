
# RA, 2019-10-09


import os, re, pickle

import tensorflow as tf

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

import progressbar
from progressbar import progressbar as progress
progressbar.streams.wrap_stderr()

from zipfile import ZipFile

import dotenv
import logging as logger

# https://docs.python.org/3/library/contextlib.html#contextlib.redirect_stdout
from contextlib import redirect_stdout



# ~~~ INITIALIZATION ~~~ #

# https://pypi.org/project/python-dotenv/
dotenv.load_dotenv(".env")

# https://docs.python.org/3/library/logging.html#logging.basicConfig
logger.basicConfig(level=logger.DEBUG, format="%(levelname)-8s [%(asctime)s] : %(message)s", datefmt="%Y%m%d %H:%M:%S %Z")
logger.getLogger('matplotlib').setLevel(logger.WARNING)


# ~~~ HELPERS ~~~ #

# Example: assert(filename == makedirs(filename))
makedirs = (lambda fn: os.makedirs(os.path.dirname(fn), exist_ok=True) or fn)


# ~~~ CONSTANTS ~~~ #

PARAM = {
	# Archive containing images/*.png and labels/*.png
	'data': ZipFile("ORIGINALS/dida_test_task/data.zip", mode='r'),
	'datapass': os.getenv("DATAPASS", "").encode("UTF8"),

	# The labels for these are wrong (by inspection)
	'bogus_label_ids': ["278"],

	'training': dict(epochs=20),

	# Neural net summary
	'out_base_model_info': makedirs("OUTPUT/info/base_model.txt"),
	'out_full_model_info': makedirs("OUTPUT/info/full_model.txt"),

	# Label predictions by the trained predictor
	'out_predictions': makedirs("OUTPUT/predictions/{id}.png"),
}


# ~~~ AUXILIARY ~~~ #

def resizer(w, h):
	return (lambda img: tf.image.resize(img, (w, h)))


def normalize_max(val=1):
	return (lambda img: (img / (tf.math.reduce_max(img) or 1) * val))


# ~~~ DATA SOURCE ~~~ #

# regex should be e.g. "images/(.*).png$"
# reader should be e.g. matplotlib.pyplot.imread
def get_data_files(regex: str, reader) -> pd.Series:
	# Archive ZipFile object
	zipfile = PARAM['data']
	# fn2id: images/132.png --> 132  or  None if no regex match
	fn2id = (lambda string: next(iter(re.compile(regex).findall(string)), None))
	# Filenames matching the passed regex
	files = filter(fn2id, (f.filename for f in zipfile.filelist))
	# data = {'132': /loaded image 132/, etc.}
	data = pd.Series({
		fn2id(fn): reader(zipfile.open(fn, pwd=PARAM['datapass']))
		for fn in files
	})
	return data


def get_data():
	logger.debug("Reading images/*.png")
	images = get_data_files("images/(.*).png$", (lambda fd: plt.imread(fd)[..., :3]))

	logger.debug("Reading labels/*.png")
	labels = get_data_files("labels/(.*).png$", (lambda fd: plt.imread(fd)[..., np.newaxis]))

	df = pd.DataFrame({'image': images, 'label': labels})
	df.loc[PARAM['bogus_label_ids'], 'label'] = np.nan

	# Does not work b/c of nan values
	#df = df.transform(resizer(128, 128))

	logger.debug("Got {n} images and {m} labels".format(n=df['image'].count(), m=df['label'].count()))

	return df


def make_dataset(df: pd.DataFrame):
	# Sanity check
	img = next(iter(df['image']))
	logger.debug("Shape before / after: {} / {}".format(img.shape, resizer(128, 128)(tf.convert_to_tensor(img)).shape))

	logger.debug("Creating tf dataset")

	X = tf.stack(list(map(resizer(128, 128), df['image'])))
	y = tf.stack(list(map(normalize_max(1), map(resizer(128, 128), df['label']))))

	# X = tf.stack([tf.stack(df_train['image'])], axis=1)

	logger.debug("X.shape = {}, y.shape = {}".format(X.shape, y.shape))

	ds_train = tf.data.Dataset.from_tensor_slices((X, y))
	ds_train = ds_train.batch(1)

	[(img_batch, lbl_batch)] = ds_train.take(1)
	logger.debug("Dataset first batch: X.shape = {}, y.shape = {}".format(img_batch.shape, lbl_batch.shape))

	return ds_train


# ~~~ EXPLORATION ~~~ #

def see_data(df: pd.DataFrame):
	for (i, row) in df.iterrows():
		fig: plt.Figure
		ax1: plt.Axes
		ax2: plt.Axes
		(fig, (ax1, ax2)) = plt.subplots(1, 2)
		df['image'].isna()[i] or ax1.imshow(row['image'])
		df['label'].isna()[i] or ax2.imshow(row['label'])
		fig.suptitle(i)
		plt.show()


# ~~~ MODEL ~~~ #

# Based heavily on (2019-10-09)
# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
def upsample(filters, size, apply_dropout=False):
	"""Upsamples an input.
	Conv2DTranspose => Batchnorm => Dropout => Relu
	Args:
	filters: number of filters
	size: filter size
	apply_dropout: If True, adds the dropout layer
	Returns:
	Upsample Sequential Model
	"""

	result = tf.keras.Sequential()

	result.add(tf.keras.layers.Conv2DTranspose(
		filters, size, strides=2, padding='same', use_bias=False,
		kernel_initializer=tf.random_normal_initializer(0., 0.02)
	))

	result.add(tf.keras.layers.BatchNormalization())

	if apply_dropout:
		result.add(tf.keras.layers.Dropout(0.5))

	result.add(tf.keras.layers.ReLU())

	return result



# Based heavily on (2019-10-09)
# https://www.tensorflow.org/tutorials/images/segmentation
def make_model():
	# Pretrained model for the encoder
	# https://arxiv.org/abs/1801.04381
	# https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html
	encoder = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

	# Sanity check
	img = tf.cast(np.random.randn(1, 128, 128, 3), tf.float32)
	encoder([img])

	OUTPUT_CHANNELS = 2

	# Freeze the encoder
	encoder.trainable = False

	# Write summary of the BASE model to file
	with redirect_stdout(open(PARAM['out_base_model_info'], 'w')):
		encoder.summary()

	# Tap into the activations of these intermediate layers
	layer_names = [
		'block_1_expand_relu',  # 64x64
		'block_3_expand_relu',  # 32x32
		'block_6_expand_relu',  # 16x16
		'block_13_expand_relu', #  8x8
		'block_16_project',     #  4x4
	]

	tap_layers = [encoder.get_layer(name).output for name in layer_names]

	# Non-trainable feature extraction model
	down_stack = tf.keras.Model(inputs=encoder.input, outputs=tap_layers)
	down_stack.trainable = False

	up_stack = [
		upsample(512, 3), #  4x4  ->  8x8
		upsample(256, 3), #  8x8  -> 16x16
		upsample(128, 3), # 16x16 -> 32x32
		upsample(64, 3),  # 32x32 -> 64x64
	]

	def unet_model(output_channels):
		# This is the last layer of the model
		# 64x64 -> 128x128
		last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same', activation='softmax')

		inputs = tf.keras.layers.Input(shape=[128, 128, 3])
		x = inputs

		# Downsampling through the model
		skips = down_stack(x)
		x = skips[-1]
		skips = reversed(skips[:-1])

		# Upsampling and establishing the skip connections
		for (up, skip) in zip(up_stack, skips):
			x = up(x)
			concat = tf.keras.layers.Concatenate()
			x = concat([x, skip])

		x = last(x)

		return tf.keras.Model(inputs=inputs, outputs=x)

	model = unet_model(OUTPUT_CHANNELS)

	# Write summary of the FULL model to file
	with redirect_stdout(open(PARAM['out_full_model_info'], 'w')):
		model.summary()

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	return model


def train(model, ds_train: tf.data.Dataset):
	# Sanity check
	[(img_batch, lbl_batch)] = ds_train.take(1)
	model.predict(img_batch)
	model.predict([img_batch])

	logger.debug("Training model")
	history = model.fit(ds_train, **PARAM['training'])

	return model


# ~~~ ENTRY ~~~ #

def main():
	model = make_model()

	df = get_data()
	df = dict(list(df.groupby(df['label'].isna())))

	# Training set and out-of-sample set
	(df_train, df_new) = (df[False], df[True])
	del df

	# Convert to a tensorflow dataset
	ds_train = make_dataset(df_train)

	# Train the predictor
	model = train(model, ds_train)

	# Save predictions to disk
	logger.debug("Saving predictions")

	for df in (df_new, df_train):
		for (i, img) in pd.Series(df['image']).iteritems():
			# Input, scale down
			img = resizer(128, 128)(img)
			# Prediction, scale up (handle the batch dimension)
			lbl = model(tf.stack([img]))[0]
			lbl = resizer(256, 256)(lbl)
			# Prediction proba for class '1'
			lbl = lbl[..., 1]
			# Remove boring dimensions
			lbl = np.squeeze(lbl)

			# https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
			plt.imsave(PARAM['out_predictions'].format(id=i), lbl, cmap='gray')


if __name__ == "__main__":
	main()
