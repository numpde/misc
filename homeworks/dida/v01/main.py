
# RA, 2019-10-09


import os, re, pickle

import tensorflow as tf

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# import progressbar
# from progressbar import progressbar as progress
# progressbar.streams.wrap_stderr()

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


# ~~~ AUXILIARY ~~~ #

# Example: assert(filename == makedirs(filename))
makedirs = (lambda file: os.makedirs(os.path.dirname(file), exist_ok=True) or file)

# Example: resizer(128, 128)(image)
resizer = (lambda w, h: (lambda image: tf.image.resize(image, (w, h))))


# ~~~ PARAMETERS ~~~ #

PARAM = {
	# Archive containing images/*.png and labels/*.png
	'data': ZipFile("ORIGINALS/dida_test_task/data.zip", mode='r'),
	'datapass': os.getenv("DATAPASS", "").encode("UTF8"),

	# The labels for these are wrong (by inspection)
	'bogus_label_ids': ["278"],

	#
	'input_channels': 3,

	# Image sizes
	'to_original_size': resizer(256, 256),
	'to_modeling_size': resizer(128, 128),

	# Settings for model.fit
	'training': dict(epochs=50),

	# Output: Neural net summary
	'out_base_model_info': makedirs("OUTPUT/info/base_model.txt"),
	'out_full_model_info': makedirs("OUTPUT/info/full_model.txt"),

	# Output: Label predictions by the trained predictor
	'out_predictions': makedirs("OUTPUT/predictions/{idx}.png"),
}


# ~~~ HELPERS ~~~ #

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


def get_data() -> pd.DataFrame:
	logger.debug("Reading images/*.png")
	images = get_data_files("images/(.*).png$", (lambda fd: plt.imread(fd)[..., 0:PARAM['input_channels']]))

	logger.debug("Reading labels/*.png")
	labels = get_data_files("labels/(.*).png$", (lambda fd: plt.imread(fd)[..., np.newaxis]))

	df = pd.DataFrame({'image': images, 'label': labels})
	df.loc[PARAM['bogus_label_ids'], 'label'] = np.nan

	# Does not work because of missing labels values
	#df = df.transform(resizer(128, 128))

	logger.debug("Got {n} images and {m} labels".format(n=df['image'].count(), m=df['label'].count()))

	return df


# ~~~ PREPROCESSING ~~~ #

def make_tf_dataset(df: pd.DataFrame) -> tf.data.Dataset:
	# Sanity check
	img = next(iter(df['image']))
	logger.debug("Shape before / after: {} / {}".format(img.shape, PARAM['to_modeling_size'](tf.convert_to_tensor(img)).shape))

	logger.debug("Creating tf dataset")

	X = tf.stack(list(map(PARAM['to_modeling_size'], df['image'])))
	y = tf.stack(list(map(normalize_max(1), map(PARAM['to_modeling_size'], df['label']))))

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

# Taken from (2019-10-09)
# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
def upsample_layer(filters, size, apply_dropout=True):
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
	# Pretrained model for the encoder with frozen weights
	# https://arxiv.org/abs/1801.04381
	# https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html
	encoder = tf.keras.applications.MobileNetV2(input_shape=[128, 128, PARAM['input_channels']], include_top=False)
	encoder.trainable = False

	# Sanity check: dimensions
	img = tf.cast(np.random.randn(1, 128, 128, PARAM['input_channels']), tf.float32)
	encoder([img])

	# Write summary of the BASE model to file
	with redirect_stdout(open(PARAM['out_base_model_info'], 'w')):
		encoder.summary()

	# Construct the U-Net predictor
	def unet():
		# Number of prediction classes
		OUTPUT_CHANNELS = 2

		# Input layer of the model
		inputs = tf.keras.layers.Input(shape=[128, 128, PARAM['input_channels']])

		# Last layer of the model (output layer)
		# 64x64 -> 128x128
		# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
		output = tf.keras.layers.Conv2DTranspose(activation='softmax', filters=OUTPUT_CHANNELS, kernel_size=3, strides=2, padding='same')

		# Tap into the activations of intermediate layers in the encoder
		tap_layers = [
			encoder.get_layer(name).output
			#
			for name in [
				'block_1_expand_relu',  # 64x64
				'block_3_expand_relu',  # 32x32
				'block_6_expand_relu',  # 16x16
				'block_13_expand_relu', #  8x8
				# "Tip" of the encoder:
				'block_16_project',     #  4x4
			]
		]

		# Frozen feature-extraction sub-model
		extractor = tf.keras.Model(inputs=encoder.input, outputs=tap_layers)
		extractor.trainable = False

		# Hook into the layers of the downsampling branch
		[tracer, *down_stack] = reversed(extractor(inputs))

		# Create the layers for the upsampling branch
		up_stack = [
			upsample_layer(filters=filters, size=3)
			#
			# Down-scaled the number of filters [512, ..., 64]
			# from the TF image-segmentation tutorial
			for filters in [
				96, #  4x4  ->  8x8
				64, #  8x8  -> 16x16
				32, # 16x16 -> 32x32
				16, # 32x32 -> 64x64
			]
		]

		assert(len(down_stack) == len(up_stack))

		# Upsample and connect feed from the encoder
		# by going left-to-right in the diagram:
		# tracer < down < down < down < down < inputs
		#      `-> up -`> up -`> up -`> up -`> output
		for (down, up) in zip(down_stack, up_stack):
			# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate
			tracer = tf.keras.layers.concatenate([down, up(tracer)], axis=-1)

		return tf.keras.Model(inputs=inputs, outputs=output(tracer))

	# Construct the U-Net predictor
	model = unet()

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

	logger.debug("Training the model")
	history = model.fit(ds_train, **PARAM['training'])

	return model


def predict_segmentation_mask(model, image):
	# Prediction (handle image size & the batch dimension)
	label = model(tf.stack([PARAM['to_modeling_size'](image)]))[0]
	label = PARAM['to_original_size'](label)
	# Prediction proba for class '1'
	label = label[..., 1]
	# Remove boring dimensions, just in case
	label = np.squeeze(label)
	return label


# ~~~ ENTRY ~~~ #

def main():
	logger.info("Loading the dataset")

	df = get_data()

	# Training set (i.e. 'label' is given) and out-of-sample set (i.e. 'label' is n/a)
	(df_train, __) = map(df.groupby(df['label'].isna()).get_group, (False, True))

	logger.info("Building the tensorflow dataset")

	ds_train = make_tf_dataset(df_train)

	logger.info("Building the predictor")

	model = make_model()

	logger.info("Training the predictor")

	model = train(model, ds_train)

	logger.info("Saving predictions to disk")

	for (idx, image) in pd.Series(df['image']).iteritems():
		# Invoke the predictor
		label = predict_segmentation_mask(model, image)

		# https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
		plt.imsave(PARAM['out_predictions'].format(idx=idx), label, cmap='gray', vmax=1)


if __name__ == "__main__":
	main()
