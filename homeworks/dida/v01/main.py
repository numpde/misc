
# RA, 2019-10-09, CC0

# Label houses in satellite images

# Based mostly on
# https://www.tensorflow.org/tutorials/images/segmentation


import os, re, json

import tensorflow as tf

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
from matplotlib.pyplot import imsave, imread

from zipfile import ZipFile

import dotenv
import logging as logger

# https://docs.python.org/3/library/contextlib.html#contextlib.redirect_stdout
from contextlib import redirect_stdout as stdout_to

from . import plot_history

# https://pypi.org/project/percache/
# import percache


# ~~~ INITIALIZATION ~~~ #

# https://pypi.org/project/python-dotenv/
dotenv.load_dotenv(".env")

# https://docs.python.org/3/library/logging.html#logging.basicConfig
logger.basicConfig(level=logger.DEBUG, format="%(levelname)-8s [%(asctime)s] : %(message)s", datefmt="%Y%m%d %H:%M:%S %Z")
logger.getLogger('matplotlib').setLevel(logger.WARNING)

# Ping tensorflow
logger.info("TF version: {}".format(tf.__version__))
tf.cast([0], dtype=tf.float32)

# https://pypi.org/project/percache/
# cache = percache.Cache("/tmp/dida-cache", livesync=True)


# ~~~ AUXILIARY ~~~ #

# Example: assert(filename == makedirs(filename))
makedirs = (lambda file: os.makedirs(os.path.dirname(file), exist_ok=True) or file)

# Example: resizer(128, 128)(image)
resizer = (lambda w, h: (lambda image: tf.image.resize(image, (w, h))))


# ~~~ PARAMETERS ~~~ #

PARAM = {
	# Archives containing images/*.png and labels/*.png
	'dida_zip': "ORIGINALS/dida_test_task/data.zip",
	'josm_zip': "OUTPUT/from_josm/rosdorf/data.zip",
	'datapass': os.getenv("DATAPASS", "").encode("UTF8"),

	'use_dida_data': True,
	'use_josm_data': False,

	# The labels for these are wrong (by inspection)
	'bogus_label_ids': ["278"],

	# Image sizes
	'to_original_size': resizer(256, 256),
	'to_modeling_size': resizer(128, 128),

	# Training / validation split
	'validation_frac': 0.2,

	#
	'transmogrify_rs': np.random.RandomState(8),

	# Settings for model.fit
	'training': dict(epochs=33),

	# Output: Neural net summary
	'out_base_model_info': makedirs("OUTPUT/info/base_model.txt"),
	'out_full_model_info': makedirs("OUTPUT/info/full_model.txt"),

	# Output: training progess
	'out_intraining_pred': makedirs("OUTPUT/training/progress/intraining_{kind}_pred.png"),
	'out_intraining_true': makedirs("OUTPUT/training/progress/intraining_{kind}_true.png"),

	# Output: model training checkpoints and last history
	'out_training_checkpoint': makedirs("OUTPUT/training/checkpoints/UV/model_checkpoint_{epoch}"),
	'out_training_history': makedirs("OUTPUT/training/last_history.txt"),

	# Output: Label predictions by the trained predictor
	'out_predictions': makedirs("OUTPUT/predictions/{idx}.png"),
}


# ~~~ HELPERS ~~~ #

def normalize_max(val=1):
	return (lambda img: (img / (tf.math.reduce_max(img) or 1) * val))


# # sparse y_true vs one-hot y_pred
# # ValueError: tf.function-decorated function tried to create variables on non-first call.
# @tf.function
# def IoU_1xN(y_true, y_pred):
# 	return tf.keras.metrics.MeanIoU(num_classes=(y_pred.shape[-1]))(y_true[..., 0], tf.argmax(y_pred, axis=-1))


# ~~~ PREPROCESSING I ~~~ #

def to_rg_noblue(image):
	assert(3 == len(image.shape)), "The image should have 3 dimensions: W x H x Channel"
	assert(image.shape[2] in [3, 4]), "The image should have 3 or 4 channels"
	if (image.shape[2] == 3):
		# Introduce an all-zero alpha channel
		image = tf.pad(image, [(0, 0), (0, 0), (0, 1)])
	return image


def to_grayscale(image):
	if image.shape[2:]:
		# If there are more than 2 dimensions, it should be 3
		assert(3 == len(image.shape))
		# Drop the alpha channel, convert to grayscale, 2d
		image = tf.image.rgb_to_grayscale(image[..., 0:3])[..., 0]
	assert(2 == len(image.shape))
	return image


# ~~~ DATA SOURCE ~~~ #

# regex should be e.g. "images/(.*).png$"
# reader should be e.g. matplotlib.pyplot.imread
def unzip_files(filename: str, regex: str, reader) -> pd.Series:
	# Access the zip archive
	zipfile = ZipFile(filename, mode='r')
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


def get_data_from(zipfile: str) -> pd.DataFrame:
	logger.debug("Reading {}/images/*.png".format(zipfile))
	images = unzip_files(zipfile, "images/(.*).png$", (lambda fd: to_rg_noblue(imread(fd))))

	logger.debug("Reading {}/labels/*.png".format(zipfile))
	labels = unzip_files(zipfile, "labels/(.*).png$", (lambda fd: to_grayscale(imread(fd))[..., np.newaxis]))

	df = pd.DataFrame({'image': images, 'label': labels, 'source': zipfile})

	# Images and labels are already normalized
	assert(1 >= tf.math.reduce_max(tf.stack(df['image'])))
	assert(1 >= tf.math.reduce_max(tf.stack(df['label'].dropna())))

	logger.debug("Got {n} images and {m} labels".format(n=df['image'].count(), m=df['label'].count()))

	return df


def get_data() -> pd.DataFrame:
	dfs = []

	if PARAM['use_dida_data']:
		df = get_data_from(PARAM['dida_zip'])
		df.loc[PARAM['bogus_label_ids'], 'label'] = np.nan
		dfs.append(df)

	if PARAM['use_josm_data']:
		df = get_data_from(PARAM['josm_zip'])
		dfs.append(df)

	df = pd.concat(dfs, axis=0, verify_integrity=True)

	# for (i, row) in df.iterrows():
	# 	try:
	# 		logger.debug("image.shape = {}, label.shape = {}, row = {}".format(row['image'].shape, row['label'].shape, i))
	# 	except:
	# 		pass

	return df


# ~~~ PREPROCESSING II ~~~ #

# Data augmentation worker
@tf.function
def transmogrify_datapoint(image, label):
	random = PARAM['transmogrify_rs']

	k = random.randint(0, 4)
	if k:
		image = tf.image.rot90(image, k=k, name=None)
		label = tf.image.rot90(label, k=k, name=None)

	if random.randint(0, 2):
		image = tf.image.flip_left_right(image)
		label = tf.image.flip_left_right(label)

	# # https://cs230-stanford.github.io/tensorflow-input-data.html#building-an-image-data-pipeline
	# image[..., 0:3] = tf.image.random_brightness(image[..., 0:3], max_delta=(32 / 255))
	# image[..., 0:3] = tf.image.random_saturation(image[..., 0:3], lower=0.5, upper=1.5)
	# image[..., 0:3] = t  f.clip_by_value(image[..., 0:3], 0, 1)

	return (image, label)


# Convert pd dataframe to tf dataset, preprocess
def make_tf_dataset(df: pd.DataFrame) -> tf.data.Dataset:
	SHUFFLE_BUFFER = 1024
	BATCH_SIZE = 16

	# Sanity check
	img = next(iter(df['image']))
	logger.debug("Shape before / after: {} / {}".format(img.shape, PARAM['to_modeling_size'](tf.convert_to_tensor(img)).shape))

	logger.debug("Creating tf dataset")

	X = tf.stack(list(map(PARAM['to_modeling_size'], df['image'])))
	y = tf.stack(list(map(PARAM['to_modeling_size'], map(normalize_max(1), df['label']))))

	logger.debug("X.shape = {}, y.shape = {}".format(X.shape, y.shape))

	ds = tf.data.Dataset.from_tensor_slices((X, y))

	# https://cs230-stanford.github.io/tensorflow-input-data.html
	ds = ds.shuffle(SHUFFLE_BUFFER, seed=12)
	ds = ds.repeat(8)
	ds = ds.map(transmogrify_datapoint, num_parallel_calls=4)
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(2)

	[(img_batch, lbl_batch)] = ds.take(1)
	logger.debug("Dataset first batch: X.shape = {}, y.shape = {}".format(img_batch.shape, lbl_batch.shape))

	return ds


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

# Based on (2019-10-09)
# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
def upsample_layer(filters, name, dropout_rate=(1/2)):

	upsampler = tf.keras.Sequential(
		[
			tf.keras.layers.Conv2DTranspose(
				2 * filters, kernel_size=3, strides=1, padding='same', use_bias=False,
				kernel_initializer=tf.random_normal_initializer(0., 0.02),
			),
			tf.keras.layers.Dropout(dropout_rate),
			tf.keras.layers.ReLU(),

			tf.keras.layers.Conv2DTranspose(
				filters, kernel_size=3, strides=2, padding='same', use_bias=False,
				kernel_initializer=tf.random_normal_initializer(0., 0.02),
			),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Dropout(dropout_rate),
			tf.keras.layers.ReLU(),
		],
		name=name
	)

	return upsampler


# Predictor definition
# Based heavily on (2019-10-09)
# https://www.tensorflow.org/tutorials/images/segmentation
def make_model():

	# Construct the U-Net predictor
	def unet():

		# Input layer of the model
		inputs = tf.keras.layers.Input(shape=[128, 128, 4])

		# Separate RGB and Alpha channels
		inputs_rgb = tf.keras.layers.Lambda(lambda x: x[..., 0:3])(inputs)
		inputs_alpha = tf.keras.layers.Lambda(lambda x: x[..., 3:])(inputs)

		# Pretrained model for the encoder with frozen weights
		# https://arxiv.org/abs/1801.04381
		# https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html
		unet_encoder = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
		unet_encoder.trainable = False

		# Sanity check: dimensions
		img = tf.cast(np.random.randn(1, 128, 128, 3), tf.float32)
		unet_encoder([img])

		# Write summary of the BASE model to file
		with stdout_to(open(PARAM['out_base_model_info'], 'w')):
			unet_encoder.summary()

		# Number of prediction classes
		OUTPUT_CHANNELS = 2

		# Last layer of the model (output layer)
		# 64x64 -> 128x128
		# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
		unet_output = tf.keras.layers.Conv2DTranspose(filters=OUTPUT_CHANNELS, kernel_size=3, strides=2, padding='same')

		# Tap into the activations of intermediate layers in the encoder
		unet_tap_layers = [
			unet_encoder.get_layer(name).output
			#
			for name in [
				# From the tf tutorial:
				'block_1_expand_relu',  # 64x64
				'block_3_expand_relu',  # 32x32
				'block_6_expand_relu',  # 16x16
				'block_13_expand_relu', #  8x8
				# 'block_16_project',     #  4x4
			]
		]

		# Frozen feature-extraction sub-model
		unet_extractor = tf.keras.Model(inputs=unet_encoder.input, outputs=unet_tap_layers)
		unet_extractor.trainable = False

		# Hook into the layers of the downsampling branch
		[tracer, *unet_down_stack] = reversed(unet_extractor(inputs_rgb))

		# Create the layers for the upsampling branch
		unet_up_stack = [
			upsample_layer(filters=filters, name=("upsample_{n}".format(n=n)))
			#
			for (n, filters) in enumerate([
				# 8, #  4x4  ->  8x8
				8, #  8x8  -> 16x16
				8, # 16x16 -> 32x32
				8, # 32x32 -> 64x64
			])
		]

		assert(len(unet_down_stack) == len(unet_up_stack))

		# Upsample and connect feed from the encoder
		# by going left-to-right in the diagram:
		# tracer <- down <- down <- down <- down <- inputs
		#      `--> up -`-> up -`-> up -`-> up -`-> output
		for (down, up) in zip(unet_down_stack, unet_up_stack):
			# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate
			tracer = tf.keras.layers.concatenate([down, up(tracer)], axis=-1)

		# Now combine with the alpha channel
		tracer = tf.keras.layers.concatenate([unet_output(tracer), inputs_alpha])

		tracer = tf.keras.layers.Conv2D(filters=OUTPUT_CHANNELS, kernel_size=1, strides=1)(tracer)
		tracer = tf.keras.layers.Softmax()(tracer)

		return tf.keras.Model(inputs=inputs, outputs=tracer)

	# Construct the U-Net predictor
	model = unet()

	# Write summary of the FULL model to file
	with stdout_to(open(PARAM['out_full_model_info'], 'w')):
		model.summary()

	optimizer = tf.keras.optimizers.Adam(amsgrad=True)
	# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
	loss = 'sparse_categorical_crossentropy'
	metrics = ['accuracy']

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	return model


def predict_segmentation_mask(model, image):
	# Prediction (handle image size & the batch dimension)
	label = model(tf.stack([PARAM['to_modeling_size'](image)]))[0]
	label = PARAM['to_original_size'](label)
	# Prediction proba for class '1'
	label = label[..., 1]
	# Flatten boring dimensions, just in case
	label = np.squeeze(label)
	return label


def train(model, ds_train: tf.data.Dataset, ds_valid: tf.data.Dataset):
	# Sanity check
	[(img_batch, lbl_batch)] = ds_valid.take(1)
	model.predict(img_batch)
	model.predict([img_batch])

	# try:
	# 	model.load_weights(tf.train.latest_checkpoint(os.path.dirname(PARAM['out_training_checkpoint'])))
	# except:
	# 	logger.exception("Could not load past model weights --")

	class ProgressCallback(tf.keras.callbacks.Callback):
		def __init__(self):
			self.logs = []

		def on_epoch_end(self, epoch, logs=None):
			self.logs.append({'epoch': epoch, **{k: float(v) for (k, v) in logs.items()}})
			with open(PARAM['out_training_history'], 'w') as fd:
				json.dump(self.logs, fd)
			plot_history.plot(self.logs)

			for (kind, ds) in dict(train=ds_train, valid=ds_valid).items():
				[(img_batch, lbl_batch)] = ds.take(1)
				label_pred = predict_segmentation_mask(model, img_batch[0])
				label_true = np.squeeze(PARAM['to_original_size'](lbl_batch[0]))
				imsave(PARAM['out_intraining_pred'].format(kind=kind), label_pred, cmap='gray', vmax=1)
				imsave(PARAM['out_intraining_true'].format(kind=kind), label_true, cmap='gray', vmax=1)

	# This is too naive:
	#checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=PARAM['out_training_checkpoint'], save_weights_only=True)

	callbacks = [ProgressCallback()]

	logger.debug("Training the model")

	history = model.fit(ds_train, validation_data=ds_valid, callbacks=callbacks, **PARAM['training'])

	return model


# ~~~ ENTRY ~~~ #

def main():
	logger.info("Loading the dataset")

	df = get_data()

	df_train: pd.DataFrame
	df_valid: pd.DataFrame

	# Training set (i.e. 'label' is given) and out-of-sample set (i.e. 'label' is n/a)
	(df_train, df_no_label) = map(df.groupby(df['label'].isna()).get_group, (False, True))

	logger.info("Training / validation data split")

	df_valid = df_train.sample(frac=PARAM['validation_frac'], replace=False, random_state=1)
	df_train = df_train.drop(df_valid.index)

	logger.info("Building the tensorflow dataset")

	ds_train = make_tf_dataset(df_train)
	ds_valid = make_tf_dataset(df_valid)

	logger.info("Building the predictor")

	model = make_model()

	logger.info("Training the predictor")

	model = train(model, ds_train, ds_valid)

	logger.info("Saving predictions to disk")

	for (idx, image) in pd.Series(df_no_label['image']).iteritems():
		label = predict_segmentation_mask(model, image)
		imsave(PARAM['out_predictions'].format(idx=idx), label, cmap='gray', vmax=1)


if __name__ == "__main__":
	main()
