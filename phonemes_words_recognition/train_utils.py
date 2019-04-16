"""
Defines a functions for training a NN.
"""

from data_generator import AudioGenerator
import _pickle as pickle

from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Lambda, BatchNormalization)
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint	  
import os

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_loss(input_to_softmax):
	the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
	input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
	label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
	output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
	# output_length = BatchNormalization()(input_lengths)
	# CTC loss is implemented in a lambda layer
	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
		[input_to_softmax.output, the_labels, output_lengths, label_lengths])
	model = Model(
		inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths], 
		outputs=loss_out)
	return model

def train_model(input_to_softmax, 
				phn,
				pickle_path,
				save_model_path,
				train_json='JSON\\train_corpus',
				valid_json='JSON\\test_corpus',
				minibatch_size=10,
				mfcc_dim=13,
				optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
				epochs=20,
				verbose=1,
				sort_by_duration=False,
				max_duration=20.0):
	

	# create a class instance for obtaining batches of data
	audio_gen = AudioGenerator(minibatch_size=minibatch_size,  mfcc_dim=mfcc_dim, max_duration=max_duration,
		sort_by_duration=sort_by_duration)
	# add the training data to the generator
	audio_gen.load_train_data(train_json)
	audio_gen.load_test_data(valid_json)
	# calculate steps_per_epoch
	if phn:
		num_train_examples=len(audio_gen.train_phn_audio_paths)
		steps_per_epoch = num_train_examples//minibatch_size
	elif not phn:
		num_train_examples=len(audio_gen.train_wrd_audio_paths)
		steps_per_epoch = num_train_examples//minibatch_size
	# calculate validation_steps
	if phn:
		num_valid_samples = len(audio_gen.test_phn_audio_paths) 
		validation_steps = num_valid_samples//minibatch_size
	elif not phn:
		num_valid_samples = len(audio_gen.test_wrd_audio_paths) 
		validation_steps = num_valid_samples//minibatch_size
	
	# add CTC loss to the NN specified in input_to_softmax
	model = add_ctc_loss(input_to_softmax)

	# CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

	# make results/ directory, if necessary
	if not os.path.exists('models'):
		os.makedirs('models')

	# add checkpointer
	checkpointer = ModelCheckpoint(filepath='models/'+save_model_path, verbose=0)

	# train the model
	generator=audio_gen.next_train(phn)
	validation_data=audio_gen.next_test(phn)
	hist = model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch,
		epochs=epochs, validation_data=validation_data, validation_steps=validation_steps,
		callbacks=[checkpointer], verbose=verbose)

	# save model loss
	with open('models/'+pickle_path, 'wb') as f:
		pickle.dump(hist.history, f)