"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""

import json
import numpy as np
import random
from python_speech_features import mfcc
import librosa
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import calc_feat_dim, spectrogram_from_file, text_to_int_sequence
from utils import conv_output_length

RNG_SEED = 123

class AudioGenerator():
	def __init__(self, mfcc_dim=13, minibatch_size=20, max_duration=10.0, 
		sort_by_duration=False):
		"""
		Params:
			desc_file (str, optional): Path to a JSON-line file that contains
				labels and paths to the audio files. If this is None, then
				load metadata right away
		"""

		#self.feat_dim = calc_feat_dim(window, max_freq)
		self.mfcc_dim = mfcc_dim
		self.feats_mean = np.zeros((self.mfcc_dim,))
		self.feats_std = np.ones((self.mfcc_dim,))
		self.rng = random.Random(RNG_SEED)
		#self.step = step
		#self.window = window
		#self.max_freq = max_freq
		self.cur_train_wrd_index = 0
		self.cur_train_phn_index = 0
		self.cur_test_wrd_index = 0
		self.cur_test_phn_index = 0
		self.max_duration=max_duration
		self.minibatch_size = minibatch_size
		#self.spectrogram = spectrogram
		self.sort_by_duration = sort_by_duration

	def get_batch(self, partition, phn= False):
		""" Obtain a batch of train or test data
		"""
		if partition == 'train':
			if not phn:
				audio_paths = self.train_wrd_audio_paths
				cur_index = self.cur_train_wrd_index
				texts = self.train_wrd_texts
			elif phn:
				audio_paths = self.train_phn_audio_paths
				cur_index = self.cur_train_phn_index
				texts = self.train_phn_texts
			else:
				raise Exception("Invalid type. "
				"Must be phn/wrd")
				
		#elif partition == 'valid':
			#audio_paths = self.valid_audio_paths
			#cur_index = self.cur_valid_index
			#texts = self.valid_texts
		elif partition == 'test':
			if not phn:
				audio_paths = self.test_wrd_audio_paths
				cur_index = self.cur_test_wrd_index
				texts = self.test_wrd_texts
			elif phn:
				audio_paths = self.test_phn_audio_paths
				cur_index = self.cur_test_phn_index
				texts = self.test_phn_texts
			else:
				raise Exception("Invalid type. "
				"Must be phn/wrd")
		else:
			raise Exception("Invalid partition. "
				"Must be train/test")

		features = [self.normalize(self.featurize(a)) for a in 
			audio_paths[cur_index:cur_index+self.minibatch_size]]

		# calculate necessary sizes
		max_length = max([features[i].shape[0] 
			for i in range(0, self.minibatch_size)])
		max_string_length = max([len(texts[cur_index+i]) 
			for i in range(0, self.minibatch_size)])
		
		# initialize the arrays
		X_data = np.zeros([self.minibatch_size, max_length, 
			self.mfcc_dim])
		labels = np.ones([self.minibatch_size, max_string_length]) * 28
		input_length = np.zeros([self.minibatch_size, 1])
		label_length = np.zeros([self.minibatch_size, 1])
		
		for i in range(0, self.minibatch_size):
			# calculate X_data & input_length
			feat = features[i]
			input_length[i] = feat.shape[0]
			X_data[i, :feat.shape[0], :] = feat

			# calculate labels & label_length
			label = np.array(text_to_int_sequence(texts[cur_index+i], phn)) 
			labels[i, :len(label)] = label
			label_length[i] = len(label)
 
		# return the arrays
		outputs = {'ctc': np.zeros([self.minibatch_size])}
		inputs = {'the_input': X_data, 
				  'the_labels': labels, 
				  'input_length': input_length, 
				  'label_length': label_length 
				 }
		return (inputs, outputs)

	def shuffle_data_by_partition(self, partition, phn = False):
		""" Shuffle the training or test data
		"""
		if partition == 'train':
			if not phn:
				self.train_wrd_audio_paths, self.train_wrd_durations, self.train_wrd_texts = shuffle_data(
					self.train_wrd_audio_paths, self.train_wrd_durations, self.train_wrd_texts)
			elif phn:
				self.train_phn_audio_paths, self.train_phn_durations, self.train_phn_texts = shuffle_data(
					self.train_phn_audio_paths, self.train_phn_durations, self.train_phn_texts)
			else:
				raise Exception("Invalid type. "
				"Must be phn/wrd")
		elif partition == 'test':
			if not phn:
				self.test_wrd_audio_paths, self.test_wrd_durations, self.test_wrd_texts = shuffle_data(
					self.test_wrd_audio_paths, self.test_wrd_durations, self.test_wrd_texts)
			elif phn:
				self.test_phn_audio_paths, self.test_phn_durations, self.test_phn_texts = shuffle_data(
					self.test_phn_audio_paths, self.test_phn_durations, self.test_phn_texts)
			else:
				raise Exception("Invalid type. "
				"Must be phn/wrd")
		else:
			raise Exception("Invalid partition. "
				"Must be train/test")

	def sort_data_by_duration(self, partition, phn=False):
		""" Sort the training or test sets by (increasing) duration
		"""
		if partition == 'train':
			if not phn:
				self.train_wrd_audio_paths, self.train_wrd_durations, self.train_wrd_texts = sort_data(
					self.train_wrd_audio_paths, self.train_wrd_durations, self.train_wrd_texts)
			elif phn:
				self.train_phn_audio_paths, self.train_phn_durations, self.train_phn_texts = sort_data(
					self.train_phn_audio_paths, self.train_phn_durations, self.train_phn_texts)
			else:
				raise Exception("Invalid type. "
				"Must be phn/wrd")
		elif partition == 'test':
			if not phn:
				self.test_wrd_audio_paths, self.test_wrd_durations, self.test_wrd_texts = shuffle_data(
					self.test_wrd_audio_paths, self.test_wrd_durations, self.test_wrd_texts)
			elif phn:
				self.test_phn_audio_paths, self.test_phn_durations, self.test_phn_texts = shuffle_data(
					self.test_phn_audio_paths, self.test_phn_durations, self.test_phn_texts)
			else:
				raise Exception("Invalid type. "
				"Must be phn/wrd")
		else:
			raise Exception("Invalid partition. "
				"Must be train/test")

	def next_train(self, phn=False):
		""" Obtain a batch of training data
		"""
		while True:
			ret = self.get_batch('train', phn)
			if not phn:
				self.cur_train_wrd_index += self.minibatch_size
				if self.cur_train_wrd_index >= len(self.train_wrd_texts) - self.minibatch_size:
					self.cur_train_wrd_index = 0
					self.shuffle_data_by_partition('train', phn)
			elif phn:
				self.cur_train_phn_index += self.minibatch_size
				if self.cur_train_phn_index >= len(self.train_phn_texts) - self.minibatch_size:
					self.cur_train_phn_index = 0
					self.shuffle_data_by_partition('train', phn)
			else:
				raise Exception("Invalid type. "
				"Must be phn/wrd")
			yield ret	 

	"""def next_valid(self, phn=False):
		 Obtain a batch of validation data
		
		while True:
			ret = self.get_batch('valid', phn)
			self.cur_valid_index += self.minibatch_size
			if self.cur_valid_index >= len(self.valid_texts) - self.minibatch_size:
				self.cur_valid_index = 0
				self.shuffle_data_by_partition('valid')
			yield ret"""

	def next_test(self, phn=False):
		""" Obtain a batch of test data
		"""
		while True:
			ret = self.get_batch('test', phn)
			if not phn:
				self.cur_test_wrd_index += self.minibatch_size
				if self.cur_test_wrd_index >= len(self.test_wrd_texts) - self.minibatch_size:
					self.cur_test_wrd_index = 0
					self.shuffle_data_by_partition('test', phn)
			elif phn:
				self.cur_test_phn_index += self.minibatch_size
				if self.cur_test_phn_index >= len(self.test_phn_texts) - self.minibatch_size:
					self.cur_test_phn_index = 0
					self.shuffle_data_by_partition('test', phn)
			else:
				raise Exception("Invalid type. "
				"Must be phn/wrd")
			yield ret

	def load_train_data(self, desc_file='JSON\\train_corpus'):
		desc_file_wrd=desc_file+'_wrd.json'
		desc_file_phn=desc_file+'_phn.json'
		self.load_metadata_from_desc_file(desc_file_phn, 'train', 'phn')
		self.load_metadata_from_desc_file(desc_file_wrd, 'train', 'wrd')
		self.fit_train()
		if self.sort_by_duration:
			self.sort_data_by_duration('train')

	#def load_validation_data(self, desc_file='valid_corpus.json'):
		#self.load_metadata_from_desc_file(desc_file, 'validation')
		#if self.sort_by_duration:
			#self.sort_data_by_duration('valid')

	def load_test_data(self, desc_file='JSON\\test_corpus'):
		desc_file_wrd=desc_file+'_wrd.json'
		desc_file_phn=desc_file+'_phn.json'
		self.load_metadata_from_desc_file(desc_file_phn, 'test', 'phn')
		self.load_metadata_from_desc_file(desc_file_wrd, 'test', 'wrd')
		
		
	def load_metadata_from_desc_file(self, desc_file, partition, type):
		""" Read metadata from a JSON-line file
			(possibly takes long, depending on the filesize)
		Params:
			desc_file (str):  Path to a JSON-line file that contains labels and
				paths to the audio files
			partition (str): One of 'train' or 'test'
		"""
		audio_paths, durations, texts = [], [], []
		with open(desc_file) as json_line_file:
			for line_num, json_line in enumerate(json_line_file):
				try:
					spec = json.loads(json_line)
					if float(spec['duration']) > self.max_duration:
						continue
					audio_paths.append(spec['key'])
					durations.append(float(spec['duration']))
					texts.append(spec['text'])
				except Exception as e:
					# Change to (KeyError, ValueError) or
					# (KeyError,json.decoder.JSONDecodeError), depending on
					# json module version
					print('Error reading line #{}: {}'
								.format(line_num, json_line))
		if partition == 'train':
			if type == 'phn':
				self.train_phn_audio_paths = audio_paths
				self.train_phn_durations = durations
				self.train_phn_texts = texts
			if type == 'wrd':
				self.train_wrd_audio_paths = audio_paths
				self.train_wrd_durations = durations
				self.train_wrd_texts = texts
		#elif partition == 'validation':
			#self.valid_audio_paths = audio_paths
			#self.valid_durations = durations
			#self.valid_texts = texts
		elif partition == 'test':
			if type == 'phn':
				self.test_phn_audio_paths = audio_paths
				self.test_phn_durations = durations
				self.test_phn_texts = texts
			if type == 'wrd':
				self.test_wrd_audio_paths = audio_paths
				self.test_wrd_durations = durations
				self.test_wrd_texts = texts
		else:
			raise Exception("Invalid partition to load metadata. "
			 "Must be train/test")
			
	def fit_train(self, k_samples=100):
		""" Estimate the mean and std of the features from the training set
		Params:
			k_samples (int): Use this number of samples for estimation
		"""
		k_samples_wrd = min(k_samples, len(self.train_wrd_audio_paths))
		samples_wrd = self.rng.sample(self.train_wrd_audio_paths, k_samples)
		k_samples_phn = min(k_samples, len(self.train_phn_audio_paths))
		samples_phn = self.rng.sample(self.train_phn_audio_paths, k_samples)
		feats_wrd = [self.featurize(s) for s in samples_wrd]
		feats_wrd = np.vstack(feats_wrd)
		feats_phn = [self.featurize(s) for s in samples_phn]
		feats_phn = np.vstack(feats_phn)
		self.feats_mean_wrd = np.mean(feats_wrd, axis=0)
		self.feats_std_wrd = np.std(feats_wrd, axis=0)
		self.feats_mean_phn = np.mean(feats_phn, axis=0)
		self.feats_std_phn = np.std(feats_phn, axis=0)
		
	def featurize(self, audio_clip):
		""" For a given audio clip, calculate the corresponding feature
		Params:
			audio_clip (str): Path to the audio clip
		"""
		(rate, sig) = wav.read(audio_clip)
		return mfcc(sig, rate, numcep=self.mfcc_dim)

	def normalize(self, feature, eps=1e-14):
		""" Center a feature using the mean and std
		Params:
			feature (numpy.ndarray): Feature to normalize
		"""
		return (feature - self.feats_mean) / (self.feats_std + eps)

def shuffle_data(audio_paths, durations, texts):
	""" Shuffle the data (called after making a complete pass through 
		training or test data during the training process)
	Params:
		audio_paths (list): Paths to audio clips
		durations (list): Durations of utterances for each audio clip
		texts (list): Sentences uttered in each audio clip
	"""
	p = np.random.permutation(len(audio_paths))
	audio_paths = [audio_paths[i] for i in p] 
	durations = [durations[i] for i in p] 
	texts = [texts[i] for i in p]
	return audio_paths, durations, texts

def sort_data(audio_paths, durations, texts):
	""" Sort the data by duration 
	Params:
		audio_paths (list): Paths to audio clips
		durations (list): Durations of utterances for each audio clip
		texts (list): Sentences uttered in each audio clip
	"""
	p = np.argsort(durations).tolist()
	audio_paths = [audio_paths[i] for i in p]
	durations = [durations[i] for i in p] 
	texts = [texts[i] for i in p]
	return audio_paths, durations, texts

def vis_train_features(index=0):
	""" Visualizing the data point in the training set at the supplied index
	"""
	# obtain spectrogram
	#audio_gen = AudioGenerator(spectrogram=True)
	#audio_gen.load_train_data()
	#vis_spectrogram_feature = audio_gen.normalize(audio_gen.featurize(vis_audio_path))
	# obtain mfcc
	audio_gen = AudioGenerator()
	audio_gen.load_train_data()
	vis_audio_path_wrd = audio_gen.train_wrd_audio_paths[index]
	vis_audio_path_phn = audio_gen.train_phn_audio_paths[index]
	vis_mfcc_feature_wrd = audio_gen.normalize(audio_gen.featurize(vis_audio_path_wrd))
	vis_mfcc_feature_phn = audio_gen.normalize(audio_gen.featurize(vis_audio_path_phn))
	# obtain text label
	vis_text_wrd = audio_gen.train_wrd_texts[index]
	vis_text_phn = audio_gen.train_phn_texts[index]
	# obtain raw audio
	vis_raw_audio_wrd, _ = librosa.load(vis_audio_path_wrd)
	vis_raw_audio_phn, _ = librosa.load(vis_audio_path_phn)
	# print total number of training examples
	print('There are %d total training examples for words.' % len(audio_gen.train_wrd_audio_paths))
	print('There are %d total training examples for phonemes.' % len(audio_gen.train_phn_audio_paths))
	# return labels for plotting
	return vis_text_wrd, vis_raw_audio_wrd, vis_mfcc_feature_wrd, vis_audio_path_wrd, vis_text_phn, vis_raw_audio_phn, vis_mfcc_feature_phn, vis_audio_path_phn


def plot_raw_audio(vis_raw_audio):
	# plot the raw audio signal
	fig = plt.figure(figsize=(12,3))
	ax = fig.add_subplot(111)
	steps = len(vis_raw_audio)
	ax.plot(np.linspace(1, steps, steps), vis_raw_audio)
	plt.title('Audio Signal')
	plt.xlabel('Time')
	plt.ylabel('Amplitude')
	plt.show()

def plot_mfcc_feature(vis_mfcc_feature):
	# plot the MFCC feature
	fig = plt.figure(figsize=(12,5))
	ax = fig.add_subplot(111)
	im = ax.imshow(vis_mfcc_feature, cmap=plt.cm.jet, aspect='auto')
	plt.title('Normalized MFCC')
	plt.ylabel('Time')
	plt.xlabel('MFCC Coefficient')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax.set_xticks(np.arange(0, 13, 2), minor=False);
	plt.show()

"""def plot_spectrogram_feature(vis_spectrogram_feature):
	# plot the normalized spectrogram
	fig = plt.figure(figsize=(12,5))
	ax = fig.add_subplot(111)
	im = ax.imshow(vis_spectrogram_feature, cmap=plt.cm.jet, aspect='auto')
	plt.title('Normalized Spectrogram')
	plt.ylabel('Time')
	plt.xlabel('Frequency')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	plt.show()"""

