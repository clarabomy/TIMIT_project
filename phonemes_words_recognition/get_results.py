import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text
from IPython.display import Audio
from glob import glob
import _pickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt

def get_predictions(index, partition, input_to_softmax, model_path, phn=False):
	""" Print a model's decoded predictions
	Params:
		index (int): The example you would like to visualize
		partition (str): One of 'train' or 'validation'
		input_to_softmax (Model): The acoustic model
		model_path (str): Path to saved acoustic model's weights
	"""
	# load the train and test data
	data_gen = AudioGenerator()
	data_gen.load_train_data()
	data_gen.load_test_data()
	
	# obtain the true transcription and the audio features 
	if partition == 'test':
		if phn:
			transcr = data_gen.test_phn_texts[index]
			audio_path = data_gen.test_phn_audio_paths[index]
		elif not phn:
			transcr = data_gen.test_wrd_texts[index]
			audio_path = data_gen.test_wrd_audio_paths[index]
		data_point = data_gen.normalize(data_gen.featurize(audio_path))
	elif partition == 'train':
		if phn:
			transcr = data_gen.train_phn_texts[index]
			audio_path = data_gen.train_phn_audio_paths[index]
		elif not phn:
			transcr = data_gen.train_wrd_texts[index]
			audio_path = data_gen.train_wrd_audio_paths[index]
		data_point = data_gen.normalize(data_gen.featurize(audio_path))
	else:
		raise Exception('Invalid partition!	 Must be "train" or "validation"')
		
	# obtain and decode the acoustic model's predictions
	input_to_softmax.load_weights(model_path)
	prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
	output_length = [input_to_softmax.output_length(data_point.shape[0])] 
	pred_ints = (K.eval(K.ctc_decode(
				prediction, output_length)[0][0])+1).flatten().tolist()
	
	# play the audio file, and display the true and predicted transcriptions
	if not phn:
		print('-'*80)
		Audio(audio_path)
		print('True transcription:\n' + '\n' + transcr)
		print('-'*80)
		print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints, phn)))
		print('-'*80)
	else:
		print('-'*80)
		Audio(audio_path)
		print('True transcription:\n' + '\n' + transcr)
		print('-'*80)
		print('Predicted transcription:\n' + '\n')
		split_true = transcr.split(" ")
		split_pred= (''.join(int_sequence_to_text(pred_ints, phn))).split(" ")
		print("\033[1;32m"+split_pred[0] + " ", end='')
		for i in range (1,len(split_true)-1):
			if split_true[i-1]==split_pred[i] or split_true[i]==split_pred[i] or split_true[i+1]==split_pred[i]:
				print("\033[1;32m"+split_pred[i]+" ", end='')
			else:
				print("\033[1;31m"+split_pred[i]+" ", end='')
		print(split_pred[len(split_true)-1] + " ", end='')
	split_pred= (''.join(int_sequence_to_text(pred_ints, phn))).split(" ")
	split_true = transcr.split(" ")
	displayAccuracy(split_true, split_pred, phn)

		
	
def compare_predictions(index, partition, inputs_to_softmax=[], model_paths=[], phn=False):
	""" Print a model's decoded predictions
	Params:
		index (int): The example you would like to visualize
		partition (str): One of 'train' or 'validation'
		input_to_softmax (Model): The acoustic model
		model_path (str): Path to saved acoustic model's weights
	"""
	# load the train and test data
	data_gen = AudioGenerator()
	data_gen.load_train_data()
	data_gen.load_test_data()
	
	# obtain the true transcription and the audio features 
	if partition == 'test':
		if phn:
			transcr = data_gen.test_phn_texts[index]
			audio_path = data_gen.test_phn_audio_paths[index]
		elif not phn:
			transcr = data_gen.test_wrd_texts[index]
			audio_path = data_gen.test_wrd_audio_paths[index]
		data_point = data_gen.normalize(data_gen.featurize(audio_path))
	elif partition == 'train':
		if phn:
			transcr = data_gen.train_phn_texts[index]
			audio_path = data_gen.train_phn_audio_paths[index]
		elif not phn:
			transcr = data_gen.train_wrd_texts[index]
			audio_path = data_gen.train_wrd_audio_paths[index]
		data_point = data_gen.normalize(data_gen.featurize(audio_path))
	else:
		raise Exception('Invalid partition!	 Must be "train" or "validation"')
		
	# obtain and decode the acoustic model's predictions
	pred_ints=[]
	for model_path, input_to_softmax in zip(model_paths, inputs_to_softmax):
		input_to_softmax.load_weights(model_path)
		prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
		output_length = [input_to_softmax.output_length(data_point.shape[0])] 
		pred_int = (K.eval(K.ctc_decode(
					prediction, output_length)[0][0])+1).flatten().tolist()
		pred_ints.append(pred_int)
	
	# play the audio file, and display the true and predicted transcriptions
	print('-'*80)
	Audio(audio_path)
	print('True transcription:\n' + '\n' + transcr)
	print('-'*80)
	i=0
	for pred_in in pred_ints:
		i=i+1
		print('Predicted transcription number', i,':\n' + '\n' + ''.join(int_sequence_to_text(pred_in, phn)))
		print('-'*80)

		
def compare_results():
	sns.set_style(style='white')

	# obtain the paths for the saved model history
	all_pickles = sorted(glob("models\\final\\*.pickle"))
	# extract the name of each model
	model_names = [item[8:-7] for item in all_pickles]
	# extract the loss history for each model
	valid_loss = [pickle.load( open( i, "rb" ) )['val_loss'] for i in all_pickles]
	train_loss = [pickle.load( open( i, "rb" ) )['loss'] for i in all_pickles]
	# save the number of epochs used to train each model
	num_epochs = [len(valid_loss[i]) for i in range(len(valid_loss))]

	fig = plt.figure(figsize=(16,5))

	# plot the training loss vs. epoch for each model
	ax1 = fig.add_subplot(121)
	for i in range(len(all_pickles)):
		ax1.plot(np.linspace(1, num_epochs[i], num_epochs[i]), 
				train_loss[i], label=model_names[i])
	# clean up the plot
	ax1.legend()  
	ax1.set_xlim([1, max(num_epochs)])
	plt.xlabel('Epoch')
	plt.ylabel('Training Loss (CTC loss)')

	# plot the validation loss vs. epoch for each model
	ax2 = fig.add_subplot(122)
	for i in range(len(all_pickles)):
		ax2.plot(np.linspace(1, num_epochs[i], num_epochs[i]), 
				valid_loss[i], label=model_names[i])
	# clean up the plot
	ax2.legend()  
	ax2.set_xlim([1, max(num_epochs)])
	plt.xlabel('Epoch')
	plt.ylabel('Validation Loss (CTC loss)')
	plt.show()
	
def displayAccuracy(split_true, split_pred, phn=False):
	l1= split_true
	l2= split_pred

	if phn:
		result = 0
		for i in range(len(l2)):
			check = [l1[x] for x in range(i-3,i+4) if x>=0 and x<len(l1)]
			if l2[i] in check:
				result += 1
		result = result/len(l1) * 100
	else:
		result = 0
		for i in range(len(l2)):
			check = [l1[x] for x in range(i-3,i+4) if x>=0 and x<len(l1)]
			if l2[i] in check:
				result += 10
			else:
				results=[]
				for word in check:
					results.append(avrg(word, l2[i]))
				result += max(results)
		result = result /(len(l2)*10)
		result = result * 100
					
				
	
	
	print ("\n\n\033[1;30m","►" * 40,"ACCURACY  =",round(result,2),"%","◄"*40)
	return 0
	
	
def checkScore(w1, w2):
	total = len(w2)
	for i in range (len(w2)):
		ch=[w1[x] for x in range(i-2,i+3) if x>=0 and x<len(w1)]
		if w2[i] in ch:
			total -= 1
	total = total * 2
	total = total / len(w2) * 10
	return int(10 - total)

def avrg(w1 , w2):
	n = min((checkScore(w1,w2),(checkScore(w2,w1))))
	n = n if n>=0 else 0
	return n