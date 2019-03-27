from sklearn.model_selection import learning_curve
from sklearn.metrics import  roc_curve, auc, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pyaudio
import wave
import librosa as lr

def plot_learning_curve(est, X_train, y_train) :
    train_sizes, train_scores, test_scores = learning_curve(estimator=est, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=5,
                                                        n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(8,10))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')
    plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')
    plt.grid(b='on')
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.6, 1.0])
    plt.show()
    
def plot_roc_curve(est,X_test,y_test) :
    probas = est.predict_proba(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(8,8))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')       
    plt.plot([0,0,1],[0,1,1],'g:')     
    plt.xlim([-0.05,1.2])
    plt.ylim([-0.05,1.2])
    plt.ylabel('Taux de vrais positifs')
    plt.xlabel('Taux de faux positifs')
    plt.show
    
def extract_features(audiofile):
    X, sample_rate = lr.load(audiofile, res_type='kaiser_fast')
    
    # we extract mfcc feature from data
    features = np.mean(lr.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
    return(features)

def test_gender_classifier(classifier, X_test, y_test):
    """Test a classifier on the test_x data towards the test_y labels and print the classification report"""
    accuracy = classifier.score(X_test, y_test)

    print("Test accuracy : ", accuracy ,"\n")

    predictions = classifier.predict(X_test)

    male_as_female = np.sum(np.logical_and(y_test==0,predictions==1))
    female_as_male = np.sum(np.logical_and(y_test==1,predictions==0))

    print("{:d} males classified as females out of {:.0f}, {:.3f} %".format(male_as_female, np.sum(y_test==0), 100*male_as_female/np.sum(y_test==0)))
    print("{:d} females classified as males out of {:.0f}, {:.3f} %\n".format(female_as_male, np.sum(y_test==1), 100*female_as_male/np.sum(y_test==1)))

        
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, cmap="coolwarm", fmt='g')
    
    print(classification_report(y_test, predictions))

def test_age_classifier(classifier, X_test, y_test):
    """Test a classifier on the test_x data towards the test_y labels and print the classification report"""
    accuracy = classifier.score(X_test, y_test)

    print("Test accuracy : ", accuracy ,"\n")

    predictions = classifier.predict(X_test)

    younger_as_older = np.sum(np.logical_and(y_test==1,predictions==0))
    older_as_younger = np.sum(np.logical_and(y_test==0,predictions==1))

    print("{:d} young people classified as older out of {:.0f}, {:.3f} %".format(younger_as_older, np.sum(y_test==1), 100*younger_as_older/np.sum(y_test==1)))
    print("{:d} older people classified as younger out of {:.0f}, {:.3f} %\n".format(older_as_younger, np.sum(y_test==0), 100*older_as_younger/np.sum(y_test==0)))

        
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, cmap="coolwarm", fmt='g')
    
    print(classification_report(y_test, predictions))

def undersample(df, target_col, minority_class) :
    df_minority = df[df[target_col] == minority_class]
    df_majority = df.drop(df_minority.index)
    ratio=len(df_minority)/len(df_majority)
    df_majority = df_majority.sample(frac=ratio)
    df1 = pd.concat((df_majority,df_minority), axis=0)
    return df1.sample(frac=1)

def model_curves(model):
    accuracy = model.history['acc']
    val_accuracy = model.history['val_acc']
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def record(audio_filename):
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(audio_filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

print("utils.py correctly charged")