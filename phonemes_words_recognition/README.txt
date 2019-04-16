This is a words and phonemes recogition AI based on TIMIT dataset.

You have to place your TIMIT directory here in phonemes_words_recognition\
install requirements:
 - pip install -r requirements.txt

you can use jupyter notebook to open "Speech-recognition-TIMIT.ipynb"

(It needs to load JSON files made with create_desc_json.py which needs the dataset to be arranged like LibriSpeech.
arrangeTIMIT.py will arrange it so that the JSON can be made.)
=> But the JSON are already present in  \JSON directory

all the other functions are used in the notebook along with explainations.

\images contains the necessary images for the notebook

\models contains already trained models and tests.

\CSV contains two csv : train and test, with all the data of the TIMIT dataset possible
