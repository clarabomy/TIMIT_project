import pandas as pd
from sys import argv


path_to_TIMIT = "TIMIT"
path_to_SPKRINFO = path_to_TIMIT+"/DOC/SPKRINFO.TXT"
possibilities= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ','\'']

def retrieveRegion(name):
	with open(path_to_SPKRINFO, "r") as read_file:
		for line in read_file:
			if(name) in line:
				return "DR"+line[9]

def retrieveGender(name):
	with open(path_to_SPKRINFO, "r") as read_file:
		for line in read_file:
			if(name) in line:
				return line[6]



def main(name, result):
	already_done = []
	if (result=="phn" or result=="phonemes"):
		result = "phn"
	elif(result=="wrd" or result=="words"):
		result="prompt"
	else:
		print("1st argument is wrong. Try 'wrd' or 'phn'.")
		return 0
	df= pd.read_csv(name+".csv", "|")
	currentPerson = None
	
	
	for index, row in df.iterrows():		
		prevPerson = row["personName"];
		if (currentPerson != prevPerson):
			currentPerson = prevPerson
			currentRegion = retrieveRegion(currentPerson)
			currentPath = "TIMIT/"+name.upper()+"/"+currentRegion+"/"+retrieveGender(currentPerson)+currentPerson+"/"
			if (result=="phn" or result=="phonemes"):
				currentFileName = currentRegion+"-"+retrieveGender(currentPerson)+currentPerson+".trans2.txt"
			else:
				currentFileName = currentRegion+"-"+retrieveGender(currentPerson)+currentPerson+".trans.txt"
			m = open(currentPath+currentFileName,'w+')
		if(row["soundName"] not in already_done):
			prompt= row[result].lower()
			for letter in prompt:
				if(letter not in possibilities):
					prompt = prompt.replace(letter,"")
			m.write(row["soundName"]+" "+prompt+"\n")
			already_done.append(row["soundName"])
		
		
if __name__ == '__main__':
	if len(argv)<3:
		argv.append("None")
	if argv[2].lower() == "test":
		main("test", argv[1].lower())
	elif argv[2].lower() == "train":
		main("train", argv[1].lower())
	else:
		main("train", argv[1].lower())
		main("test", argv[1].lower())
