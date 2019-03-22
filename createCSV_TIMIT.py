import argparse
import timit_utils as tu
import os



# On définit l'argument nécéssaire pour le path
parser = argparse.ArgumentParser(description='Create two CSV files : TRAIN.csv, TEST.csv for every sound of TIMIT database.\nFormat is : soundName, personName, TST/TRN, soundPath, wrdPath, phnPath, txtPath, gender, region, age, recordDate, birthDate, height, race, education, prompt, promptStart, promptEnd'    )
parser.add_argument("-p", "--path", help="path to TIMIT, ex : C:/Users/xxxxx/TIMIT.\nDefault is current path.", type=str, default=str(os.getcwd()))
arg = parser.parse_args()



path_to_TIMIT = arg.path

path_to_SPKRINFO = path_to_TIMIT+"/DOC/SPKRINFO.TXT"
path_to_allfilelist = path_to_TIMIT+"/allfilelist.txt"




########################## CLASSES ##########################
class Person:
    """class for a person with every information in SPKRINFO.TXT"""
    def __init__(self, IDList, name, gender, region, use, birthDate, recordDate, age, height, race, education):
        self.IDList = IDList
        self.gender=gender
        self.name = name
        self.region = region
        self.use = use
        self.age=age
        self.birthDate = birthDate
        self.recordDate= recordDate
        self.height=height
        self.race = race
        self.education = education

    def getIDList(self):
        return self.IDList
    def getName(self):
        return self.name
    def getGender(self):
        return self.gender
    def getRegion(self):
        return self.region
    def getUse(self):
        return self.use
    def getBirthDate(self):
        return self.birthDate
    def getRecordDate(self):
        return self.recordDate
    def getAge(self):
        return self.age
    def getHeight(self):
        return self.height
    def getRace(self):
        return self.race
    def getEducation(self):
        return self.education


class Sound:
    """class for a single sound"""
    def __init__ (self, Pers, name, pathToIt):
        self.Pers = Pers
        self.name = name
        self.soundPath = "TIMIT/"+pathToIt+".wav"
        self.wrdPath = "TIMIT/"+pathToIt+".wrd"
        self.phnPath = "TIMIT/"+pathToIt+".phn"
        self.txtPath = "TIMIT/"+pathToIt+".txt"
        prompt, start, end = getPromptStartEnd(pathToIt+".txt")
        self.prompt = prompt
        self.start = start
        self.end = end
    def getPers(self):
        return self.Pers
    def getName(self):
        return self.name
    def getSoundPath(self):
        return self.soundPath
    def getWrdPath(self):
        return self.wrdPath
    def getPhnPath(self):
        return self.phnPath
    def getTxtPath(self):
        return self.txtPath
    def getPrompt(self):
        return self.prompt
    def getStart(self):
        return self.start
    def getEnd(self):
        return self.end


########################## FUNCTIONS ##########################

def retrieveBirthDate(name):
    """retrieve date of birthdate in SPKRINFO.TXT and return it
    path is path to SPKRINFO.TXT"""
    with open(path_to_SPKRINFO, "r") as read_file:
        for line in read_file:
            if(name) in line:
              birth_date = line[27:35]
    return birth_date


def retrieveRecordDate(name):
    """retrieve date of record in SPKRINFO.TXT and return it
    path is path to SPKRINFO.TXT"""
    with open(path_to_SPKRINFO, "r") as read_file:
        for line in read_file:
            if(name) in line:
              record_date=line[17:25]
    return record_date


def retrieveAge(name):
    """retrieve date of record and birthdate in SPKRINFO.TXT and return age
    path is path to SPKRINFO.TXT"""
    with open(path_to_SPKRINFO, "r") as read_file:
        for line in read_file:
            if(name) in line:
              record_date=line[17:25]
              birth_date = line[27:35]
    return getAge_fromdates(record_date,birth_date)


def retrieveRegion(name):
    """Retrieve the number region in SPKRINFO.TXT and return the name of the region
    path is path to SPKRINFO.TXT
    ; DR - Speaker dialect region number (1 - New England
;                                     2 - Northern
;                                     3 - North Midland
;                                     4 - South Midland
;                                     5 - Southern
;                                     6 - New York City
;                                     7 - Western
;                                     8 - Army Brat (moved around))"""
    with open(path_to_SPKRINFO, "r") as read_file:
        for line in read_file:
            if(name) in line:
                region_int = int(line[9])
        if (region_int == 1):
            return "New England"
        if (region_int == 2):
            return "Northern"
        if (region_int == 3):
            return "North Midland"
        if (region_int == 4):
            return "South Midland"
        if (region_int == 5):
            return "Southern"
        if (region_int == 6):
            return "New York City"
        if (region_int == 7):
            return "Western"
        if (region_int == 8):
            return "Army Brat (moved around)"


def retrieveUse(name):
    """retrieve for which dataset (TRAIN : TRN or TEST : TST) the person used their voice
    path is path to SPKRINFO.TXT"""
    with open(path_to_SPKRINFO, "r") as read_file:
        for line in read_file:
            if(name) in line:
                return line[12:15]


def retrieveHeight(name):
    """ retrieve height in ft and in from SPKRINFO.TXT, convert it to cm and return it (float round to 2)
    path is path to SPKINFO.TXT"""
    with open(path_to_SPKRINFO, "r") as read_file:
        for line in read_file:
            if(name) in line:
                ft = int(line[37])
                inc = line [39:41]
    if (inc[1]=="\""):
        inc= int(inc[0])
    else:
        inc=int(inc)
    inc = inc + 12*ft

    return round(inc*2.54,2)


def retrieveRace(name):
    """ retrieve race from SPKRINFO.TXT, and return it
    path is path to SPKINFO.TXT"""
    with open(path_to_SPKRINFO, "r") as read_file:
        for line in read_file:
            if(name) in line:
                return line[44:47]


def retrieveEducation(name):
    """ retrieve education from SPKRINFO.TXT, and return it
    path is path to SPKINFO.TXT"""
    with open(path_to_SPKRINFO, "r") as read_file:
        for line in read_file:
            if(name) in line:
                return line[49:52]


def getAge_fromdates(record_date, birth_date):
    """calculate age from  record and birthdate and return it"""
    split_rdate = record_date.split("/")
    split_bdate = birth_date.split("/")
    if (split_bdate[2] == "??"):
        return -1
    yr1=int(split_rdate[2])
    yr2=int(split_bdate[2])
    mth1=int(split_rdate[0])
    mth2=int(split_bdate[0])
    d1=int(split_rdate[1])
    d2=int(split_bdate[1])
    age = yr1-yr2
    if(mth1<mth2):
        age = age-1
    else:
        if(mth1==mth2 and d1<d2):
            age=age-1
    return age

def getAge(name):
    """get the record and birthdate , calculate the age and return it"""
    with open(path_to_SPKRINFO, "r") as read_file:
        for line in read_file:
            if(name) in line:
              record_date=line[17:25]
              birth_date = line[27:35]
    return getAge_fromdates(record_date,birth_date)

def getIDFileName(name, path_to_allfilelist):
    """get the List of all sound made by the person (called by name)
    path is path to allfilelist.txt """
    idList = []
    with open(path_to_allfilelist, "r") as read_file:
        for line in read_file:
            if(name.lower()) in line:
                if line[1]=="r":
                    idList.append(line[16:-1])
                else:
                    idList.append(line[15:-1])
    return idList

def getPathToIt(nameOfSound, path_to_allfilelist):
    """get path to the .WRD, .PHN, .WAV and .TXT
    with a song name and allfilelist.txt path as arguments
    used in Sound class instantiation"""
    with open(path_to_allfilelist, "r") as read_file:
        for line in read_file:
            if(nameOfSound) in line:
                return line[:-1].upper()

def getPromptStartEnd(path_to_txt):
    """get Prompt, and the Start/End of it and return them
    path is path to sound_name.txt
    used in Sound class instantiation"""
    with open(path_to_txt, "r") as read_file:
        for line in read_file:
            total_split = line.split(" ")
    return " ".join(total_split[2:]), total_split[0], total_split[1]

def writeLine(sound):
    """write a csv Line for every sound of TIMIT database with all possible parameters of Sound/Person """
    s=";"
    p=sound.getPers()
    line = sound.getName()+s+p.getName()+s+p.getUse()+s+sound.getSoundPath()+s+sound.getWrdPath()+s+sound.getPhnPath()+s+sound.getTxtPath()+s+p.getGender()+s+p.getRegion()
    line = line +s+str(p.getAge())+s+p.getRecordDate()+s+p.getBirthDate()+s+str(p.getHeight())+s+p.getRace()+s+p.getEducation()+s+sound.getPrompt()
    line = line +s+sound.getStart()+s+sound.getEnd()+"\n"
    return line


######################## MAIN ########################

corpus = tu.Corpus(path_to_TIMIT)
train = corpus.train
test= corpus.test


#On crée une liste avec toutes les personnes de Train database
people_train_list=[]
for i in range (len(train.people)):
    this_name = train.person_by_index(i).name
    this_person = Person(getIDFileName(this_name,path_to_allfilelist), this_name, train.person_by_index(i).gender,
                    retrieveRegion(this_name),retrieveUse(this_name), retrieveBirthDate(this_name), retrieveRecordDate(this_name),
                    retrieveAge(this_name), retrieveHeight(this_name), retrieveRace(this_name), retrieveEducation(this_name))
    people_train_list.append(this_person)


#On crée une liste avec toutes les personnes de Test database
people_test_list=[]
for i in range (len(test.people)):
    this_name = test.person_by_index(i).name
    this_person = Person(getIDFileName(this_name,path_to_allfilelist), this_name, test.person_by_index(i).gender,
                    retrieveRegion(this_name),retrieveUse(this_name), retrieveBirthDate(this_name), retrieveRecordDate(this_name),
                    retrieveAge(this_name), retrieveHeight(this_name), retrieveRace(this_name), retrieveEducation(this_name))
    people_test_list.append(this_person)


#On crée une liste avec tous les sons de chaque personne de Train database
sound_train_list=[]
for person in people_train_list:
        for sound in getIDFileName(person.name, path_to_allfilelist):
            this_sound = Sound(person, sound, getPathToIt(sound, path_to_allfilelist))
            sound_train_list.append(this_sound)


#On crée une liste avec tous les sons de chaque personne de Test database
sound_test_list=[]
for person in people_test_list:
        for sound in getIDFileName(person.name, path_to_allfilelist):
            this_sound = Sound(person, sound, getPathToIt(sound, path_to_allfilelist))
            sound_test_list.append(this_sound)


#On écrit toutes les lignes (une par son) pour chaque son de Train database dans TRAIN.csv
with open (path_to_TIMIT+"\\train.csv", "w+") as write_file:
    for sound in sound_train_list:
        write_file.write(writeLine(sound))


#On écrit toutes les lignes (une par son) pour chaque son de Test database dans TEST.csv
with open (path_to_TIMIT+"\\test.csv", "w+") as write_file:
    for sound in sound_test_list:
        write_file.write(writeLine(sound))
