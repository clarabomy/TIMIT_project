import argparse
import timit_utils as tu
import os
import pandas as pd


def get_args():
    """Function to get arguments from console"""
    desc = 'Create new filtered datasets from train.csv and test.csv of TIMIT database according to your own parameters (you can use createCSV_TIMIT.py to create them). Every parameter is optional.'
    desc = desc + ' \nIf you use several phonemes it uses the &(AND) operator, for several words it uses the |(OR) operator.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-n", "--name", help="Name of the individual, ex: \"ABC0\"", type=str)
    parser.add_argument("-a", "--age", help="Age range of the individual, ex: \"0 40\" (means \"age<40\") OR one age used to split into 2 results.", type=str)
    parser.add_argument("-g", "--gender", help="Gender of the individual: M|F ", type=str)
    parser.add_argument("-r", "--region", help="Region of the individual, ex: \"DR3\" or \"North Midland\"", type=str)
    parser.add_argument("-e", "--edu", "--education", help="Education level of the individual, ex: \"PHD\"", type=str)
    parser.add_argument("-ra", "--race", help="Race of the individual, ex: \"WHT\" or \"white\"", type=str)
    parser.add_argument("-he", "--height", help="Height range of the individual (in cm), ex: \"160 180\"", type=str)
    parser.add_argument("-s", "--sound", help="Name of the sound, ex: \"sa1\"", type=str)
    parser.add_argument("-d", "--duration", help="Duration range of the record (in tenth milliseconds), ex: \"10000 50000\" (min:14644, max:124621)", type=str)
    parser.add_argument("-w", "--words", nargs = '*', help="Words used for the filter, ex: \"water\" ", type=str)
    parser.add_argument("-p", "--phonemes", nargs = '*', help="Phonemes used for the filter, ex: \"sh\"", type=str)

    #nargs = '*' : the last argument take zero or more parameter
    args = parser.parse_args()

    #name verif
    if (args.name!=None):
        args.name=args.name.upper()

    #age verif
    age_min = None
    age_max = None
    if (args.age!=None):
        age=args.age.split(" ")
        if(len(age)>2 or len(age)<0):
            print("Error, age range : ",age," not recognized. Ignored.")
        else:
            if(len(age)==2):
                try:
                    age_min = int(age[0])
                    age_max = int(age[1])
                    if (age_max<age_min):
                        age_max, age_min = age_min, age_max
                except ValueError:
                    print("Error, age range : ",age," not recognized. Ignored.")
            else:
                try:
                    age_min = False;
                    age_max = int(age[0]);
                except ValueError:
                    print("Error, age range : ",age," not recognized. Ignored.")


    #gender verif
    gender = None
    if (args.gender!=None):
        args.gender=args.gender.lower()
        if (args.gender in ["w","woman","f","female"]):
           gender="F"
        else:
            if (args.gender in ["m","man","male"]):
             gender="M"
            else:
                print("Error, gender : ",gender," not recognized. Ignored.")


    #region verif
    region = None
    if (args.region!=None):
        regions = ["new england", "northern", "north midland", "south midland", "southern", "new york city", "western", "army brat (moved around)"]
        if len(args.region)==3:
            try:
                index=int(args.region[2])-1
                region = regions[index]
            except ValueError:
                print("Error, region: ",region," not recognized. Ignored.")
                region = None;
        else:
            region = args.region
        region=region.lower()
        if region=="army brat":
            region = "army brat (moved around)"
        if region not in regions:
            print("Error, region: ",region," not recognized. Ignored.")
            region = None;
        else :
            region = region.title()
            if(region[0]=="A"):
                region = "Army Brat (moved around)"


    #education verif
    edu = None
    if (args.edu!=None):
        edus = ["HS","AS","BS","MS","PHD","??"]
        edus2 = ["HS ","AS ","BS ","MS ","PHD","?? "]
        edu = args.edu.upper()
        if edu not in edus+edus2:
            print("Error, education: ",edu," not recognized. Ignored.")
            edu = None;
        else:
            if edu in edus:
                edu = edus2[edus.index(edu)]
                

    #race verif
    race = None
    if(args.race !=None):
        races = ["WHT", "AMR", "BLK", "SPN", "ORN", "???"]
        races2 = ["WHITE", "AMERICAN INDIAN", "BLACK", "SPANISH", "ORIENTAL", "UNKNOWN"]
        race=args.race.upper()
        if (race in races2):
            race = races[races2.index(race)]
        if race not in races:
            print("Error, race: ",race," not recognized. Ignored.")
            race = None;

    #height verif
    height_min = None
    height_max = None
    if (args.height!=None):
        height=args.height.split(" ")
        if(len(height)!=2):
            print("Error, height range : ",height," not recognized. Ignored.")
        else:
            try:
                height_min= int(height[0])
                height_max = int(height[1])
                if (height_max<height_min):
                    height_max, height_min = height_min, height_max
            except ValueError:
                height_min = None
                height_max = None
                print("Error, height range : ",height," not recognized. Ignored.")

    #sound verif
    if (args.sound!=None):
        args.sound.lower()

    #duration verif
    duration_min = None
    duration_max = None
    if (args.duration!=None):
        duration=args.duration.split(" ")
        if(len(duration)!=2):
            print("Error, duration range : ",duration," not recognized. Ignored.")
        else:
            try:
                duration_min= int(duration[0])
                duration_max = int(duration[1])
                if (duration_max<duration_min):
                    duration_max, duration_min = duration_min, duration_max
                if (duration_max<14644):
                    print ("Error duration range: ", duration, " is too short. Ignored. Try higher numbers!")
                    duration_min = None
                    duration_max = None
            except ValueError:
                print("Error, duration range : ",duration," not recognized. Ignored.")


    #words verif
    if(args.words== None):
        args.words=[]
        

    #phonemes verif
    if(args.phonemes== None):
        args.phonemes=[]

    return {"personName": args.name, "gender": gender, "region": region, "education": edu, "race": race, "soundName": args.sound},args.words, args.phonemes, age_min, age_max, height_min, height_max, duration_min, duration_max

def createFilters(df, to_be_checked, words=None, phonemes=None, age_min=None, age_max=None, height_min=None, height_max=None, duration_min=None, duration_max=None):
    #bool var for splitting into 4 dataset if only one age
    split = False;
    #create for each valid string parameter a bool condition filter
    #add this filter to all filter
    all_filters = True;
    for key, value in to_be_checked.items():
        this_filter = df[key]==value
        all_filters = all_filters & this_filter

    #create for words list a bool condition filter
    #add this filter to all filter   
    if(words):
        this_filter= False
        for word in words:
            this_filter= this_filter | df["prompt"].str.contains(word)
        all_filters = all_filters & this_filter

    #create for phn list a bool condition filter
    #add this filter to all filter     
    if(phonemes):
        this_filter= True
        for phn in phonemes:
            this_filter= this_filter & df["phn"].str.contains(phn)
            all_filters = all_filters & this_filter

            
    if(duration_min and duration_max):
        this_filter= df["end"].between(duration_min, duration_max)
        all_filters = all_filters & this_filter


    if(height_min and height_max):
        this_filter= df["height"].between(height_min, height_max)
        all_filters = all_filters & this_filter

        
    if(age_min and age_max):
        this_filter= df["age"].between(age_min, age_max)
        all_filters = all_filters & this_filter
    else:
        if (age_max):
            #if age_max is not None, but age_min is, it means there is only one age. So we must split into 2 datasets one above one under
            split = True

    return all_filters, split

def dfFiltered_tocsv(csv, df, title, all_filters, split, age_max):

    if(not split):
        #create the filtered csv if not empty
        try:
            dff=df[all_filters]
            dff = dff.loc[:, ~df.columns.str.contains('Unnamed')]
            if (dff.empty):
                print("No matching result for",csv+".csv. Try something else!")
            else:
                dff.to_csv(title+csv+".csv", sep="|")
        except KeyError:
            print("No matching result for",csv+".csv. Try something else!")
    else:
        #first part under the age selected
        this_filter= df["age"]<=age_max
        all_filters2= all_filters & this_filter
        try:
            dff=df[all_filters2]
            dff = dff.loc[:, ~df.columns.str.contains('Unnamed')]
            if (dff.empty):
                print("No matching result for",csv+".csv under the age selected.")
            else:
                dff.to_csv(title+csv+"_underAge_"+str(age_max)+".csv", sep="|")
        except KeyError:
            print("No matching result for",csv+".csv under the age selected.")
        #second part over the age selected
        this_filter= df["age"]>age_max
        all_filters2= all_filters & this_filter
        try:
            dff=df[all_filters2]
            dff = dff.loc[:, ~df.columns.str.contains('Unnamed')]
            if (dff.empty):
                print("No matching result for",csv+".csv over the age selected.")
            else:
                dff.to_csv(title+csv+"_overAge_"+str(age_max)+".csv", sep="|")
        except KeyError:
            print("No matching result for",csv+".csv over the age selected.")

def addToTitle(title, data, name):
    if (data!= None):
        if isinstance(data, list):
            if(data):
                title = title+name+"_"
        else:
            if isinstance(data, tuple):
                if(data[0]!=None and data[0]!=False and data[1]!=None):
                    title=title+name+"-"+str(data[0])+"-"+str(data[1])+"_"
    return title
    
def main():
    print()
    # get the params entered
    params, words, phonemes, age_min, age_max, height_min, height_max, duration_min, duration_max=get_args()

    to_be_checked={}
    title=""
    # get each param which arent None(each relevant param) in to_be_checked
    # and create the title for the future filtered files
    for key, value in params.items():
        if value != None:
            to_be_checked[key]=value;
            title = title + value+"_";
    for key, value in {"words":words, "phonemes":phonemes, "age":(age_min, age_max), "height":(height_min, height_max), "duration":(duration_min, duration_max)}.items():
        title = addToTitle(title, value, key)



    if(not age_min and age_max):
        print ("you entered only one age, therefore you gonna have the csv split into 2 parts : one over this age and one under this age")

    #open the full train csv and create all filters according to it
    df= pd.read_csv("train.csv", "|")
    all_filters, split = createFilters(df, to_be_checked, words, phonemes, age_min, age_max, height_min, height_max, duration_min, duration_max)
    dfFiltered_tocsv("train", df, title, all_filters, split, age_max)
    
            
    #open the full test csv and create all filters according to it
    df= pd.read_csv("test.csv", "|")
    all_filters, split = createFilters(df, to_be_checked, words, phonemes, age_min, age_max, height_min, height_max, duration_min, duration_max)
    dfFiltered_tocsv("test", df, title, all_filters, split, age_max)

    #add words and phonemes to the dictionary so...
    if(words):
        to_be_checked["words"]=" | ".join(words)

    if(phonemes):
        to_be_checked["phonemes"]=" | ".join(phonemes)
        
    if (age_max and age_min):
        to_be_checked["age"]="between "+str(age_min)+" and "+str(age_max)
    else:
        if(age_max):
            to_be_checked["age"]="split into two from age: "+ str(age_max)

    if(height_max and height_min):
        to_be_checked["height"]="between "+str(height_min)+" and "+str(height_max)

    if(duration_max and duration_min):
        to_be_checked["duration"]="between "+str(duration_min)+" and "+str(duration_max)



    #... we can print in console the the dictionary (relevant valid params) that were used for used to know
    print("Valid left parameters are : ", to_be_checked)




        
if __name__ == '__main__':
    main()

