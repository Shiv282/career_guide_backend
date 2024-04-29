# **1. Importing Necessary Libraries** 📚

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pickle 
import time
from db import *
from keras.models import load_model
import re
from keras.preprocessing.sequence import pad_sequences

#pickleFile=open("new_weights.pkl","rb")
#regressor=pickle.load(pickleFile) # our model


# **2. Loading Dataset**
'''
df = pd.read_csv('./mldata.csv')
df.head()

df['workshops'] = df['workshops'].replace(['testing'],'Testing')
df.head()

#print(df.columns.unique)

n = df['Suggested Job Role'].unique()
#print(len(n))

#print('The shape of our training set: %s professionals and %s features'%(df.shape[0],df.shape[1]))


# **5. Feature Engineering**

## (a) Binary Encoding for Categorical Variables

newdf = df
newdf.head(10)

cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
for i in cols:
    #print(i)
    cleanup_nums = {i: {"yes": 1, "no": 0}}
    df = df.replace(cleanup_nums)

#print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

## (b) Number Encoding for Categorical 

mycol = df[["reading and writing skills", "memory capability score"]]
for i in mycol:
    #print(i)    
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
    df = df.replace(cleanup_nums)

category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 
                    'Interested Type of Books']]
for i in category_cols:
    df[i] = df[i].astype('category')
    df[i + "_code"] = df[i].cat.codes

#print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

## (c) Dummy Variable Encoding

#print(df['Management or Technical'].unique())
#print(df['hard/smart worker'].unique())

df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])
df.head()

df.sort_values(by=['certifications'])

#print("List of Numerical features: \n" , df.select_dtypes(include=np.number).columns.tolist())


category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 'Interested Type of Books']]
for i in category_cols:
  print(i)

Certifi = list(df['certifications'].unique())
print(Certifi)
certi_code = list(df['certifications_code'].unique())
#print(certi_code)

Workshops = list(df['workshops'].unique())
#print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
#print(Workshops_code)

Certi_l = list(df['certifications'].unique())
certi_code = list(df['certifications_code'].unique())
C = dict(zip(Certi_l,certi_code))

Workshops = list(df['workshops'].unique())
#print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
#print(Workshops_code)
W = dict(zip(Workshops,Workshops_code))

Interested_subjects = list(df['Interested subjects'].unique())
#print(Interested_subjects)
Interested_subjects_code = list(df['Interested subjects_code'].unique())
ISC = dict(zip(Interested_subjects,Interested_subjects_code))

interested_career_area = list(df['interested career area '].unique())
#print(interested_career_area)
interested_career_area_code = list(df['interested career area _code'].unique())
ICA = dict(zip(interested_career_area,interested_career_area_code))

Typeofcompany = list(df['Type of company want to settle in?'].unique())
#print(Typeofcompany)
Typeofcompany_code = list(df['Type of company want to settle in?_code'].unique())
TOCO = dict(zip(Typeofcompany,Typeofcompany_code))

Interested_Books = list(df['Interested Type of Books'].unique())
#print(Interested_subjects)
Interested_Books_code = list(df['Interested Type of Books_code'].unique())
IB = dict(zip(Interested_Books,Interested_Books_code))

Range_dict = {"poor": 0, "medium": 1, "excellent": 2}
#print(Range_dict)


A = 'yes'
B = 'No'
col = [A,B]
for i in col:
  if(i=='yes'):
    i = 1
  #print(i)


f =[]
A = 'r programming'
clms = ['r programming',0]
for i in clms:
  for key in C:
    if(i==key):
      i = C[key]
      f.append(i)
#print(f)

C = dict(zip(Certifi,certi_code))
  
#print(C)

import numpy as np
array = np.array([1,2,3,4])
array.reshape(-1,1)

def inputlist(Name,Contact_Number,Email_address,
      Logical_quotient_rating, coding_skills_rating, hackathons, 
      public_speaking_points, self_learning_capability, 
      Extra_courses_did, Taken_inputs_from_seniors_or_elders,
      worked_in_teams_ever,Introvert, reading_and_writing_skills,
      memory_capability_score, smart_or_hard_work, Management_or_Techinical,
      Interested_subjects, Interested_Type_of_Books,certifications, workshops, 
      Type_of_company_want_to_settle_in, interested_career_area):
  #1,1,1,1,'Yes','Yes''Yes''Yes''Yes',"poor","poor","Smart worker", "Management","programming","Series","information security"."testing","BPA","testing"
  Afeed = [Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points]

  input_list_col = [self_learning_capability,Extra_courses_did,Taken_inputs_from_seniors_or_elders,worked_in_teams_ever,Introvert,reading_and_writing_skills,memory_capability_score,smart_or_hard_work,Management_or_Techinical,Interested_subjects,Interested_Type_of_Books,certifications,workshops,Type_of_company_want_to_settle_in,interested_career_area]
  feed = []
  K=0
  j=0
  for i in input_list_col:
    if(i=='Yes'):
      j=2
      feed.append(j)
       
      #print("feed 1",i)
    
    elif(i=="No"):
      j=3
      feed.append(j)
       
      #print("feed 2",j)
    
    elif(i=='Management'):
      j=1
      k=0
      feed.append(j)
      feed.append(K)
       
      #print("feed 10,11",i,j,k)

    elif(i=='Technical'):
      j=0
      k=1
      feed.append(j)
      feed.append(K)
       
      #print("feed 12,13",i,j,k)

    elif(i=='Smart worker'):
      j=1
      k=0
      feed.append(j)
      feed.append(K)
       
      #print("feed 14,15",i,j,k)

    elif(i=='Hard Worker'):
      j=0
      k=1
      feed.append(j)
      feed.append(K)
      #print("feed 16,17",i,j,k)
    
    else:
      for key in Range_dict:
        if(i==key):
          j = Range_dict[key]
          feed.append(j)
         
          #print("feed 3",i,j)

      for key in C:
        if(i==key):
          j = C[key]
          feed.append(j)
          
          #print("feed 4",i,j)
      
      for key in W:
        if(i==key):
          j = W[key]
          feed.append(j)
          
          #print("feed 5",i,j)
      
      for key in ISC:
        if(i==key):
          j = ISC[key]
          feed.append(j)
          
          #print("feed 6",i,j)

      for key in ICA:
        if(i==key):
          j = ICA[key]
          feed.append(j)
          
          #print("feed 7",i,j)

      for key in TOCO:
        if(i==key):
          j = TOCO[key]
          feed.append(j)
          
          #print("feed 8",i,j)

      for key in IB:
        if(i==key):
          j = IB[key]
          feed.append(j)
          
          #print("feed 9",i,j)

   
       
  t = Afeed+feed    
  output = regressor.predict([t])
  
  return(output)'''

'''sample_input = {
        "Name": "John",
        "Contact_Number": "1234567890",
        "Email_address": "john@example.com",
        "Logical_quotient_rating": 1,
        "coding_skills_rating": 1,
        "hackathons": 1,
        "public_speaking_points": 1,
        "self_learning_capability": "Yes",
        "Extra_courses_did": "Yes",
        "Taken_inputs_from_seniors_or_elders": "Yes",
        "worked_in_teams_ever": "Yes",
        "Introvert": "poor",
        "reading_and_writing_skills": "poor",
        "memory_capability_score": "Smart worker",
        "smart_or_hard_work": "Smart worker",
        "Management_or_Techinical": "Management",
        "Interested_subjects": "programming",
        "Interested_Type_of_Books": "Series",
        "certifications": "testing",
        "workshops": "BPA",
        "Type_of_company_want_to_settle_in": "BPA",
        "interested_career_area": "information security"
    }'''


# Call the function
#sample_input = ['','','',10,10,10,10,]


'''

['Logical quotient rating'
,'coding skills rating', 
'hackathons', 
'public speaking points', 
'self-learning capability?',
'Extra-courses did', 
'Taken inputs from seniors or elders', 
'worked in teams ever?', 
'Introvert', 
'reading and writing skills', 
'memory capability score',  
'B_hard worker', 
'B_smart worker', 
'A_Management', 
'A_Technical', 
'Interested subjects_code', 
'Interested Type of Books_code', 
'certifications_code', 
'workshops_code', 
'Type of company want to settle in?_code',  
'interested career area _code',
'Suggested Job Role']
'''



#sample_input = ['2','3','6','5','0','1','1', '1', '1', '2', '0', '0','1','1','0','7','13','7','2','7','5']
#print(interested_career_area)
#print(interested_career_area_code)
#output = regressor.predict([sample_input])
#result = inputlist(sample_input)
#print("Result ::::::",output)


def predict(input_list):
  with open('model.pkl', 'rb') as f:
    regressor = pickle.load(f)
  output = regressor.predict([input_list])
  return output

def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

def vak_predict(sentences):
   model = load_model('./LearningStyleClassifier_new.h5')
   with open('tokenizer_new.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
   with open('labelEncoder_new.pickle', 'rb') as le:
    le = pickle.load(le)
   
   output = []
   for sentence in sentences:
    temp = []
    print(sentence)
    sentence = clean(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=48, truncating='pre')
    result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba =  np.max(model.predict(sentence))
    print(f"{result} : {proba}\n\n")
    temp.append(result)
    temp.append(proba)
    output.append(temp)
   return output
