from __future__ import unicode_literals
import pandas as pd
import random
import nltk, string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
import re

from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from nltk import ngrams
import urllib
from bs4 import BeautifulSoup
import requests
import webbrowser
from gtts import gTTS
import os    



speak = gTTS(text="This is the pc speaking", lang='en')
speak.save("pcvoice.mp3")

import speech_recognition as sr
r=sr.Recognizer()
r.energy_threshold=400
r.dynamic_energy_threshold=False
import time
nltk.download('averaged_perceptron_tagger')

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('Dataset_Library.csv',sep=',',engine='python')
cq=pd.read_csv('DatasetQUESTIONS.csv',sep=',')
ca=pd.read_csv('DatasetANSWERS.csv',sep=',')
qq_all=pd.read_csv('QQ.csv',sep=',', engine='python')
qa_all=pd.read_csv('QA.csv',sep=',')

qq=qq_all[qq_all['Review'].isna()==True].reset_index(drop=True)
qa=qa_all[qa_all['Review'].isna()==True].reset_index(drop=True)

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def lemmitize_tokens(tokens):
    return [wordnet_lemmatizer.lemmatize(item) for item in tokens]

def normalize(text):
    return stem_tokens(lemmitize_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map))))


vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
vectorizer_no_stop = TfidfVectorizer(tokenizer=normalize)

def cosine_sim(text1, text2,threshold):
    x=0

    try:
        tfidf = vectorizer.fit_transform([text1, text2])
    except:
        tfidf = vectorizer_no_stop.fit_transform([text1, text2])
    x=((tfidf * tfidf.T).A)[0,1]
    
    if x<threshold:
        try:
            tfidf = vectorizer.fit_transform([cleanupthestatement(text1), getSetIntersection(text1,text2)])
        except:
            tfidf = vectorizer_no_stop.fit_transform([cleanupthestatement(text1), getSetIntersection(text1,text2)])
        x=((tfidf * tfidf.T).A)[0,1]
        
        if x<0.95:
            x=0
    
    return x


def cleanupthestatement(stmt):
    stmt=re.sub(r'[^\w\s]','',stmt)
    reconstructed_sentence=""
    word_tokens=nltk.word_tokenize(str(stmt))
    if len(word_tokens)<5:
        reconstructed_sentence=stmt
    else:
        no_stop_word = [w for w in word_tokens if not w in stop_words] 
        no_stop_word = [] 
        for w in word_tokens: 
            if w not in stop_words: 
                no_stop_word.append(w)
        lemmitized_sentence = [w for w in no_stop_word] 
        lemmitized_sentence = [] 
        for w in no_stop_word: 
            lemmitized_sentence.append(wordnet_lemmatizer.lemmatize(w))
        reconstructed_sentence=""
        for w in lemmitized_sentence: 
            reconstructed_sentence=reconstructed_sentence + w + " "
    reconstructed_sentence=reconstructed_sentence.strip()
    return reconstructed_sentence




def getSetIntersection(stmt1,stmt2):
    s1=set(cleanupthestatement(stmt1).split())
    s2=set(cleanupthestatement(stmt2).split())
    s=s1 & s2
    txt=""
    lst = list(s)
    for w in lst: 
        txt=txt + w + " "
    txt=str(txt).strip()
    return txt




pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB'],
    'wh' : ['WDT','WP','WRB'],
    'mdc' : ['MD']
}
def getcountof(x, flag):
    cnt = 0
    x=stmt=re.sub(r'[^\w\s]','',x)
    try:
        wiki=nltk.pos_tag(nltk.word_tokenize(str(x.lower())))
        for wrd,tup in wiki:
            if tup in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt     




def getNgram(stmt):
    stmt=re.sub(r'[^\w\s]','',stmt)
    lst=[]
    n = 3
    threegrams = ngrams(nltk.word_tokenize(stmt), n)
    for grams in threegrams:
        lst.append(str(nltk.pos_tag(grams)[0][1] + '-' + nltk.pos_tag(grams)[1][1] + '-' + nltk.pos_tag(grams)[2][1]) )
    return lst

def checkFistWordq(stmt):
    stmt=re.sub(r'[^\w\s]','',stmt.strip())
    cnt=0
    lst=getNgram(stmt)
    if len(lst)>0:
        chkwrd=lst[0].split('-')[0]
        if chkwrd in pos_family['verb'] or chkwrd in pos_family['wh'] or chkwrd in pos_family['mdc']:
            cnt=1
    return cnt

def getfeatures(xx):
    xx['char_count'] = xx['Statements'].apply(len)
    xx['word_count'] = xx['Statements'].apply(lambda x: len(x.split()))
    xx['word_density'] = xx['char_count'] / (xx['word_count']+1)
    xx['punctuation_count'] = xx['Statements'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
    xx['title_word_count'] = xx['Statements'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    xx['upper_case_word_count'] = xx['Statements'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    xx['noun_count'] = xx['Statements'].apply(lambda x: getcountof(x,'noun'))
    xx['verb_count'] = xx['Statements'].apply(lambda x: getcountof(x,'verb'))
    xx['adj_count'] = xx['Statements'].apply(lambda x: getcountof(x,'adj'))
    xx['adv_count'] = xx['Statements'].apply(lambda x: getcountof(x,'adv'))
    xx['pron_count'] = xx['Statements'].apply(lambda x: getcountof(x,'pron'))
    xx['wh_count'] = xx['Statements'].apply(lambda x: getcountof(x,'wh'))
    xx['ngram_count'] = xx['Statements'].apply(lambda x: len(getNgram(x)))
    xx['chk_fword'] = xx['Statements'].apply(lambda x: checkFistWordq(x))
   


getfeatures(df)




rfr=RandomForestClassifier(n_estimators=10,random_state=123)
X=df.iloc[:,3:].values
Y=df.iloc[:,2:3].values
Y_labelencoder=LabelEncoder()
Y=Y_labelencoder.fit_transform(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
rfr.fit(X_train, Y_train)


def findcategory(userinput):
    #convert input to dataframe
    uip=pd.DataFrame(data=[[userinput]],columns=["Statements"])
    getfeatures(uip)
    inputvalues=uip.iloc[:,1:].values
    category=rfr.predict(inputvalues)[0]
    return Y_labelencoder.classes_[int(category)]




def getsimilary(stmt,doc,chattype,threshold):
    similarity=0
    if re.sub(r'[^\w\s]','',stmt).lower()==re.sub(r'[^\w\s]','',doc).lower():
        similarity=5
    else:
        similarity=cosine_sim(str(stmt).lower(),str(doc).lower(),threshold)
    return similarity




def getquestanswer(stmt,threshold):
   
    ans=""
    try:
        qq['sim']=qq.apply(lambda x : getsimilary(stmt,x['Question'],'q',threshold),axis=1)
        qa['sim']=qa.apply(lambda x : getsimilary(stmt,x['Answer'],'q',threshold),axis=1)
        if max(qq['sim'])>=threshold:
            x=qq[qq['sim']==max(qq['sim'])]
            ansnum=x.reset_index()['Answer'][0]
            ans=str(qa[qa['Sn']==int(random.choice(ansnum.split(',')))].reset_index()['Answer'][0])
        else:
            if max(qa['sim'])>=threshold:
                ans=qa[qa['sim']==max(qa['sim'])].reset_index()['Answer'][0]
    except:
        ans='I am sorry, but I do not understand'
    if ans=="":
        ans='I am sorry, but I do not understand'
    return str(ans)

def getchatanswer(stmt,threshold):
    ans=""
    try:
        cq['sim']=cq.apply(lambda x : getsimilary(stmt,x['Question'],'c',threshold),axis=1)
        if max(cq['sim'])>=threshold:
            x=cq[cq['sim']==max(cq['sim'])]
            ansnum=x.reset_index()['Answer'][0]
            ans=str(ca[ca['Sn']==int(random.choice(ansnum.split(',')))].reset_index()['Answer'][0])
        else:
            ans=getquestanswer(stmt,threshold)
    except:
        ans='I am sorry, but I do not understand'
    if ans=="":
        ans='I am sorry, but I do not understand'
    return str(ans)



def getGoogleResponse(text):
    resp=""
    try:
        text = urllib.parse.quote_plus(text)
        url = 'https://google.com/search?q=' + text
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        for g in soup.find_all(class_='g'):
            strs=(g.text)
            break
        resp=strs.split('.')[0].strip()
        if resp.find('http')>1:
            resp='Sorry, I could not understand'
    except:
        pass
    return resp




def saveUnAsnweredQuestInLibrary(question):
    try:
        qq_all=pd.read_csv('QQ.csv')
        maxsn=int(max(qq_all['Sn']))+1
        fl=open('QQ.csv','a')
        fl.writelines("\n"+str(maxsn)+","+question+",,"+'u')
        fl.close()
    except:
        pass

def speakup(usrip):
    print('Cotina : ' + usrip)
    speak.Speak(usrip)

def listentome():
    usrip=""
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("You : ")
            audio=r.listen(source)
            usrip=str(r.recognize_google(audio))
            print(usrip)
    except:
        speakup("Please ask something or say Bye to End the conversation.")
        usrip=""
    return usrip



def AskQuestion():
    
    while True:
        userip=listentome()
        if userip!="":
            answer=""
            if userip.lower()=='bye':
                speakup("Bye. Nice to meet you")
                break
            else:
                if userip.strip()!="":
                    if findcategory(userip)=="c":
                        answer=getchatanswer(userip,0.6)
                    else:
                        answer=getquestanswer(userip,0.6)

                    if answer=="I am sorry, but I do not understand":
                        saveUnAsnweredQuestInLibrary(userip.strip())
                        answer=getGoogleResponse(userip)

                    speakup(answer)




def saveAnswer(answer,qno,question):
    maxsn=0
    try:
        qa_all=pd.read_csv('QA.csv')
        maxsn=int(max(qa_all['Sn']))+1
        fl=open('QA.csv','a')
        fl.writelines("\n"+str(maxsn)+","+answer + ","+'r')
        fl.close()        

        question_Text=str(qno) + "," + str(question) + "," + str(maxsn) + "," + 'r'
        with open("QQ.csv","r+") as f:
            new_f = f.readlines()
            f.seek(0)
            for line in new_f:
                if line.split(',')[0]==str(qno):
                    f.write(question_Text + "\n")
                else:
                    f.write(line)
            f.truncate()
            f.close()
    except:
        pass




def updateLibrary():
    usrresp=""
    greet_msg=''
    cntQuestion=0
    counter=0
    try:
        qq_all=pd.read_csv('QQ.csv')
        qq_unanswered=qq_all[qq_all['Review'].fillna('X')=='u'].reset_index(drop=True)
        cntQuestion=int(qq_unanswered['Review'].count())
        
        if cntQuestion==0:
            speakup('Oh! Sorry. I do not have any open questions on my library. Thanks')
            time.sleep(1)
        
        else:
            speakup('I have ' + str(cntQuestion) + ' unanswered questions. I will ask you those one by one. Please say next to skip or Bye to exit the conversation.')
            time.sleep(1)
                
            while counter<cntQuestion:

                speakup(qq_unanswered['Question'][counter])
                time.sleep(1)

                usrresp=listentome()
                if usrresp!="":
                    usrresp=usrresp.lower().strip()
                    if usrresp=='bye':
                        speakup('Bye. Nice to meet you.')
                        break
                    elif usrresp=='next':
                        counter=counter + 1
                    else:
                        saveAnswer(usrresp,int(qq_unanswered['Sn'][counter]),qq_unanswered['Question'][counter])
                        counter=counter + 1

            if counter==cntQuestion and counter>0:
                speakup('Thank you for your response.')

    except:
        pass



def TalkNLearn():
    usrresp=""
    speakup(str(random.choice(['Hi, My Name is Cotina.','Hi, This is Cotina','Hi there, its me Cotina','Olaa. I am Cotina'])))
    time.sleep(1)
    askflag=False
    while True:
        if askflag==False:
            speakup(str(random.choice(['Do you want to review the library or ask me questions','Please let me know if you would like to contribute to my library or ask me question? To exit the conversation, please say Bye'])))
            askflag=True

        usrresp=listentome()
        if usrresp=="":
            askflag=False
        
        if usrresp!="":
            usrresp=usrresp.lower().strip()
            if usrresp.find('contribute')>0 or usrresp.find('library')>0:
                askflag=False
                updateLibrary()
                break
            elif usrresp.find('ask')>0 or usrresp.find('question')>0:
                speakup('Sure. Please go ahead and ask your question.')
                time.sleep(1)
                askflag=False
                AskQuestion()
                break
            elif usrresp=='bye' or usrresp=='exit':
                speakup('Bye. Thank you for your time.')
                break



def AskQuestionText():
    
    while True:
        userip=input('You : ').strip()
        if userip!="":
            answer=""
            if userip.lower()=='bye':
                print('Cotina : Bye. Nice to meet you')
                break
            else:
                if userip.strip()!="":
                    if findcategory(userip)=="c":
                        answer=getchatanswer(userip,0.6)
                    else:
                        answer=getquestanswer(userip,0.6)

                    if answer=="I am sorry, but I do not understand":
                        saveUnAsnweredQuestInLibrary(userip.strip())
                        answer=getGoogleResponse(userip)

                    
                    print('Cotina : ' + answer)



TalkNLearn()


AskQuestionText()

