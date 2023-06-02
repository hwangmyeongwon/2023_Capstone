from flask import Flask,render_template,json
import openai
import speech_recognition as sr
from pyautogui import *
import clipboard,keyboard,pyaudio,time
from pydub import AudioSegment
import pymysql,random,requests
from google.cloud import texttospeech
from playsound import playsound

app=Flask(__name__)

cnt=0
messages=[]

def tt_speak(t):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', 'Korean')
    engine.say(t)
    return engine.runAndWait()

def fixedstart(Type):
    if(Type=='WEB'):
        user_content = "당신은 백엔드 전문가입니다. 지금은 면접 상황입니다. 당신은 면접관 입니다. 사용자가 답변을 하면 답변을 보고 질문을 해주세요.한국어로만 말해줘.당신은 면접관 입니다.면접관 처럼 말하세요.질문은 하나만 해줘.한국어만 말해주세요.면접관처럼 해줘"
        messages.append({"role" : "user", "content" : f"{user_content}"})
    elif(Type=='AI'):
        user_content = "당신은 딥러닝 전문가 입니다. 지금은 면접 상황 입니다. 당신은 면접관 입니다. 사용자가 답변을 하면 그 답변을 보고 질문을 한국어로만 해주세요."
        messages.append({"role" : "user", "content" : f"{user_content}"})
def st_text():
    global transcript
    r= sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        audio = r.listen(source) # 음성 읽어오기
        
    try:
        transcript=r.recognize_google(audio, language="ko-KR")
        print("Google Speech Recognition thinks you said "+transcript)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return transcript

def gpt(user_content):
    
    openai.api_key = "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"#나만의 APi key
    global messages
    
    user_content = st_text() #사용자 입력
    messages.append({"role" : "user", "content" : f"{user_content}"})#유저가 보낼 메세지를 리스트에 저장하고        
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)#메세지가 들어가서 GPT 알고리즘
    assistant_content = completion.choices[0].message["content"].strip()
    messages.append({"role": "assistant", "content": f"{assistant_content}"})#챗봇에게 응답받은 메세지를 리스트에 저
    
    last_messages = messages[-1]
    return last_messages["content"]

def skill(Type):
    if(Type=="WEB"):
        sql="SELECT * FROM interview_type.SkillAsk WHERE Type='WEB'"
    elif(Type=="AI"):
        sql="SELECT * FROM interview_type.Ityoe WHERE Type='딥러닝'"
    cursor.execute(sql)
    data_list=cursor.fetchall()
    random_row = random.choice(data_list)
    cursor.close()
    conn.close()
    return random_row[1]

def manstart():
    user_content = "안녕하세요. 지금부터 면접을 진행하겠습니다. 당신은 회사 임원입니다. 지금은 인적성 관련 질문 면접 상황입니다. 당신은 면접관입니다. 면접관처럼 말해주세요. 당신은 질문만 하세요. 사용자가 답변을 하면 답변을 보고 질문을 해주세요. 그리고 질문은 하나씩 해주세요."
    messages.append({"role" : "user", "content" : f"{user_content}"})     

def interviewing():
    video_url = '../static/img/interviewing.mp4'
    return render_template('interviews.html', video_url=video_url)

def listening():
    Video_url ='../static/img/listening.mp4'
    return render_template('interviews.html', video_url=video_url)    

@app.route('/') #메인페이지(접속)
def home(): 
    return render_template('mainpage.html')
@app.route('/mainpages')
def mainpage():
    return render_template('mainpage2.html')
@app.route('/prepare') #면접 준비창 (카메라 확인,음성확인)
def prepare():
    return render_template('PreparingInterview.html')
@app.route('/prepareS')
def prepares():
    return render_template('PreparingInterview2.html')
@app.route('/mypage') #마이페이지
def mypage():
    return render_template('myPage.html')
@app.route('/inter') #면접보는페이지
def interhome():
    return render_template('interview.html')
@app.route('/result') #피드백 페이지
def result():
    return render_template('result.html')
@app.route('/results')
def results():
    return render_template('result2.html')
@app.route('/mainS') #마이페이지로 들어가는 메인페이지
def mains():
    return render_template('mainS.html')
@app.route('/signIn')
def signIn():
    return render_template('signIn.html')
@app.route('/signUp')
def signUp():
    return render_template('signUp.html')


@app.route('/get')
def get_bot_speak():
        
        global cnt
        global data
        user_content = "" 
        while(True):
            global Type
            cnt+=1
            if(cnt==1):
                data=fixedstart(Type)
                data="안녕하세요. 저는 AI 면접관 '갠원' 입니다. 지금 부터 면접을 시작 할게요. 자기소개와 프로젝트를 말해주세요."
                tt_speak(data)
            elif(cnt==2):
                user_content = st_text()
            elif(cnt==3):
                data=gpt(user_content)
                tt_speak(data)
            elif(cnt==4):
                user_content=st_text()
            elif(cnt==5):
                data= gpt(user_content)
                tt_speak(data)
            elif(cnt==6):
                user_content=st_text()
            elif(cnt==7):
                data="이제 기술면접 질문으로 넘어가겠습니다."
                data=skill(Type)
                tt_speak(data)
            elif(cnt==8): 
                user_content=st_text()
            elif(cnt==9):
                data = manstart()
                data = "인적성 검사 시작하겠습니다. 본인의 좌우명은 무엇입니까 "
                messages.append({"role" : "user", "content" : f"{data}"})   
                tt_speak(data)
            elif(cnt==10):
                user_content=st_text()
            elif(cnt==11):
                data= gpt(user_content)
                tt_speak(data)
            elif(cnt==12):
                user_content=st_text()
            elif(cnt==13):
                data= gpt(user_content)
                tt_speak(data)
            elif(cnt==14):
                user_content=st_text()
                
            elif(cnt==15):
                data = "면접이 끝났습니다. 수고하셨습니다."
                tt_speak(data)
          
            return json.dumps(data)



if __name__ == '__main__':
    app.run(debug=True)
