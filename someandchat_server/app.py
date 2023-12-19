# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import openai
import firebase_admin
from firebase_admin import credentials, auth, firestore
from firebase_admin import db
import requests
import json
from datetime import datetime, timedelta
import pytz
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'  # 세션을 파일 시스템에 저장
Session(app)
openai.api_key = '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'

cred = credentials.Certificate("someandchatkey.json")
# Firebase 앱을 초기화합니다.
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://someandchat-default-rtdb.firebaseio.com/' # Firebase Realtime Database URL
})
# Firestore 클라이언트 초기화
firedb = firestore.client()
# GRU 모델 불러오기
model = tf.keras.models.load_model('model/likeabiliyt_gru.h5')

relationship_stages = {
    "0%": "서로 간에 아직 관심이 없거나 막 알게 된 상태.#대화가 주로 일상적이거나 겉돌며, 개인적인 정보를 거의 공유하지 않음.#서로의 존재를 잘 모르거나, 만남이 거의 없음.",
    "25%": "서로에 대해 약간의 관심이 생기기 시작함.#개인적인 이야기를 나누기 시작함.#서로를 좀 더 알고 싶어하는 호기심이 생김.",
    "50%": "서로에 대한 관심이 명확해지고, 더 자주 만나고 싶어함.#대화가 깊어지며, 서로의 취향이나 생각을 공유함.#서로를 향한 애정이나 호감이 분명해지기 시작함.",
    "75%": "서로에 대한 강한 호감과 애정이 생김.#자주 만나며, 서로의 일상에 깊이 관여하기 시작함.#서로를 향한 진지한 감정이나 미래에 대한 생각을 공유함.",
    "100%": "서로에 대한 깊은 애정과 사랑을 느낌.#관계가 연인으로 발전할 준비가 되어 있음.#서로에 대한 신뢰와 이해가 깊으며, 함께 시간을 보내는 것이 매우 자연스러움."
}

# .txt 파일에서 데이터 불러오기
def load_data(file):
    data = file.read().decode('utf-8')
    return data

# 텍스트 전처리
def preprocess_text(text):
    text_cleaned = re.sub(r"[^가-힣A-Za-z0-9]", " ", text)
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([text_cleaned])
    sequence = tokenizer.texts_to_sequences([text_cleaned])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    return padded_sequence

# 호감도 추론
def predict_likeability(text):
    preprocessed_text = preprocess_text(text)
    predictions = model.predict(preprocessed_text)
    return np.argmax(predictions, axis=-1)


@app.route('/')
def home():
    uid = session.get('uid')
    if uid:
        # UID가 세션에 저장되어 있으면 로그인 상태로 간주
        # 여기서 uid를 메인 페이지로 전달하거나 필요한 작업을 수행할 수 있습니다.
        chatbots = firedb.collection(uid).stream()

        # Firestore에서 가져온 데이터의 'name' 필드를 리스트로 변환
        chatbot_list = [chatbot.to_dict().get('name') for chatbot in chatbots]
        return render_template("index.html", chatbot_list=chatbot_list)
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

@app.route('/chat/<chatbot_name>')
def chat(chatbot_name):
    uid = session.get('uid')
    if uid:
        # UID가 세션에 저장되어 있으면 로그인 상태로 간주
        doc_ref = firedb.collection(uid).document(chatbot_name)
        doc = doc_ref.get()

        if doc.exists:
            ref = db.reference(f'/{chatbot_name}{uid}')
            data = ref.get()
            chat_history = []
            if data:
                for key, value in data.items():
                    chat_history.append(value)
            return render_template("chat.html", chatbot_name=chatbot_name, chat_history=chat_history)
        else:
            return render_template("fail.html")
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

@app.route('/chat/')
def chatFail():
    uid = session.get('uid')
    if uid:
        return render_template("fail.html")
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

@app.route('/chatRoom/')
def chatRoom():
    uid = session.get('uid')
    if uid:
        # UID가 세션에 저장되어 있으면 로그인 상태로 간주
        # Firestore에서 사용자의 챗봇 목록을 가져옵니다.
        chatbots = firedb.collection(uid).stream()
        chatbot_list = []

        for chatbot in chatbots:
            chatbot_dict = chatbot.to_dict()
            chatbot_name = chatbot_dict.get('name')
            ref = db.reference(f'/{chatbot_name}{uid}')
            data = ref.get()
            if data:
                last_key = list(data.keys())[-1]
                last_message = data[last_key].get('assistant', "메시지 없음")
                last_time = data[last_key].get('time', "알 수 없음")
            else:
                last_message = '채팅을 시작해보세요!'
                last_time = ''

            chatbot_list.append({
                'name': chatbot_name,
                'last_message': last_message,
                'last_time': format_time_ago(last_time) if last_time else ''
            })
        return render_template("chatRoom.html", chatroom_list=chatbot_list)
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

@app.route('/login/')
def login():
    return render_template("login.html")

@app.route('/hotplace/')
def hotplace():
    uid = session.get('uid')
    if uid:
        # UID가 세션에 저장되어 있으면 로그인 상태로 간주
        # 여기서 uid를 메인 페이지로 전달하거나 필요한 작업을 수행할 수 있습니다.
        return render_template("hotplace.html")
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

@app.route('/hotplace/seoul')
def hotplaceSeoul():
    uid = session.get('uid')
    if uid:
        # UID가 세션에 저장되어 있으면 로그인 상태로 간주
        # 여기서 uid를 메인 페이지로 전달하거나 필요한 작업을 수행할 수 있습니다.
        return render_template("hotplaceSeoul.html")
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

@app.route('/hotplace/busan')
def hotplaceBusan():
    uid = session.get('uid')
    if uid:
        # UID가 세션에 저장되어 있으면 로그인 상태로 간주
        # 여기서 uid를 메인 페이지로 전달하거나 필요한 작업을 수행할 수 있습니다.
        return render_template("hotplaceBusan.html")
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

@app.route('/hotplace/jeju')
def hotplaceJeju():
    uid = session.get('uid')
    if uid:
        # UID가 세션에 저장되어 있으면 로그인 상태로 간주
        # 여기서 uid를 메인 페이지로 전달하거나 필요한 작업을 수행할 수 있습니다.
        return render_template("hotplaceJeju.html")
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

@app.route('/someanalysis')
def someAnalysis():
    uid = session.get('uid')
    if uid:
        # UID가 세션에 저장되어 있으면 로그인 상태로 간주
        # 여기서 uid를 메인 페이지로 전달하거나 필요한 작업을 수행할 수 있습니다.
        return render_template("someAnalysis.html")
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

@app.route('/makechatbot')
def makeChatbot():
    uid = session.get('uid')
    if uid:
        # UID가 세션에 저장되어 있으면 로그인 상태로 간주
        # 여기서 uid를 메인 페이지로 전달하거나 필요한 작업을 수행할 수 있습니다.
        return render_template("makeChatbot.html")
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

@app.route('/register')
def register():
    return render_template("register.html")


@app.route('/signup', methods=['POST'])
def signup():
    if request.method == 'POST':
        data = request.json
        email = data['email']
        password = data['password']

        try:
            user = auth.create_user(
                email=email,
                password=password
            )
            print("회원가입 성공:", user.uid)
            # 회원가입 후 리다이렉트 또는 다른 응답을 반환할 수 있습니다.
            return "성공"
        except Exception as e:
            print("회원가입 실패:", str(e))
            # 회원가입 후 리다이렉트 또는 다른 응답을 반환할 수 있습니다.
            if(str(e) == "The user with the provided email already exists (EMAIL_EXISTS)."):
                return "이미 해당 아이디로 가입한 정보가 있습니다."
            elif(str(e) == "Error while calling Auth service (INVALID_EMAIL)."):
                return "이메일 형식이 잘못되었습니다."
            elif(str(e) == "Invalid password string. Password must be a string at least 6 characters long."):
                return "비밀번호는 6자 이상이어야 합니다."
            else:
                return "실패"


@app.route('/loginfire', methods=['POST'])
def loginFire():
    if request.method == 'POST':
        data = request.json
        email = data['email']
        password = data['password']

        # Firebase API 키 (Web API Key)
        api_key = 'AIzaSyBf-B79GbewDUFUhzZDPuIOs3O1O6bVjJA'

        # Firebase 인증 REST API URL
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"

        # 이메일과 비밀번호를 사용하여 Firebase에 로그인 요청
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"email": email, "password": password, "returnSecureToken": True})

        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()  # 오류 발생 시 예외 발생

            # 성공적인 응답 처리
            user_details = response.json()
            print("로그인 성공:", user_details['localId'])
            # 로그인 성공 시 세션에 UID 저장
            user = auth.get_user_by_email(email)
            session['uid'] = user.uid
            return "성공"
        except requests.exceptions.HTTPError as e:
            # HTTP 오류 처리
            error_json = e.response.json()
            error_message = error_json.get("error", {}).get("message", "")
            print("로그인 실패:", error_message)

            if error_message == "EMAIL_NOT_FOUND":
                return "해당 이메일이 존재하지 않습니다."
            elif error_message == "INVALID_PASSWORD":
                return "비밀번호가 틀렸습니다."
            elif error_message == "INVALID_LOGIN_CREDENTIALS":
                return "이메일이나 비밀번호가 틀렸습니다"
            else:
                return "로그인 실패"

        except Exception as e:
            # 기타 예외 처리
            print("로그인 실패:", str(e))
            return "예외 실패"

@app.route('/chat/send_message', methods=['POST'])
def send_message():
    uid = session.get('uid')
    user_message = request.form['message']  # 혹은 request.json['message']
    chatbot_name = request.form['chatbot_name']
    ref = db.reference(f'/{chatbot_name}{uid}')
    data = ref.get()
    chatbotInfo = firedb.collection(uid).document(chatbot_name).get().to_dict()
    chatbotInfo.get('details')
    messages = [{"role": "system",
                 "content": f"너의 이름은 '{chatbot_name}'이고 너의 MBTI는 '{chatbotInfo.get('mbti')}'이야. 그래서 '{chatbotInfo.get('mbti')}'의 성격을 가지고 있어. 너는 23살 여성이야. 너랑 대화하는 user는 남성이야. 너는 '{chatbotInfo.get('details')}'과 같은 특성을 가지고 있어. 채팅하는 상대방과 친해지고 싶어해. 친구가 되기 위해 노력해. 직업은 대학생이며 진짜 대학생이라고 생각하고 창작해서 답변해줘. 대화를 할 때 반말을 사용해. 존댓말은 친하지 않아 보이니 피해줘. 길게 얘기하지말고 되도록 한 문장으로 얘기해"}]
    # 여기서 ChatGPT와의 통신을 구현하고 결과를 받아온다.
    if data is not None:
        for key in data:
            messages.append({"role": "user", "content": data[key].get('user')})
            messages.append({"role": "assistant", "content": data[key].get('assistant')})
            messages.append({"role": "user", "content": user_message})
    completion = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",  # gpt-3.5-turbo gpt-4-1106-preview
        messages=messages
    )
    chat_response = completion.choices[0].message.content
    """ 대화 내용을 Firebase Realtime Database에 저장 """
    # 현재 UTC 시간을 가져옵니다.
    current_time_utc = datetime.utcnow()

    # ISO 8601 포맷으로 시간을 포매팅합니다.
    formatted_time_utc = current_time_utc.isoformat() + "Z"
    ref.push().set({"user": user_message, "assistant": chat_response, "time": formatted_time_utc})

    return jsonify({'message': chat_response})

@app.route('/logout', methods=['POST'])
def logout():
    if 'uid' in session:
        session.pop('uid')  # 세션에서 'uid'를 삭제합니다.
        return "로그아웃되었습니다."
    else:
        return "이미 로그아웃되었거나 로그인하지 않았습니다."

@app.route('/uploadBot', methods=['POST'])
def upload_bot():
    data = request.json
    chatbot_name = data.get('name')
    chatbot_mbti = data.get('mbti')
    chatbot_details = data.get('details')

    uid = session.get('uid')
    # Firestore에서 해당 chatbot_name 문서가 존재하는지 확인
    doc_ref = firedb.collection(uid).document(chatbot_name)
    doc = doc_ref.get()

    if doc.exists:
        # 이미 존재하는 경우 오류 메시지 반환
        return jsonify({"error": "챗봇 이름이 이미 존재합니다."}), 400

    # 새로운 챗봇 문서 생성
    doc_ref.set({
        'name': chatbot_name,
        'mbti': chatbot_mbti,
        'details': chatbot_details
    })

    return jsonify({"message": "챗봇 생성 성공", "name": chatbot_name, "mbti": chatbot_mbti, "details": chatbot_details})

def format_time_ago(time_str):
    if not time_str:
        return "알 수 없음"
    try:
        # 시간대 정보가 있는지 확인하고 UTC 시간으로 파싱합니다.
        if time_str.endswith('Z'):
            naive_time = datetime.fromisoformat(time_str.rstrip("Z"))
            utc_time = pytz.utc.localize(naive_time)
        else:
            # 시간대 정보가 없으면 이미 UTC로 간주합니다.
            utc_time = datetime.fromisoformat(time_str)

        # 한국 시간대로 변환합니다.
        korea_timezone = pytz.timezone('Asia/Seoul')
        korea_time = utc_time.astimezone(korea_timezone)

        # 현재 한국 시간을 구합니다.
        now_korea = datetime.now(korea_timezone)

        # 시간 차이를 계산합니다.
        time_diff = now_korea - korea_time
        seconds_diff = time_diff.total_seconds()

        # 음수 시간 차이를 처리합니다.
        if seconds_diff < 0:
            return "시간 데이터 오류"
        elif seconds_diff < 60:  # 1분 미만
            return f"{int(seconds_diff)}초 전"
        elif seconds_diff < 3600:  # 1시간 미만
            minutes_diff = seconds_diff / 60
            return f"{int(minutes_diff)}분 전"
        elif seconds_diff < 86400:  # 24시간 미만
            hours_diff = seconds_diff / 3600
            return f"{int(hours_diff)}시간 전"
        else:
            days_diff = seconds_diff / 86400
            return f"{int(days_diff)}일 전"
    except Exception as e:
        print(f"Error parsing time: {e}")
        return "알 수 없음"

@app.route('/model', methods=['POST'])
def model_endpoint():
    if 'file' not in request.files:
        print(1)
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files['file']
    if file.filename == '':
        print(2)
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    if file and file.filename.endswith('.txt'):
        text_data = load_data(file)
        likeability_score = predict_likeability(text_data)
        likeability_labels = ['0%', '25%', '50%', '75%', '100%']
        result = likeability_labels[likeability_score[0]]
        print(8)
        return jsonify({"result": result})

    else:
        print(3)
        return jsonify({"error": "텍스트 파일만 업로드 가능합니다."}), 400


@app.route('/result/<persent>')
def result(persent):
    uid = session.get('uid')
    if uid:
        sentence = "썸설명 \n\n그렇다 \t하하"
        return render_template("result.html", persent = persent, sentence = relationship_stages[persent])
    else:
        # 로그인되지 않은 상태면 로그인 페이지로 리다이렉트 또는 다른 처리 수행
        return render_template("redirect.html")

if __name__ == '__main__':
   app.run('0.0.0.0',port=5000,debug=True)