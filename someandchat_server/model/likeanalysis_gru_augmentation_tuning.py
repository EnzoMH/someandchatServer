# -*- coding: utf-8 -*-
"""호감도분석_GRU_데이터증강_20231215_Tuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vR7xu7IKLSM9AT62ccfHwvnpvdK0qKrs
"""

import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk # 텍스트 데이터를 처리
import numpy as np # 말뭉치를 배열로 표현
import random
import operator
import string # 표준 파이썬 문자열을 처리

from sklearn.metrics.pairwise import cosine_similarity # 코사인유사도 기반 문장의 맥락예측에 사용
from sklearn.feature_extraction.text import TfidfVectorizer # 벡터라이징 라이브러리


from google.colab import drive

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # LogisticRegression import 추가

import os
import numpy as np
# from google.colab import drive

# 구글 드라이브 마운트
# drive.mount('/content/drive')
def loadfile(path):
    X = []
    Y = []
    for label in ('0', '25', '50', '75', '100'):
        print("Loading text files for the label: " + label)
        label_path = os.path.join(path, label)
        for filename in os.listdir(label_path):
            if filename.endswith('.txt'):
                with open(os.path.join(label_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                # 레이블을 숫자로 변환
                if label == '0':
                    Y.append(0)
                elif label == '25':
                    Y.append(1)
                elif label == '50':
                    Y.append(2)
                elif label == '75':
                    Y.append(3)
                elif label == '100':
                    Y.append(4)
                # 텍스트 데이터를 X에 추가
                X.append(text)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
# 경로를 구글 드라이브 경로로 변경
directory_path = r'C:\Users\MyoengHo Shin\Desktop\likeability_son\'
# loadfile 함수 호출
X, Y = loadfile(directory_path)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

#4-1. 이모지 사용 함수
def count_emojis(text):
    emoji_pattern = re.compile('['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        ']', flags=re.UNICODE)
    return len(emoji_pattern.findall(text))


#4.2. 대화 양방향성 관련 함수
def check_bidirectional_conversation(text):
    a_contributions = len(re.findall(r'A:', text))
    b_contributions = len(re.findall(r'B:', text))
    return a_contributions > 0 and b_contributions > 0


# 4-3. 답장 속도 기반 호감도 예측
from datetime import datetime
import re
import numpy as np
# 대화 데이터를 .txt 파일에서 불러오기
conversation = []
for label in ('0', '25', '50', '75', '100'):
    label_path = os.path.join(directory_path, label)
    for filename in os.listdir(label_path):
        if filename.endswith('.txt'):
            with open(os.path.join(label_path, filename), 'r', encoding='utf-8') as file:
                conversation.extend(file.readlines())


# 호감도 레이블 생성 함수
def create_likeability_labels(conversation):
    response_times = []
    last_message_time = None
    for line in conversation:
        if line.startswith("A:") or line.startswith("B:"):
            # 시간 정보 추출 및 변환
            time_str = re.search(r'\((\d{2}):(\d{2})\)', line)
            if time_str:
                hours, minutes = map(int, time_str.groups())
                current_time = hours * 60 + minutes  # Convert to minutes
                # 답장 속도 계산
                if last_message_time is not None:
                    response_time = current_time - last_message_time
                    response_times.append(response_time)
                last_message_time = current_time
    # 평균 응답 시간 계산 및 호감도 레이블 생성
    avg_response_times = np.mean(response_times) if response_times else 0
    likeability_labels = []
    for response_time in response_times:
        if response_time <= 10:
            likeability = 100
        elif response_time <= 30:
            likeability = 75
        elif response_time <= 60:
            likeability = 50
        elif response_time <= 180:
            likeability = 25
        else:
            likeability = 0
        likeability_labels.append(likeability)
    return likeability_labels
# 대화 데이터를 기반으로 호감도 레이블 생성
likeability_labels = create_likeability_labels(conversation)

import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 데이터 증강 함수
def augment_text(text, num_augments=1):
    words = text.split()
    augmented_texts = []

    for _ in range(num_augments):
        random.shuffle(words)
        augmented_texts.append(' '.join(words))

    return augmented_texts

# 원본 데이터에 대한 데이터 증강
augmented_X = []
augmented_Y = []

for text, label in zip(X, Y):
    augmented_texts = augment_text(text, num_augments=2)  # 각 텍스트당 2개의 증강된 텍스트 생성
    augmented_X.extend(augmented_texts)
    augmented_Y.extend([label] * len(augmented_texts))

# 증강된 데이터를 원본 데이터에 추가 (수정)
X_extended = list(X) + augmented_X  # 리스트 합병
Y_extended = np.concatenate((Y, np.array(augmented_Y)))  # Numpy 배열 합병

# 데이터 전처리
# 1) 특수문자 제거
X_cleaned = [re.sub(r"[^가-힣A-Za-z0-9]", " ", text) for text in X_extended]

# 2) & 3) 토크나이징
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_cleaned)
X_seq = tokenizer.texts_to_sequences(X_cleaned)

# 4) & 5) 포스트패딩
max_sequence_length = 100
X_padded = pad_sequences(X_seq, maxlen=max_sequence_length)

# 데이터 셔플링 및 분할
X_train, X_temp, y_train, y_temp = train_test_split(X_padded, Y_extended, test_size=0.4, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

# 모델 정의
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length))
model.add(GRU(128, return_sequences=True))
model.add(GRU(128))
model.add(Dense(5, activation='softmax'))  # 클래스가 5개

# 모델 구조 출력
model.summary()

"""## 6-1. Epoch by Adam"""

# 모델 컴파일 및 학습 adam
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# 평가 (테스트 및 검증 데이터셋)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

import matplotlib.pyplot as plt

# 훈련 및 검증 데이터에 대한 정확도와 손실 추출
acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']
loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(1, len(acc) + 1)

# 정확도 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 예측
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 혼동 행렬 생성
cm = confusion_matrix(y_test, y_pred_classes)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 분류 보고서 출력 (Precision, Recall, F1-Score)
print(classification_report(y_test, y_pred_classes))

# """## 6-2. Epoch by Nadam"""

# # 모델 컴파일 및 학습 Nadam
# model.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# # 평가 (테스트 및 검증 데이터셋)
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# val_loss, val_accuracy = model.evaluate(X_val, y_val)
# print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# import matplotlib.pyplot as plt

# # 훈련 및 검증 데이터에 대한 정확도와 손실 추출
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# # 정확도 그래프
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# # 손실 그래프
# plt.subplot(1, 2, 2)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 모델 예측
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # 혼동 행렬 생성
# cm = confusion_matrix(y_test, y_pred_classes)

# # 혼동 행렬 시각화
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

# # 분류 보고서 출력 (Precision, Recall, F1-Score)
# print(classification_report(y_test, y_pred_classes))

# """## 6-3. Epoch by RMSProp"""

# # 모델 컴파일 및 학습 RMSProp
# model.compile(optimizer='RMSProp', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history1 = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# # 평가 (테스트 및 검증 데이터셋)
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# val_loss, val_accuracy = model.evaluate(X_val, y_val)
# print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# import matplotlib.pyplot as plt

# # 훈련 및 검증 데이터에 대한 정확도와 손실 추출
# acc = history1.history['accuracy']
# val_acc = history1.history['val_accuracy']
# loss = history1.history['loss']
# val_loss = history1.history['val_loss']

# epochs = range(1, len(acc) + 1)

# # 정확도 그래프
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# # 손실 그래프
# plt.subplot(1, 2, 2)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 모델 예측
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # 혼동 행렬 생성
# cm = confusion_matrix(y_test, y_pred_classes)

# # 혼동 행렬 시각화
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

# # 분류 보고서 출력 (Precision, Recall, F1-Score)
# print(classification_report(y_test, y_pred_classes))

