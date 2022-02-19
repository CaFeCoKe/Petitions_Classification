import os
import time

import pandas as pd
import numpy as np
from numpy.random import RandomState

import requests
from bs4 import BeautifulSoup

import re
from konlpy.tag import Okt

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import torch
import torchtext
from torchtext.legacy.data import Field
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import BucketIterator
from torchtext.vocab import Vectors

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

# 크롤링한 데이터가 존재하지 않을 시에 크롤링 진행
if not os.path.exists('./crawling.csv'):
    result = pd.DataFrame()
    for i in range(595226, 603465):  # 595226~603464번 글이 2021년에 작성된 글
        URL = "http://www1.president.go.kr/petitions/" + str(i)

        response = requests.get(URL)    # get 요청에 대한 응답저장 (= 페이지 접속)
        html = response.text        # 접속 페이지의 HTML코드 저장
        time.sleep(1)       # 요청 시간 딜레이 (접근 실패 방지)

        soup = BeautifulSoup(html, 'html.parser')       # parser를 이용해 HTML코드를 객체 구조화 (parshing)
        title = soup.find('h3', class_='petitionsView_title')   # 청원제목: h3 태그, class petitionsView_title에 해당
        count = soup.find('span', class_='counter')     # 청원 참여인원: span 태그, class counter에 해당

        for content in soup.select('div.petitionsView_write > div.View_write'):     # petitionsView_write 하위의 View_write에서 데이터 추출
            content

        a = []
        for tag in soup.select('ul.petitionsView_info_list > li'):      # ul 태그 petitionsView_info_list 하위의 리스트 속성값 추출
            a.append(tag.contents[1])       # [<p>abc</p>, 'abcd'] 중 두번째 내용만 추출

        # 추출한 데이터 데이터프레임으로 변환
        if len(a) != 0:
            df1 = pd.DataFrame({'start': [a[1]],
                                'end': [a[2]],
                                'category': [a[0]],
                                'count': [count.text],
                                'title': [title.text],
                                'content': [content.text.strip()[0:13000]]  # 글 앞뒤 공백 제거후 데이터길이 13000자 제한(엑셀의 셀 내 글자 수 32767자 제한)
                                })

            result = pd.concat([result, df1])   # result 데이터프레임에 현재 추출한 데이터 누적 병합 (세로)
            result.index = np.arange(len(result))   # 데이터 프레임 안댁스설정

        if i % 60 == 0:
            print("Count:" + str(i)
                  + ",  Local Time:" + time.strftime('%Y-%m-%d', time.localtime(time.time()))
                  + " " + time.strftime('%X', time.localtime(time.time()))
                  + ",  Data Length:" + str(len(result)))

    # 크롤링 데이터 확인 및 csv파일로 저장
    print(result.shape)
    result.to_csv('./crawling.csv', index=False, encoding='utf-8-sig')


# 데이터 전처리
df = pd.read_csv('./crawling.csv')      # csv파일에서 데이터 불러오기 (시간절약)


# 공백문자 공백으로 치환
def remove_white_space(text):
    text = re.sub(r'[\t\r\n\f\v]', ' ', str(text))
    return text


# 특수문자 공백으로 치환
def remove_special_char(text):
    text = re.sub('[^ ㄱ-ㅣ가-힣 0-9]+', ' ', str(text))
    return text

# 청원 제목에 공백문자, 특수문자 제거
df.title = df.title.apply(remove_white_space)
df.title = df.title.apply(remove_special_char)

# 청원 내용에 공백문자, 특수문자 제거
df.content = df.content.apply(remove_white_space)
df.content = df.content.apply(remove_special_char)


# 토크나이징(Tokenizing)
okt = Okt()

df['title_token'] = df.title.apply(okt.morphs)      # 청원제목을 형태소(morphs) 단위로 토크나이징
df['content_token'] = df.content.apply(okt.nouns)       # 청원내용을 명사(nouns) 단위로 토크나이징

df['token_final'] = df.title_token + df.content_token   # 토큰화된 제목과 내용을 합쳐서 저장
df['count'] = df['count'].replace({',' : ''}, regex=True).apply(lambda x : int(x))       # 참여인원의 단위에 쉼표를 제거하고 object형에서 int형 반환

df['label'] = df['count'].apply(lambda x: 'Yes' if x >= 1000 else 'No')       # 참여인원이 1000 이상이면 yes, 아니면 no로 label에 저장
df_drop = df[['token_final', 'label']]      # token_final과 label만 분석에 필요

df_drop.to_csv('./Tokenizing.csv', index=False, encoding='utf-8-sig')   # 토크나이징된 데이터 저장


# 단어 임베딩
# Skip-Gram 방식, 임베딩 벡터 크기는 100, 문맥파악을 위한 앞, 뒤 토큰수 2, 단어 최소 빈도 수를 1회로 제한
embedding_model = Word2Vec(df_drop['token_final'],
                           sg=1, vector_size=100, window=2, min_count=1, workers=4)

print(embedding_model)

embedding_model.wv.save_word2vec_format('./petitions_tokens_w2v')    # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format('./petitions_tokens_w2v')   # 모델 로드


# 데이터셋 설계
# 데이터셋 분할및 저장
rng = RandomState()

# 훈련데이터, 테스트데이터 분할(8:2)
train = df_drop.sample(frac=0.8, random_state=rng)
test = df_drop.loc[~df_drop.index.isin(train.index)]

# 훈련데이터, 테스트데이터 저장
train.to_csv('./train.csv', index=False, encoding='utf-8-sig')
test.to_csv('./test.csv', index=False, encoding='utf-8-sig')


# 토크나이징
def tokenizer(text):
    text = re.sub('[\[\]\']', '', str(text))    # ( ['a', 'b']  -> a, b )
    text = text.split(', ')     # ,를 기준으로 각 토큰 분리
    return text


TEXT = Field(tokenize=tokenizer)    # Input (TEXT = token_final)
LABEL = Field(sequential=False)   # Output (LABEL = label)

# 데이터셋 생성 (파일에서 읽어옴)
train, validation = TabularDataset.splits(
    path='./', train='train.csv', validation='test.csv', format='csv',
    fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

# 단어장 만들기
vectors = Vectors(name="./petitions_tokens_w2v")

# 훈련데이터의 단어장을 생성 (임베딩 벡터 값을 저장된 Word2Vec 모델로 초기화)
TEXT.build_vocab(train, vectors=vectors, min_freq=1, max_size=None)
LABEL.build_vocab(train)

vocab = TEXT.vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 배치 사이즈를 8로 하여 훈련, 평가(테스트)데이터의 Batch Data 생성
train_iter, validation_iter = BucketIterator.splits(
        datasets=(train, validation), batch_size=8, device=device, sort=False)


# TextCNN 설계
class TextCNN(nn.Module):
    def __init__(self, vocab_built, emb_dim, dim_channel, kernel_wins, num_class):
        super(TextCNN, self).__init__()

        self.embed = nn.Embedding(len(vocab_built), emb_dim)    # vocab size * emd dimension
        self.embed.weight.data.copy_(vocab_built.vectors)   # Word2Vec으로 학습한 데이터 가져오기

        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])      # 임베딩 결과를 Conv2d를 통해 필터 생성
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)      # 오버피팅 방지용 드롭아웃 40퍼 설정
        self.fc = nn.Linear(len(kernel_wins) * dim_channel, num_class)

    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)      # 첫번째 축에 차원 추가(텍스트데이터(2D) -> 이미지화(3D))

        con_x = [self.relu(conv(emb_x)) for conv in self.convs]     # 리스트 형태의 필터를 각각 통과하여 feature map 생성 후 relu를 통해 activation map 생성

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]      # Max Pooling 진행

        fc_x = torch.cat(pool_x, dim=1)     # 1차원 Pooling 벡터를 concat하여 하나의 fc 생성
        fc_x = fc_x.squeeze(-1)     # 차원 줄이기
        fc_x = self.dropout(fc_x)       # 드롭아웃 적용

        logit = self.fc(fc_x)

        return logit


# 훈련 데이터로 학습하여 모델화
def train(model, device, train_itr, optimizer):
    model.train()       # Train model로 변경
    corrects, train_loss = 0.0, 0

    for batch in train_itr:
        text, target = batch.text, batch.label
        text = torch.transpose(text, 0, 1)      # 역행렬로 변환
        target.data.sub_(1)     # 타겟의 각 값을 1씩 줄이기
        text, target = text.to(device), target.to(device)

        optimizer.zero_grad()       # optimizer 초기화
        output = model(text)
        loss = F.cross_entropy(output, target)      # Loss 함수로 교차 엔트로피 사용 (Softmax로 Yes/No 분류 + Negative Log Loss 계산)
        loss.backward()      # 역전파로 Gradient를 계산 후 파라미터에 할당
        optimizer.step()      # 파라미터 업데이트

        train_loss += loss.item()       # Loss 값 누적
        result = torch.max(output, 1)[1]     # 인덱스별로 계산된 출력 값중 가장 큰 클래스 저장
        corrects += (result.view(target.size()).data == target.data).sum()      # 예측값과 레이블 데이터를 비교하여 맞는 것은 누적

    train_loss /= len(train_itr.dataset)        # Loss 값을 Batch 값으로 나누어 미니 배치마다의 Loss 값의 평균을 구함
    accuracy = 100.0 * corrects / len(train_itr.dataset)        # 정확도 값을 Batch 값으로 나누어 미니 배치마다의 정확도 평균을 구함

    return train_loss, accuracy


# 모델 평가 함수
def evaluate(model, device, itr):
    model.eval()
    corrects, val_loss = 0.0, 0

    for batch in itr:
        text = batch.text
        target = batch.label
        text = torch.transpose(text, 0, 1)
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)

        output = model(text)
        loss = F.cross_entropy(output, target)

        val_loss += loss.item()
        result = torch.max(output, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    val_loss /= len(itr.dataset)
    accuracy = 100.0 * corrects / len(itr.dataset)

    return val_loss, accuracy


# 모델 학습 실행
model = TextCNN(vocab, 100, 10, [3, 4, 5], 2).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.005)        # adam optimizer 사용, 학습률은 0.005

best_test_acc = 0.0

"""
loss_tr = []
acc_tr = []
loss_val = []
acc_val = []
"""

for epoch in range(1, 10 + 1):       # epoch 10번 실행

    tr_loss, tr_acc = train(model, device, train_iter, optimizer)
    print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))

    val_loss, val_acc = evaluate(model, device, validation_iter)
    print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, val_loss, val_acc))

    if val_acc > best_test_acc:     # 현 epoch의 정확도가 더 높을 시 갱신
        best_test_acc = val_acc
        print("model saves at {} accuracy".format(best_test_acc))

        torch.save(model.state_dict(), "TextCNN_Best_Validation.pt")       # 정확도 갱신시 현 모델 저장 및 갱신

    print('-----------------------------------------------------------------------------')

    """
    # overfitting 확인하기 위한 플롯 그리기
    loss_tr.append(tr_loss)
    loss_val.append(val_loss)
    acc_tr.append(tr_acc)
    acc_val.append(val_acc)

np1 = np.array(loss_tr)
np2 = np.array(loss_val)
np3 = np.array(acc_tr)
np4 = np.array(acc_val)

plt.subplot(2, 1, 1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(np1, label='Loss of train')
plt.plot(np2, label='Loss of Validation')
plt.legend()  # 라벨표시를 위한 범례

plt.subplot(2, 1, 2)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.plot(np3, label='acc of train')
plt.plot(np4, label='acc of validation')
plt.legend()  # 라벨표시를 위한 범례

plt.show()
"""