import os
import time

import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

import re
from konlpy.tag import Okt

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from numpy.random import RandomState

import torchtext
from torchtext.legacy.data import Field
from torchtext.legacy.data import TabularDataset

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
LABEL = Field(sequential = False)   # Output (LABEL = label)

# 데이터셋 생성 (파일에서 읽어옴)
train, validation = TabularDataset.splits(
    path='./', train='train.csv', validation='test.csv', format='csv',
    fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

# 테스트
print("Train:", train[0].text,  train[0].label)
print("Validation:", validation[0].text, validation[0].label)
