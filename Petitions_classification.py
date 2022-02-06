import os
import time

import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

import re

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

# csv파일에서 데이터 불러오기
df = pd.read_csv('./crawling.csv')
# print(df.head())

# 데이터 전처리
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