import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

# 크롤링
result = pd.DataFrame()

for i in range(595226, 603465):  # 595226~603464번 글이 2021년에 작성된 글
    URL = "http://www1.president.go.kr/petitions/" + str(i)

    response = requests.get(URL)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    title = soup.find('h3', class_='petitionsView_title')
    count = soup.find('span', class_='counter')

    for content in soup.select('div.petitionsView_write > div.View_write'):
        content

    a = []
    for tag in soup.select('ul.petitionsView_info_list > li'):
        a.append(tag.contents[1])

    if len(a) != 0:
        df1 = pd.DataFrame({'start': [a[1]],
                            'end': [a[2]],
                            'category': [a[0]],
                            'count': [count.text],
                            'title': [title.text],
                            'content': [content.text.strip()[0:13000]]
                            })

        result = pd.concat([result, df1])
        result.index = np.arange(len(result))

    if i % 60 == 0:
        print("Sleep 90seconds. Count:" + str(i)
              + ",  Local Time:" + time.strftime('%Y-%m-%d', time.localtime(time.time()))
              + " " + time.strftime('%X', time.localtime(time.time()))
              + ",  Data Length:" + str(len(result)))
        time.sleep(90)

# 크롤링 데이터 확인 및 엑셀파일로 저장
print(result.shape)
df = result
df.head()
df.to_csv('data/crawling.csv', index = False, encoding = 'utf-8-sig')