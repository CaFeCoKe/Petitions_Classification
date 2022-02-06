# Petitions_Classification
 Using PyTorch-TextCNN

## 1. 사용 라이브러리
- Pytorch : TextCNN 모델 설계
- Pandas, numpy : 데이터셋 만들기
- Beautiful Soup, request : 크롤링
- KoNLPy : 한국어 자연어 처리(토크나이징)
- gensim : 단어 임베딩

## 2. 알고리즘 순서도

## 3. 결과
- 원본 데이터 및 토크나이징된 데이터
![1](https://user-images.githubusercontent.com/86700191/152671181-8f4b42a1-00d8-4754-b3bf-8f8265f5098f.PNG)


## 4. 유의점
- KoNLPy의 Class는 Java기반으로 되어있기 때문에 install전 ver.1.7.0 이상의 JDK, ver.0.5.7 이상의 JPype1이 설치 되어 있어야하며, JAVA_HOME을 환경변수로 설정 해놓아야 한다. (공식 사이트 참고)
- [JPype1 설치](https://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype)시 OS의 비트에 맞게 설치하야하며, Python의 버전과도 동일해야 설치가 가능하다. (cp36 = Python 3.6)
- 크롤링시 연속으로 페이지에 접근할 때 접근이 불가능한 페이지가 존재 할수 있으니 대기시간이 필요하다. 대기시간을 너무 짧게 하면 데이터 추출이 제대로 이루어지지 않는다.
    - 대기시간 20초로 진행시 약 8000개의 데이터를 668개밖에 추출하지 못한 결과
    ![캡처](https://user-images.githubusercontent.com/86700191/152285591-b26b7f83-58bd-4fc9-95e3-f11ff2030a32.PNG)
    - 한 페이지당 대기시간 1초로 진행시 비교적 정확하게 가져온 결과
    ![abcd](https://user-images.githubusercontent.com/86700191/152335229-cf4ac49b-a467-4f66-aac8-7e479f54dcd3.PNG)

## 5. 참고자료(사이트)
- [PyTorch 공식 설명](https://pytorch.org/docs/stable/index.html)
- [Pandas 공식 설명](https://pandas.pydata.org/docs/reference/index.html)
- [KoNLPy 공식 사이트](https://konlpy.org/ko/latest/)
- [Beautiful Soup 공식 문서](https://beautiful-soup-4.readthedocs.io/en/latest/)
- [Base Code](https://github.com/bjpublic/DeepLearningProject)
