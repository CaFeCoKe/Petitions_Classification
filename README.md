# Petitions_Classification
프로젝트의 목표는 '주목받을만한 청원 분류하는 모델만들기'이다. 높은 청원 참여인원을 받은 글들의 특징을 학습하여 새 글이 입력되었을 때 유사성을 계산하여 판단하는 모델을 만들어 보며 토크나이징, 임베딩등 NLP에 필요한 기술중 일부를 사용하고, 텍스트 데이터를 CNN에 적용한 TextCNN을 사용하게 된다. 
 
https://user-images.githubusercontent.com/86700191/154801568-3fa87c0f-ca8c-41c7-9535-8ce106298ff0.mp4

## 1. 사용 라이브러리
- Pytorch (torch) : TextCNN 모델 설계
- Beautiful Soup, request : 크롤링
- Pandas, numpy : 크롤링 데이터 만들기
- KoNLPy : 한국어 자연어 처리(토크나이징)
- gensim : 단어 임베딩
- torchtext, re : 데이터셋, 단어장 만들기

## 2. 알고리즘 순서도
![Petitions_Classication](https://user-images.githubusercontent.com/86700191/154838789-748d213b-f8c9-40fc-95bd-4edc9a5fcfa3.png)

## 3. 네트워크 구성도
![TextCNN](https://user-images.githubusercontent.com/86700191/155136244-ce4b1661-966d-4e5e-bfb7-a8f1e66262bc.png)

## 4. 결과
- 원본 데이터 및 토크나이징된 데이터
![1](https://user-images.githubusercontent.com/86700191/152671181-8f4b42a1-00d8-4754-b3bf-8f8265f5098f.PNG)
- 임베딩한 결과 및 테스트(중심단어=코로나)
![2](https://user-images.githubusercontent.com/86700191/152723997-e18c50e3-6c7e-4b08-9fc8-da685172835d.PNG)
- 손실값과 정확도 (Overfitting 구간 확인)
![overfiiting](https://user-images.githubusercontent.com/86700191/154800955-ad1e5b9c-287e-46e0-a7e7-4be90f1da4bc.PNG)

## 5. 유의점
- KoNLPy의 Class는 Java기반으로 되어있기 때문에 install전 ver.1.7.0 이상의 JDK, ver.0.5.7 이상의 JPype1이 설치 되어 있어야하며, JAVA_HOME을 환경변수로 설정 해놓아야 한다. (공식 사이트 참고)
- [JPype1 설치](https://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype)시 OS의 비트에 맞게 설치하야하며, Python의 버전과도 동일해야 설치가 가능하다. (cp36 = Python 3.6)
- 크롤링시 연속으로 페이지에 접근할 때 접근이 불가능한 페이지가 존재 할수 있으니 대기시간이 필요하다. 대기시간을 너무 짧게 하면 데이터 추출이 제대로 이루어지지 않는다.
    - 60페이지 크롤링시 대기시간 20초로 진행시 약 8000개의 데이터를 668개밖에 추출하지 못한 결과
    ![캡처](https://user-images.githubusercontent.com/86700191/152285591-b26b7f83-58bd-4fc9-95e3-f11ff2030a32.PNG)
    - 한 페이지당 대기시간 1초로 진행시 비교적 정확하게 가져온 결과
    ![abcd](https://user-images.githubusercontent.com/86700191/152335229-cf4ac49b-a467-4f66-aac8-7e479f54dcd3.PNG)
- torchtext 모듈 import 문제 (legacy components)
![torchtext](https://user-images.githubusercontent.com/86700191/152776011-4090c9ea-c6bc-46eb-8fd2-236c7865a668.PNG)

 22/03/15 추가) torchtext를 업그레이드를 하여 v0.12.0으로 한다면 v0.9.0부터 legacy로 지원해주었던 마이그레이션 클래스들이 지원이 안될 것이다.
![legacy](https://user-images.githubusercontent.com/86700191/158297203-bb789adb-664d-4af7-90d9-e4674a80e956.PNG)

  - 해결방법 1. torchtext를 v0.9.0에서 v0.11.0까지 어느 버전이든 다운그레이드하여 사용한다.
  
  - 해결방법 2. [torchtext 공식 마이그레이션 튜토리얼](https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb) 을 보고 새로 공부하여 적용한다.
## 6. 참고자료(사이트)
- [PyTorch 공식 설명](https://pytorch.org/docs/stable/index.html)
- [Pandas 공식 설명](https://pandas.pydata.org/docs/reference/index.html)
- [KoNLPy 공식 사이트](https://konlpy.org/ko/latest/)
- [Beautiful Soup 공식 문서](https://beautiful-soup-4.readthedocs.io/en/latest/)
- [Base Code](https://github.com/bjpublic/DeepLearningProject)
- [Word2Vec의 학습방식(CBOW vs Skip-gram)](https://wikidocs.net/22660)
- [Yoon Kim의 Convolutional Neural Networks for Sentence Classification 논문](https://arxiv.org/abs/1408.5882v2)
- [Denny Britz의 TextCNN Tensorflow 구현](https://github.com/dennybritz/cnn-text-classification-tf)
