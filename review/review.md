# Pegasus Paper 분석


## 1. Abstract
  1. 최근 Transformers 기반의 모델은 대용량의 데이터로 사전학습을 통해서 요약 과제를 포함한 다양한 task에서 성공을 이루었다.
  2. 하지만 문서 요약 과제에 최적화된 방법에 대한 연구는 많이 이루어지지 않았다.
  3. 이 연구를 통해서 encoder-decoder 구조의 transformer 기반의 모델의 새로운 학습 방법을 제시한다.


## 2. Introduction
  1. 문서 요약의 목표는 입력 문서에서 정확하고 간결한 요약문을 생성하는 것이다.
  2. 단순히 문장 혹은 단어를 추출해서 연결하는 추출요약과는 다르게 생성요약은 단어를 새로 생성하면서 요약문을 만든다.
  3. 이 연구를 통해서 생성 요약을 위한 사전학습 목표 (pre-training objectives)를 제안한다.
      * 문장 전체를 masking을 하여서 나머지 문서를 통해서 이 문장을 생성한다. (Gap Sentence Generation)
      * 단순히 문장을 무작위로 선정하는 것보다 문서 내에서 중요한 문장을 선택하는 것이 높은 성능을 이끌 수 있었다.

## 3. Pre-training Objectives
  1. Gap Sentence Generation
      1. 계기 : 최대한 downstream task에 유사하게 사전학습 목표를 설정하는 것이 높은 성능을 가질 것이다.
      2. 방법 : 입력 문서를 통해서 그와 관련하고 요약문과 같은 문장을 생성한다. 
      3. 특징 
            * 단순히 입력 문장의 단어 및 구문을 이어 붙이는 것이 아니기 때문에 추출 요약 데이터 가지고서 사전 학습하는 것과는 다르다.
      4. 과정
            1. 문장들을 선택해서 이 문장들을 전체 [mask1] 처리를 한다.
            2. 이 문장들을 이어서 유사 요약문을 만든다.
            3. 문장들을 선택할 때 여러 방법으로 선택의 기준을 정할 수 있다.
                * Random : 무작위로 문장들을 선택
                * Lead : 시작 부분의 문장들을 선택
                * Principal 
                  1. 중요도에 따라서 문장들을 선택
                  2. 해당 문장과 그 문장을 제외한 문서의 나머지 부분 사이에서 ROUGE1-F1 점수를 계산한다
                      1. 각각으 독립적이 문장으로 계산을 할지(Ind) 아니면 연속된 문장으로 점수르 계산 할 지에 대해서(Seq)로 구분한다.
                      2. ROUGE-F1 점수를 계산하 때 원래 방법댈 계산을 할 지(Orig), n-gram을 하나으 집합으로 생각하여 계산을 할 지(Uniq)로 구분한다.
                      3. 따라서 총 4개의 방법으로 문장의 선택 방법이 존재한다.

                  3. 문장 선택에 대한 예시
                      * ![스크린샷 2022-03-13 오전 10 58 27](https://user-images.githubusercontent.com/48673702/158041547-e3259477-4b0e-4829-8e3b-4dca984c621b.png)
                      *  하나의 문서에서 gap sentence를 어떻게 선정할 지에 대한 예시
                          1. Red : Leed (문서의 처음 부분을 gap sentence로 선정)
                          2. Green : Random (문서에서 무작위로 문장을 선정해서 gap-setence를 선정)
                          3. Blue : Ind-Orig (문서에서 각각의 문장을 rouge-f1 score를 계산하는 방법으로 나머지 문서와 점수를 계산하고 gap-sentence를 선정)
        

  2. Masked Language Model
      1. 방향 : BERT 논문과 유사하게 입력 문서에서 전체 토큰의 15%정도 되는 토큰을 선정한다.
          * 80% : [MASK2] token으로 변환  
          * 10% : random token으로 변환한다.
          * 10% : 변환하지 말고 그대로 둔다.  
      2. 논문에서의 활용 방향
          * GSG & MLM을 같이 활용해서 모델 학습을 진행 그러나 PEGASUS Large 모델을 학습할 때는 (많은 수의 pretraining steps를 학습) 할 때는 도움을 주지 못한 것으로 실험 결과가 나왔다.
          * PEGASUS Large Model을 학습할 때는 GSG만을 사용해서 모델을 학습하였다.

## 4. Model Structure  
  ![ksnip_20220120-144831](https://user-images.githubusercontent.com/48673702/150281114-8934accd-622a-4892-a738-abf67545560b.png)

## 5. Pre-training Corpus
  1. C4 : 750G 정도의 웹 페이지 데이터
  2. HugeNews : 3.8TB 정도의 뉴스, 블로그, 신문 및 크롤링 된 웹페이지 데이터
  
## 6. Downstream Tasks/Datasets
  1. XSum : BBC 뉴스 데이터
  2. CNN/Daily Mails : CNN, Daily 뉴스 데이터
  3. NEWSROOM : 작가 혹은 에디터가 작성한 기사 요약 데이터셋
  4. Gigaword : 뉴스 기사를 통해서 헤드라인을 생성하는 데이터셋
  5. arXiv : 과학 publication의 body를 기반으로 abstract를 생성하는 데이터셋
  6. 그 외 다양하게 존재

## 7. Model size
  1. PEGASUS - Base
      * Layer : 12
      * Hidden size : 768
      * Feedforward layer size : 3072
      * Attention Head map : 12
  2. PEGASUS - Large
      * Layer : 16
      * Hidden size : 1024
      * Feedforward size : 4096
      * Attention Head map : 16

 ## 8. Ablations on PEGASUS-base
  1. Pre-training Corpus 실험 결과
      1. C4 데이터셋을 학습한 모델보다 HugeNews를 학습한 모델이 XSum, CNN/DailyMail (뉴스 기사 요약 데이터셋) 에서 높은 성능을 가지는 것을 확인할 수 있었다.
      2. C4 데이터셋을 학습한 모델이 HugeNews를 학습한 모델보다 WikiHow, Reddit TIFU 데이터셋(non-news dataset)에서 높은 점수를 가지는 것을 확인할 수 있었다.
      3. image
        ![스크린샷 2022-03-13 오전 11 19 53](https://user-images.githubusercontent.com/48673702/158042067-a777bf1f-603b-49bf-b6c8-6c46c6adfbe0.png)

  2. GSR 실험 결과
      1. 어떠한 방식으로 gap sentence들을 선정하는 것이 높은 성능을 가지는가에 대한 실험
      2. PEGASUS base Model & C4 pretraining dataset 기준
      3. 대체적으로 Ind-Orig 방법이 높은 성능을 가지는 것을 확인할 수 있었다.
      4. News Dataset에는 Lead 방법이 높은 성능을 가졌지만 WikiHow, Reddit (non-news dataset)에는 낮은 성능을 가지는 것을 확인할 수 있었다.
      5. image
        ![스크린샷 2022-03-13 오전 11 32 45](https://user-images.githubusercontent.com/48673702/158042362-2536b6bd-a4a4-4141-ac60-bdfdf40234a6.png)

  3. Gap sentence의 비율에 대한 실험
      1. 대체적으로 15% ~ 50%가 높은 성능을 가지는 것을 확인할 수 있다.
      2. 각 Downstream Task에 따라서 최적의 비율이 다르다.
      3. image
        ![스크린샷 2022-03-13 오전 11 37 38](https://user-images.githubusercontent.com/48673702/158042483-ef03ecb0-b5a6-49a9-90a0-ba305fbda4be.png)  

  4. MLM 
      1. Gap Sentence를 30%로 선정하고 Masked token을 15%로 선정해서 모델을 학습
      2. MLM & Ind-Orig 방법이 random과 비슷한 성능을 가지는 것을 확인할 수 있다.
      3. 경험적으로 MLM이 pre-training 초기 checkpoint(100k ~ 200k)을 fine-tuning을 할 때 성능을 높여주는 것을 확인할 수 있었다.
      4. 하지만 pre-training step(500k)이 많아질 수록 그 효과가 적어지는 것을 확인
      5. 논문에서는 결과적으로 PEGASUS large 모델은 MLM을 사용하지 않았다.

## 9. 실험 결과
  1. Downstream tasks
      1. 대부분의 과제에 대해서 PEGASUS-base 모델이 이전의 SOTA 보다 높은 성능을 가지는 것을 확인할 수 있었다.
      2. PEGASUS-large 모델이 PEGASUS-base 모델보다 더 높은 성능을 가졌다.
     
  2. image
      ![스크린샷 2022-03-13 오전 11 54 35](https://user-images.githubusercontent.com/48673702/158042945-16c833a4-b2d7-4770-bb0e-0e7638d25f4f.png)  


## 10. Hayperparameter
  ![스크린샷 2022-03-13 오후 12 00 23](https://user-images.githubusercontent.com/48673702/158043095-9591f70d-a0b6-4d72-98e5-a00105b9617a.png)

