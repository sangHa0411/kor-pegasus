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
                    2. 해당 문장과 그 문장을 제외한 문서의 나머지 부분 사이에서 ROUGE1-F1 점수를 계산한다.


  2. Masked Language Model
      1. 방향 : BERT 논문과 유사하게 입력 문서에서 전체 토큰의 15%정도 되는 토큰을 선정한다.
          * 80% : [MASK2] token으로 변환  
          * 10% : random token으로 변환한다.
          * 10% : 변환하지 말고 그대로 둔다.  


## 4. Model Structure  
  ![ksnip_20220120-144831](https://user-images.githubusercontent.com/48673702/150281114-8934accd-622a-4892-a738-abf67545560b.png)

## 5. Model size & hyperparameters
  1. PEGASUS - Base
      * Layer : 12
      * Hidden size : 768
      * Feedforward layer size : 3072
      * Attention Head map : 12
      * Train Batch size : 256
  2. PEGASUS - Large
      * Layer : 16
      * Hidden size : 1024
      * Feedforward size : 4096
      * Attention Head map : 16
      * Train Batch Size : 8192
  3. Hyperparameter
      * Optimizer : Adafactor
      * Square root learning rate decay
      * Dropout : 0.1

 