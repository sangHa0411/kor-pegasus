# kor-pegasus

## 1. Goal
  1. 한국어로 된 Pegasus 모델 개발 (제작 중)
  2. Pegasus 모델의 학습 과정 이해

## 2. Model Structure
  1. Encoder - Decoder Model
  2. 기존의 Document에서 k% 비율의 문장 만큼을 가져와서 Target Text로 설정한다.
     * 문장이 빠진 자리에는 MASK1 토큰을 삽입한다.
  3. 입력 문장의 전체 토큰의 갯수의 n% 만큼의 토큰에 MASK2 토큰으로 Masking 처리를 한다. 
  4. Decoder의 출력 문장만 을 ground truth와 비교하는 것 이외에도 Encoder의 출력에서 Masking 된 토큰을 찾는 방법으로 학습을 진행한다.
     1. Gap Sentence Generation : GSG
     2. Masked Language Modeling : MLM
     
  ![ksnip_20220120-144831](https://user-images.githubusercontent.com/48673702/150281114-8934accd-622a-4892-a738-abf67545560b.png)

## 3. Environment
   1. Google Colab Pro Plus
   2. A100 GPU

## 4. Data
   1. 모두의 말뭉치 신문 데이터
   2. 크기 : 20GB
   3. 주소 : https://corpus.korean.go.kr/
  
## 5. Training
   1. Batch Size : 8
   2. Gradient Accumulation Sentence : 16
   3. Total Batch Size : 128
   4. Warmup Steps : 20000
   5. Epochs : 10
   6. Weight Decay : 1e-2

## 6. Model Size : Base
   1. Encoder & Decoder Layers : 12
   2. Hidden Size : 768
   3. Feedforward Size : 3072
   4. Attention Head Size : 12

## 7. Code
```python

from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('sh110495/kor-pegasus')

config = AutoConfig.from_pretrained('sh110495/kor-pegasus')
model = AutoModelForSeq2SeqLM.from_pretrained('sh110495/kor-pegasus', config=config)

```
