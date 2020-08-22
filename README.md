# HAAFOR challenge 2020 

두 개의 뉴스 데이터가 주어졌을 때, 두 뉴스 사이의 전후 관계를 판단하는 챌린지.

## 데이터 전처리 

매 전/후 뉴스 쌍마다 0, 1 중 랜덤하게 숫자를 생성하여 0이면 '후|전', 1이면 '전|후' 형태로 모델의 입력을 생성하였다. 이때 제목은 포함시키지 않았는데, 제목을 넣었을 때 모델의 성능이 특별히 없었으며 또한 제목을 포함할 시 모델의 입력 최대 길이를 넘어가는 경우가 자주 생기기 때문이었다.

## 모델 구조 

Huggingface의 사전학습된 ALBERT 모델('albert-base-v2')을 기반으로, 위에 4-layer bidirectional GRU와 선형 레이어를 쌓아서 모델을 생성했다. 구체적으로, 입력 문장을 ALBERT 모델에 통과시켜 얻은 final layer output을 4-layer bidirectional GRU 모델에 통과시킨 후, dropout과 선형 레이어에 통과시킨 결과를 시그모이드 함수를 통과시켜 최종적으로 0과 1 사이의 값을 얻는 모델이다. Threshold는 따로 변경하지 않고 0.5를 기준으로 사용하였다.

## Requirements

```
numpy==1.18.1
pandas==1.0.4
torch==1.3.1
tqdm==4.46.1
transformers==3.0.2
```

## Usage

* Clone this repository.

* Unzip given data files into Data foler.

* Training
    ```
    python train.py
    ```

* Evaluation
    1. Unzip weight file using the script.
    ```
    cd weights
    cat weight.tar* | tar xvf -
    ```
    2. Evaluate the evaluation.csv file
    ```
    python evaluate.py
    ```
    3. After the evaluation, the result file is located at `Data/answer.csv`.
