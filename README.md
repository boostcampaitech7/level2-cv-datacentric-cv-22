# **다국어 영수증 OCR**

<p align="center">
<img width="1229" alt="image" src="https://github.com/user-attachments/assets/9fe28d3b-abe1-4b77-8c7b-8e79b17525e8" width="90%" height="90%"/>
</p>

## 1. Competiton Info

  Data-Centric AI는 Model-Centric AI의 반대 개념으로, 성능 향상 및 최적화를 위해 데이터의 수집, 관리, 분석 등의 기법을 통해 데이터의 품질과 가치를 최대화하는 것이다. 본 대회는 이러한 관점에서 **모델 성능은 고정하고, 데이터 보완, 전처리와 증강을 통해서만 성능을 향상시키는 방식으로 진행**되었다. 베이스라인은 사전 학습된 VGG-16을 backbone으로 하는 EAST 모델을 사용하였다.

- **데이터셋 구성** : 중국어, 일본어, 태국어, 베트남어로 된 영수증 이미지
    - 각 언어당 (train) 100장씩 총 400장, (test) 30장씩 총 120장

### Timeline

- 2024.10.28 ~ 2024.11.07

### Evaluation

- 평가지표: F1 score
- Ground Truth 박스와 Prediction 박스간 IoU(Intersection Over Union, Detector의 정확도를 평가하는 지표)가 50이 넘는 예측에 대해 True로 판단한다

## 2. Team Info

### MEMBERS

| <img src="https://github.com/user-attachments/assets/812313de-7fb9-462d-b749-c552b1d38411" width="200" height="150"/> | <img src="https://github.com/user-attachments/assets/812313de-7fb9-462d-b749-c552b1d38411" width="200" height="150"/> | <img src="https://github.com/user-attachments/assets/6e38e938-52d6-49ea-9113-750c47cd6f61" width="200" height="150"/> | <img src="https://github.com/user-attachments/assets/812313de-7fb9-462d-b749-c552b1d38411" width="200" height="150"/> | <img src="https://github.com/user-attachments/assets/6e38e938-52d6-49ea-9113-750c47cd6f61" width="200" height="150"/> |
| --- | --- | --- | --- | --- |
| [김예진](https://github.com/yeyechu) | [배형준](https://github.com/BaeHyungJoon) | [송재현](https://github.com/mongsam2) | [이재효](https://github.com/jxxhyo) | [차성연](https://github.com/MICHAA4) |

### Project Objective and Direction

- Data-Centric에 집중하기

### Team Component

- **EDA와 데이터 전처리** : 김예진, 배형준, 송재현, 이재효, 차성연 / **데이터 수집** : 이재효
- **데이터 증강** : 김예진, 배형준, 이재효

## 3. Data EDA

<img width="798" alt="image" src="https://github.com/user-attachments/assets/b81189ac-9601-4c04-93d1-4fd0a858f4fd">

- 입력 이미지는 다양한 크기로 분포했으나, 최종적으로 1024 x 1024로 Resize 및 Crop하여 사용하며, 이미지당 BBox는 약 40 ~ 90개이다

<img width="832" alt="image-2" src="https://github.com/user-attachments/assets/6aef7236-7ddf-4b7d-b402-255e71a75470">

- 베트남어 데이터에서 BBox 개수가 가장 많았고, 각 언어당 약 2,000개의 세로 BBox가 포함되어 있었으나, 실제 데이터는 90도 회전된 텍스트 또는 하나의 텍스트에 관한 BBox였다

## 4. Data-Centric

### 외부 데이터 수집

- [CORD 데이터셋](https://github.com/clovaai/cord) (+ 선 Annotation 추가)

### 데이터 전처리

- 1차 전처리 (선 Annotation 수정, 배경 제거)
    
    <img width="792" alt="스크린샷 2024-12-06 오전 10 27 30" src="https://github.com/user-attachments/assets/fec48e64-5c37-46b4-b061-1c09743f59f5">

    <img width="506" alt="스크린샷 2024-12-06 오전 10 27 42" src="https://github.com/user-attachments/assets/91dffa0e-acd4-46d5-bed5-5f34ba703c96">

    
- 2차 전처리 (실선 제거, Polygon → Square)
    
    <img width="795" alt="스크린샷 2024-12-06 오전 10 27 50" src="https://github.com/user-attachments/assets/44801c81-b3a1-4ec3-8b8c-49bf480591eb">

- 3차 전처리 (실선 Annotation 추가)
    
    <img width="793" alt="스크린샷 2024-12-06 오전 10 28 04" src="https://github.com/user-attachments/assets/1027ab53-b065-44bc-bf44-57dd90c9cda8">

- Normalize (데이터 분포 균일화)

### 데이터 증강

- Invert
- Salt & Pepper

## 5. Result

### 실험 결과

- 일반 실험 이미지
  
![image](https://github.com/user-attachments/assets/ff76d408-ae01-4502-83f2-d1897bf6f758)

- 앙상블 실험 결과 (최종 성능 결과) 이미지
  
![image](https://github.com/user-attachments/assets/bb318ace-0c7e-4227-82f9-43f97f59c931)

### Feedback

- Test 데이터셋 성능과 유사한 Validation 데이터셋 성능을 찾기 어려웠음.
- 모델 중심이 아닌 데이터 중심의 실험을 통하여 데이터 전처리 및 가공의 중요성을 깨닫게 된 프로젝트였음.

## 5. Report

- [Wrap-up Report](https://www.notion.so/Data-Centric-13714f1bea6d809fb249cf6dfc2e7fe7?pvs=21)

## 6. How to Run

### Project File Structure

```bash
├── code 
    ├── dataset.py          # 데이터셋 로드, 전처리 및 augmentation 적용 코드
    ├── deteval.py          # 모델 감지 성능 평가 코드
    ├── inference.py        # 모델 추론 코드, 입력 이미지에 대해 예측 수행
    ├── inference_prev.py    # 이전 버전의 추론 코드 (삭제 유/무)? 
    ├── train.py            # 모델 학습을 위한 코드, 데이터셋을 통해 모델 학습
    ├── validate.py         # 모델 검증 코드, 검증 데이터셋을 사용하여 성능 평가
    ├── requirements.txt    # 프로젝트에 필요한 패키지와 버전 정보 기록
├── super_resolution 
    └── tool # 이미지 초해상도 관련 코드 및 도구 모음
└── utils_independent 
		├── convert_rgb.py # 이미지의 색상 공간을 RGB로 변환하는 코드
    ├── ensemble.py # 앙상블 기법을 통해 여러 모델의 예측 결과 결합 코드
    └── ...
```

- 현재 레포지토리를 클론한다
    
    ```bash
    git clone <https://github.com/boostcampaitech7/level2-cv-datacentric-cv-22.git>
    ```
    
- 아래 경로에 데이터셋을 다운받는다
    
    ```bash
    cd level2-cv-datacentric-cv-22/data
    ```
    
- 모델 학습
    
    ```bash
    python train.py
    ```
    
- 추론
    
    ```bash
    python inference.py
    ```
