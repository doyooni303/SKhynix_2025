# SKHynix PBL TAT 정합성 향상 모델 개발 (4회차)
----
## 시간 축과 공정 순서가 반영된 샘플 대상의 Transformer 기반 예측 모델(TimeOperTransformer)

공정 데이터를 활용하여 각 공정별 Turn Around Time(TAT)을 예측하는 딥러닝 시스템입니다. 시계열 데이터의 특성을 고려하여 Transformer 또는 LSTM을 사용한 모델로 구성되어 있습니다.

## 📋 목차
- [시스템 개요](#시스템-개요)
- [요구사항](#요구사항)
- [빠른 시작](#빠른-시작)
- [설정 파일 구성](#설정-파일-구성)
- [사용법](#사용법)
- [출력 결과](#출력-결과)
- [문제 해결](#문제-해결)

## 🎯 시스템 개요

### 주요 기능
- **시계열 공정 데이터 처리**: 시간별 공정 데이터를 효율적으로 처리
- **다양한 데이터 타입 지원**: 범주형/연속형 변수 동시 처리
- **딥러닝 모델**: Transformer 또는 LSTM 기반 예측 모델
- **자동 전처리**: 데이터 정제 및 특성 엔지니어링 자동화
- **성능 평가**: RMSE, MAE, MAPE 등 다양한 메트릭 제공

### 데이터 구조
- **시간 정보**: `timekey_hr` (시간별 키)
- **공정 정보**: `oper_id` (공정 ID), `oper_group` (공정 그룹)
- **입력 변수**: 범주형 변수 (예: days, shift, x1) + 연속형 변수 (예: x2-x21)
- **예측 목표**: `y` (Turn Around Time)

## 🔧 요구사항

### 시스템 요구사항
- Python 3.9+
- CUDA 지원 GPU (선택사항, CPU로도 실행 가능)
- 메모리: 8GB 이상 권장

### 필수 패키지(requirements.txt 확인)
```bash
# 기본 패키지들
openpyxl
pandas
matplotlib
seaborn
scipy
scikit-learn

# 보조 패키지들
pyyaml

# cuda 버젼에 맞게 pytorch 설정 (https://pytorch.org/get-started/previous-versions/)
# 별도로 아래의 명령어로 설치해야 함
# !pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# TabTransformer 패키지
tab-transformer-pytorch
```


## 🚀 빠른 시작

### 1. 데이터 준비
Excel 파일을 프로젝트 폴더에 배치하고 `dataset.yaml`에서 경로를 설정합니다.

### 2. 설정 파일 확인
기본 설정 파일들이 제공됩니다:
- `dataset.yaml`: 데이터 관련 설정
- `model.yaml`: 모델 구조 설정  
- `training.yaml`: 훈련 관련 설정

### 3. 실행
```bash
python main.py
```

기본 설정으로 실행되며, 모든 과정이 자동으로 진행됩니다.

## ⚙️ 설정 파일 구성

### dataset.yaml
```yaml
data_path: "/path/to/your/excel/file.xlsx"
categorical_columns: ["oper_group","days","shift","x1"]
continuous_columns: ["x2","x3","x4",...,"x21"]
sheet_names: ["Data_Set1(사외)", "Data_Set2(사외)"]
time_window: 24        # 시간 윈도우 크기
batch_size: 8          # 배치 크기
target_column: "y"     # 예측 대상 컬럼
```

**주요 설정 항목:**
- `data_path`: Excel 파일 경로
- `categorical_columns`: 범주형 변수 목록
- `continuous_columns`: 연속형 변수 목록
- `time_window`: 한 번에 처리할 시간 단위 개수(기본 24시간)

### model.yaml
```yaml
backbone_type: transformer  # transformer 또는 lstm
embedding_dim: 8
dropout: 0.1

transformer: 
  num_layers: 6           # 레이어 수
  num_heads: 8            # 어텐션 헤드 수
  hidden_dim: 512         # 은닉 차원
  feedforward_dim: 2048   # 피드포워드 차원

lstm: 
  num_layers: 2           # LSTM 레이어 수
  hidden_dim: 128         # 은닉 상태 차원
  bidirectional: true     # 양방향 LSTM 사용
```

**모델 선택:**
- `transformer`: 복잡한 패턴 학습에 적합, 더 많은 계산 자원 필요
- `lstm`: 상대적으로 빠른 훈련, 시계열 데이터에 특화

### training.yaml
```yaml
num_epochs: 10           # 훈련 에폭 수
learning_rate: 0.001     # 학습률
patience: 10             # 조기 종료 대기 에폭
save_path: 'best_model.pth'
output_dir: 'results'    # 결과 저장 폴더
```

## 📖 사용법

### 기본 사용법
```bash
python main.py
```

### 사용자 정의 설정으로 실행
```bash
python main.py \
    --dataset_config ./my_configs/dataset.yaml \
    --model_config ./my_configs/model.yaml \
    --training_config ./my_configs/training.yaml \
    --gpu 0
```

### 매개변수 설명
- `--dataset_config`: 데이터셋 설정 파일 경로
- `--model_config`: 모델 설정 파일 경로  
- `--training_config`: 훈련 설정 파일 경로
- `--gpu`: 사용할 GPU 번호 (기본값: 0)

## 📊 출력 결과

### 1. 훈련 과정 로그 (예시)
```
Dataset 구성 완료:
  - 총 샘플 수: 1,234개
  - 시간 윈도우 크기: 24
  - 전체 데이터셋 max_oper_per_hour: 50

Epoch   1: Train Loss=0.1234, Val Loss=0.1456, Train MAPE=12.34%, Val MAPE=14.56%
→ Best model saved! (Val Loss: 0.1456)
```

### 2. 결과 파일
실행 완료 후 `results/` 폴더에 다음 파일들이 생성됩니다:

**metrics.json**: 성능 지표
```json
{
  "test_loss": 0.1234,
  "metrics": {
    "rmse": 0.1234,
    "mae": 0.0987,
    "mape": 12.34,
    "valid_count": 5678
  }
}
```

**predictions.csv**: 예측 결과
| timekey_hr | oper_id | predicted | actual | error | abs_error | abs_percent_error |
|------------|---------|-----------|--------|--------|-----------|-------------------|
| 2024010100 | 1001    | 2.34      | 2.45   | -0.11  | 0.11      | 4.49              |

### 3. 저장된 모델
- `best_model.pth`: 최고 성능 모델 체크포인트

## 🔍 성능 메트릭 이해하기

- **RMSE** (Root Mean Square Error): 예측 오차의 제곱 평균 제곱근
- **MAE** (Mean Absolute Error): 예측 오차의 절댓값 평균
- **MAPE** (Mean Absolute Percentage Error): 예측 오차의 절대 백분율 평균
- **Valid Count**: 유효한 예측 데이터 개수

## ❗ 문제 해결

### 자주 발생하는 문제들

#### 1. 메모리 부족 오류
```
RuntimeError: CUDA out of memory
```
**해결방법:**
- `dataset.yaml`에서 `batch_size`를 줄여보세요 (예: 8 → 4)
- `model.yaml`에서 `hidden_dim`을 줄여보세요 (예: 512 → 256)

#### 2. 데이터 로드 오류
```
FileNotFoundError: Excel file not found
```
**해결방법:**
- `dataset.yaml`의 `data_path` 경로가 정확한지 확인
- Excel 파일이 존재하는지 확인
- 시트 이름이 `sheet_names`와 일치하는지 확인

#### 3. 컬럼 이름 오류
```
KeyError: 'column_name' not found
```
**해결방법:**
- Excel 파일의 실제 컬럼명과 설정파일의 컬럼명이 일치하는지 확인
- 대소문자, 공백 등도 정확히 일치해야 합니다

#### 4. GPU 사용 불가
```
CUDA is not available
```
**해결방법:**
- CPU 모드로 자동 전환되므로 계속 사용 가능
- GPU를 사용하고 싶다면 CUDA 설치 확인

### 성능 최적화 팁

#### 1. 모델 크기 조정
- 데이터가 적으면 작은 모델 사용 (hidden_dim=128)
- 데이터가 많으면 큰 모델 사용 (hidden_dim=512)

#### 2. 배치 크기 조정  
- GPU 메모리에 맞게 batch_size 조정
- 일반적으로 8, 16, 32 중 선택

#### 3. 시간 윈도우 조정
- 짧은 패턴: time_window=12
- 긴 패턴: time_window=48 (샘플이 너무 커져서 권장하진 않음)

### 디버깅 모드
상세한 로그를 보고 싶다면:
```bash
# 로그 레벨을 DEBUG로 변경
export PYTHONPATH=.:$PYTHONPATH
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python main.py
```

## 📞 지원

문제가 지속될 경우:
1. 로그 파일 (`training.log`) 확인
2. 설정 파일들 재검토
3. 데이터 형식 및 컬럼명 확인
4. GPU/메모리 사용량 모니터링

---