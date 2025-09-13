# SKHynix PBL TAT 정합성 향상 모델 개발 (4회차)
----
## 범주형 및 연속형 변수를 구분하여 대응 가능한 TabTransformer 기반 예측 모델

TabTransformer 모델을 활용하여 공정 데이터 기반으로 각 공정별 Turn Around Time(TAT)을 예측하는 딥러닝 시스템입니다. 범주형 특성과 연속형 특성을 효과적으로 결합하여 정확한 TAT 예측을 제공합니다.

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
- **TabTransformer 기반 예측**: 범주형/연속형 변수 동시 처리에 최적화된 Transformer 모델
- **다양한 데이터 타입 지원**: oper_grou, days, shift 등 범주형 변수와 연속형 공정 변수 처리
- **자동 전처리**: 라벨 인코딩, 표준화 등 자동화된 데이터 전처리
- **유연한 데이터 분할**: 시계열 특성을 고려한 순차적 데이터 분할
- **성능 평가**: MAE, MAPE, RMSE 등 다양한 메트릭 제공

### 데이터 구조
- **범주형 변수**: `oper_group`, `days`, `shift`, `x1`
- **연속형 변수**: `x2` ~ `x21`
- **예측 목표**: `y` 

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
Excel 파일을 프로젝트 폴더에 배치하고 `configs/dataset.yaml`에서 경로를 설정합니다.

### 2. 설정 파일 확인
`configs/` 폴더에 기본 설정 파일들이 제공됩니다:
- `dataset.yaml`: 데이터 관련 설정
- `model.yaml`: 모델 구조 설정  
- `training.yaml`: 훈련 관련 설정

### 3. 실행
```bash
# 기본 설정으로 훈련 실행
python main.py

# 특정 GPU 사용하여 훈련 실행
python main.py --gpu 0 --mode train

# 사용자 정의 실험명으로 실행
python main.py --exp-name my_experiment
```

## ⚙️ 설정 파일 구성

### configs/dataset.yaml
```yaml
file_path: "/path/to/your/excel/file.xlsx"
categorical_columns: ["oper_group", "days", "shift", "x1"]
continuous_columns: ["x2", "x3", ..., "x21"]
target_column: "y"
sheet_names: ["Data_Set1(사외)", "Data_Set2(사외)"]
additional_drop_columns: ["lot_cd", "oper_area"]
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
normalize_continuous: true
categories: [277, 7, 3, 20]  # 각 범주형 변수의 고유값 개수
```

**주요 설정 항목:**
- `file_path`: Excel 파일 경로
- `categorical_columns`: 범주형 변수 목록 (공정 그룹, 일자, 교대조 등)
- `continuous_columns`: 연속형 변수 목록 (x2~x21)
- `categories`: 각 범주형 변수의 고유값 개수 (모델 구조 결정에 사용)
- `normalize_continuous`: 연속형 변수 표준화 여부

### configs/model.yaml
```yaml
dim: 32                    # 임베딩 차원
dim_out: 1                 # 출력 차원 (회귀이므로 1)
depth: 6                   # Transformer 레이어 수
heads: 8                   # 어텐션 헤드 수
dim_head: 16               # 각 헤드의 차원
attn_dropout: 0.1          # 어텐션 드롭아웃
ff_dropout: 0.1            # 피드포워드 드롭아웃
mlp_hidden_mults: [4, 2]   # MLP 은닉층 배수
mlp_act: "ReLU"            # MLP 활성화 함수
```

**모델 매개변수 설명:**
- `dim`: TabTransformer의 기본 임베딩 차원
- `depth`: Transformer 인코더 레이어 개수
- `heads`: Multi-Head Attention의 헤드 수
- `mlp_hidden_mults`: 최종 MLP의 은닉층 크기 배수

### configs/training.yaml
```yaml
training:
  batch_size: 256          # 배치 크기
  learning_rate: 0.001     # 학습률
  num_epochs: 100          # 최대 훈련 에폭
  weight_decay: 0.01       # 가중치 감쇠
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 10
  early_stopping_patience: 20

validation:
  metric: "mae"            # 조기 종료 기준 메트릭

evaluation:
  metrics: ["mae", "mape"]  # 평가 메트릭

logging:
  log_interval: 10         # 로그 출력 간격
  save_model_every: 20     # 모델 저장 간격
```

## 📖 사용법

### 기본 사용법 (훈련)
```bash
python main.py
```

### 평가 모드로 실행
```bash
python main.py --mode eval --model-path ./outputs/exp_20241201_120000/best_model.pth
```

### 매개변수 설명
- `--config-dir`: 설정 파일이 있는 디렉토리 경로 (기본값: configs)
- `--mode`: 실행 모드 (train 또는 eval, 기본값: train)
- `--model-path`: 평가 모드에서 사용할 모델 경로
- `--output-dir`: 결과 저장 디렉토리 (기본값: outputs)
- `--exp-name`: 실험명 (미지정시 자동 생성)
- `--gpu`: 사용할 GPU 번호

### 실행 예시
```bash
# 특정 GPU에서 사용자 정의 실험명으로 훈련
python main.py --gpu 1 --exp-name tat_prediction_v1

# 훈련된 모델로 평가 수행
python main.py --mode eval \
    --model-path ./outputs/tat_prediction_v1/best_model.pth \
    --output-dir ./evaluation_results
```

## 📊 출력 결과

### 1. 훈련 과정 로그 (예시)
```
Loading and preprocessing data...
Categorical columns: ['oper_group', 'days', 'shift', 'x1']
Continuous columns: ['x2', 'x3', ..., 'x21']

Data loaded successfully:
  - Categories: [277, 7, 3, 20]
  - Number of continuous features: 20
  - Train samples: 38,800
  - Validation samples: 4,850
  - Test samples: 4,850

Model created with 145,673 parameters

Epoch  10/100: Train Loss: 0.0456, Val MAE: 0.0423, Val MAPE: 8.92
Epoch  20/100: Train Loss: 0.0398, Val MAE: 0.0401, Val MAPE: 8.54
→ Best model saved! (Val MAE: 0.0401)
```

### 2. 결과 파일
실행 완료 후 `outputs/실험명/` 폴더에 다음 파일들이 생성됩니다:

**test_results.json**: 종합 성능 지표
```json
{
  "experiment_info": {
    "exp_name": "exp_20241201_120000",
    "timestamp": "2024-12-01T12:00:00",
    "mode": "train"
  },
  "test_metrics": {
    "MAE": 0.0398,
    "MAPE": 8.24
  },
  "dataset_info": {
    "categories": [277, 7, 3, 20],
    "num_continuous": 20,
    "train_samples": 38800,
    "val_samples": 4850,
    "test_samples": 4850
  }
}
```

**test_predictions.csv**: 예측 결과
| actual | predicted | residual |
|--------|-----------|----------|
| 2.45   | 2.42      | 0.03     |
| 1.89   | 1.91      | -0.02    |
| 3.12   | 3.08      | 0.04     |

### 3. 저장된 모델 파일
- `best_model.pth`: 최고 성능 모델 체크포인트
- `model_epoch_20.pth`, `model_epoch_40.pth`, ...: 주기적 저장된 모델
- `config.json`: 실험에 사용된 설정 파일 백업

## 🔍 성능 메트릭 이해하기

- **MAE** (Mean Absolute Error): 예측 오차의 절댓값 평균, 단위는 TAT와 동일
- **MAPE** (Mean Absolute Percentage Error): 예측 오차의 절대 백분율 평균 (%)
- **RMSE** (Root Mean Square Error): 예측 오차의 제곱 평균 제곱근

## ❗ 문제 해결

### 자주 발생하는 문제들

#### 1. 메모리 부족 오류
```
RuntimeError: CUDA out of memory
```
**해결방법:**
- `training.yaml`에서 `batch_size`를 줄여보세요 (예: 256 → 128)
- `model.yaml`에서 `dim`을 줄여보세요 (예: 32 → 16)
- `model.yaml`에서 `depth`를 줄여보세요 (예: 6 → 4)

#### 2. 데이터 로드 오류
```
FileNotFoundError: Excel file not found
```
**해결방법:**
- `dataset.yaml`의 `file_path` 경로가 정확한지 확인
- Excel 파일이 존재하는지 확인
- 시트 이름이 `sheet_names`와 정확히 일치하는지 확인

#### 3. 범주형 변수 개수 불일치
```
ValueError: categories mismatch
```
**해결방법:**
- 실제 데이터의 범주형 변수 고유값 개수 확인:
```python
import pandas as pd
df = pd.read_excel('your_file.xlsx', sheet_name='Data_Set1(사외)', header=1)
for col in ['oper_group', 'days', 'shift', 'x1']:
    print(f"{col}: {df[col].nunique()} unique values")
```
- `dataset.yaml`의 `categories` 값을 실제 데이터에 맞게 수정

#### 4. 훈련 중 Loss가 감소하지 않는 경우
**해결방법:**
- 학습률을 줄여보세요: `learning_rate: 0.0001`
- 모델 복잡도를 조정해보세요: `depth: 4`, `dim: 16`
- 정규화를 강화해보세요: `weight_decay: 0.1`

#### 5. 과적합 발생
```
Train loss decreases but validation loss increases
```
**해결방법:**
- 드롭아웃을 증가시키세요: `attn_dropout: 0.2`, `ff_dropout: 0.2`
- 조기 종료 patience를 줄여보세요: `early_stopping_patience: 10`
- 훈련 데이터를 늘려보세요: `train_ratio: 0.85`

### 성능 최적화 팁

#### 1. 하이퍼파라미터 튜닝 순서
1. **배치 크기**: GPU 메모리에 맞는 최대 크기 설정
2. **학습률**: 0.001부터 시작하여 조정
3. **모델 크기**: 데이터 크기에 맞게 dim, depth 조정
4. **정규화**: 과적합 방지를 위한 드롭아웃, weight_decay 조정

#### 2. 데이터별 권장 설정
- **소규모 데이터 (< 10K 샘플)**: dim=16, depth=3, batch_size=64
- **중규모 데이터 (10K-50K)**: dim=32, depth=6, batch_size=256 (기본값)
- **대규모 데이터 (> 50K)**: dim=64, depth=8, batch_size=512

#### 3. 훈련 모니터링
```bash
# 훈련 로그 실시간 확인
tail -f training.log

# GPU 사용량 모니터링
nvidia-smi -l 1
```

### 디버깅 모드
상세한 디버깅 정보를 원할 경우:
```bash
# 로깅 레벨을 DEBUG로 설정하여 실행
python -c "import logging; logging.basicConfig(level=logging.DEBUG)" && python main.py
```

### 실험 관리
여러 실험을 체계적으로 관리하려면:
```bash
# 실험 1: 기본 설정
python main.py --exp-name baseline

# 실험 2: 더 깊은 모델
python main.py --exp-name deep_model
# (model.yaml에서 depth: 8로 변경 후 실행)

# 실험 3: 큰 배치 크기
python main.py --exp-name large_batch
# (training.yaml에서 batch_size: 512로 변경 후 실행)
```

## 📞 지원

문제가 지속될 경우:
1. 로그 파일 (`training.log`) 확인
2. 설정 파일들 재검토 (configs/ 폴더 내 모든 YAML 파일)
3. 데이터 형식 및 컬럼명 정확성 확인
4. GPU/메모리 사용량 모니터링
5. 범주형 변수의 고유값 개수와 설정값 일치 확인

---

**참고**: TabTransformer는 범주형과 연속형 데이터를 효과적으로 결합하여 처리하는 모델로, 공정 데이터와 같은 혼합 데이터 타입에 특히 적합합니다.