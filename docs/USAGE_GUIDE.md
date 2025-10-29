# 물류 수요예측 시스템 - 실행 가이드

## 빠른 시작 (Quick Start)

### 1. 환경 설정

#### Python 버전 확인
```bash
python --version
# Python 3.8 이상 필요
```

#### 필수 라이브러리 설치
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

또는 requirements.txt 사용:
```bash
pip install -r requirements.txt
```

#### requirements.txt 내용
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### 2. 데이터 준비

#### 필수 컬럼
```
date         : 날짜 (예: 2024.01.01)
sku_code     : SKU 코드
degr         : 온도대 (예: 냉장, 냉동, 상온)
box_qty      : 박스 수량
```

#### 데이터 파일 위치
```
ML/
├── main.py
├── demand_forecast_system.py
└── shipment.csv  ← 여기에 데이터 파일 배치
```

### 3. 실행

#### 기본 실행
```bash
cd ML
python main.py
```

#### 실행 화면
```
======================================================================
🚀 입고 데이터 수요예측 시스템
======================================================================

✓ 데이터 파일 발견: shipment.csv

======================================================================
Step 1: 시스템 초기화
======================================================================
✓ 출력 디렉토리 생성: outputs

======================================================================
Step 2: 데이터 로딩
======================================================================
✓ 데이터 로딩 완료: 245673 rows

...
```

---

## 단계별 실행 가이드

### Step 1: 시스템 초기화
```python
from demand_forecast_system import DemandForecastSystem

dfs = DemandForecastSystem(output_dir='outputs')
```

**설명**:
- `output_dir`: 결과 파일이 저장될 폴더 (기본값: 'outputs')
- 폴더가 없으면 자동 생성

### Step 2: 데이터 로딩
```python
# 방법 1: CSV 파일에서 직접 로딩
dfs.load_data(data_path='shipment.csv')

# 방법 2: DataFrame 전달
import pandas as pd
df = pd.read_csv('shipment.csv')
dfs.load_data(df=df)

# 방법 3: Excel 파일
dfs.load_data(data_path='shipment.xlsx')
```

**지원 파일 형식**:
- CSV (`.csv`)
- Excel (`.xlsx`)
- TSV (`.txt`)

### Step 3: 데이터 전처리
```python
dfs.preprocess_data()
```

**자동 처리 내용**:
- 날짜 형식 변환
- 요일, 월, 계절 특성 생성
- 주말/평일 구분
- 온도대 표준화
- 결측치 처리

**출력 예시**:
```
✓ 데이터 전처리 완료
  - 기간: 2024-01-01 ~ 2024-12-31
  - SKU 종류: 156개
  - 온도대: ['냉장' '냉동' '상온']
```

### Step 4: 집계 데이터 생성
```python
dfs.create_aggregations()
```

**생성되는 집계**:
- 일별 집계: SKU + 온도대별
- 주별 집계: 주간 단위

**출력 예시**:
```
✓ 집계 데이터 생성 완료
  - 일별 데이터: 12,450 rows
  - 주별 데이터: 1,820 rows
```

### Step 5: 패턴 분석
```python
dfs.analyze_patterns()
```

**분석 항목**:
1. 요일별 입고 패턴
2. 온도대별 입고 패턴
3. SKU별 입고 패턴 (Top 10)
4. 계절별 입고 패턴

**생성 파일**:
- `weekday_pattern.png`
- `temperature_pattern.png`
- `sku_pattern.png`
- `seasonal_pattern.png`

### Step 6: SKU 클러스터링
```python
# 6-1. 특성 추출
dfs.extract_sku_features()

# 6-2. 클러스터링 수행
dfs.perform_sku_clustering(n_clusters=4, method='kmeans')

# 6-3. 시각화
dfs.visualize_clusters()
```

**파라미터**:
- `n_clusters`: 클러스터 개수 (기본값: 4)
- `method`: 'kmeans' 또는 'dbscan'

**생성 파일**:
- `sku_clustering.png`
- `cluster_heatmap.png`

### Step 7: 수요예측 모델 구축
```python
dfs.build_forecast_models()
```

**구축되는 모델**:
1. 이동평균 (Moving Average)
2. 지수평활 (Exponential Smoothing)
3. 요일 패턴 기반

### Step 8: 모델 평가
```python
results = dfs.evaluate_forecasts()
```

**평가 지표**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

**출력 예시**:
```
모델 성능 비교:
              Model     MAE    RMSE  MAPE(%)
    moving_average   45.23   67.89    18.34
exponential_smoothing   42.56   65.12    17.21
     weekday_pattern   38.90   61.45    15.67
```

### Step 9: 미래 수요 예측
```python
forecast = dfs.generate_forecast_report(forecast_days=7)
```

**파라미터**:
- `forecast_days`: 예측할 일수 (기본값: 7)

**생성 파일**:
- `forecast_report.csv`

### Step 10: 예측 결과 시각화
```python
# 자동으로 가장 많은 수요의 SKU 선택
dfs.visualize_forecast()

# 특정 SKU 지정
dfs.visualize_forecast(sku_code='2014728', days_back=30)
```

**파라미터**:
- `sku_code`: 시각화할 SKU (None이면 자동 선택)
- `days_back`: 표시할 과거 일수 (기본값: 30)

### Step 11: 종합 리포트 생성
```python
dfs.generate_summary_report()
```

**생성 파일**:
- `summary_report.txt`

**포함 내용**:
1. 데이터 기본 정보
2. 요일별 패턴 분석
3. 온도대별 패턴 분석
4. Top 5 SKU
5. 수요예측 기반 추천사항

---

## 고급 사용법

### 1. 커스텀 클러스터링

#### 다양한 클러스터 개수 실험
```python
for k in [2, 3, 4, 5, 6]:
    print(f"\n{'='*70}")
    print(f"Testing with {k} clusters")
    print(f"{'='*70}")

    dfs.perform_sku_clustering(n_clusters=k, method='kmeans')
    dfs.visualize_clusters()
```

#### DBSCAN 사용
```python
# DBSCAN은 클러스터 개수를 자동으로 결정
dfs.perform_sku_clustering(method='dbscan')
```

### 2. 특정 SKU 그룹 분석

```python
# 특정 온도대만 분석
dfs.df_processed = dfs.df_processed[dfs.df_processed['temp_category'] == '냉장']
dfs.create_aggregations()
dfs.analyze_patterns()
```

### 3. 배치 처리

```python
import glob

# 여러 파일 자동 처리
for file in glob.glob('data/*.csv'):
    print(f"\nProcessing {file}...")

    dfs = DemandForecastSystem(output_dir=f'outputs/{file.stem}')
    dfs.load_data(data_path=file)
    dfs.preprocess_data()
    dfs.create_aggregations()
    dfs.analyze_patterns()
    dfs.extract_sku_features()
    dfs.perform_sku_clustering()
    dfs.visualize_clusters()
    dfs.build_forecast_models()
    dfs.evaluate_forecasts()
    dfs.generate_summary_report()
```

### 4. 결과 데이터 활용

#### 클러스터 정보 추출
```python
# 클러스터별 SKU 리스트
for cluster_id in range(4):
    skus = dfs.sku_features[dfs.sku_features['cluster'] == cluster_id]['sku_code'].tolist()
    print(f"Cluster {cluster_id}: {skus}")

# CSV로 저장
dfs.sku_features.to_csv('outputs/sku_clusters.csv', index=False)
```

#### 예측 결과 활용
```python
# 예측 데이터 로드
forecast_df = pd.read_csv('outputs/forecast_report.csv')

# 특정 날짜의 총 예측량
date_forecast = forecast_df[forecast_df['date'] == '2024-01-15']
total_boxes = date_forecast['forecast_boxes'].sum()
print(f"2024-01-15 예상 입고량: {total_boxes:.0f} boxes")

# SKU별 주간 예측
weekly_forecast = forecast_df.groupby('sku_code')['forecast_boxes'].sum()
print(weekly_forecast)
```

---

## 출력 파일 가이드

### 폴더 구조
```
outputs/
├── weekday_pattern.png
├── temperature_pattern.png
├── sku_pattern.png
├── seasonal_pattern.png
├── sku_clustering.png
├── cluster_heatmap.png
├── model_comparison.png
├── forecast_visualization.png
├── forecast_report.csv
└── summary_report.txt
```

### 파일별 설명

#### 1. weekday_pattern.png
**내용**: 요일별 평균 입고량 및 평일 vs 주말 비교
**활용**: 요일별 인력 배치 계획

#### 2. temperature_pattern.png
**내용**: 온도대별 총 입고량 및 평균 입고량
**활용**: 온도대별 보관 공간 배분

#### 3. sku_pattern.png
**내용**: 입고량 기준 Top 10 SKU
**활용**: 핵심 SKU 식별 및 우선 관리

#### 4. seasonal_pattern.png
**내용**: 계절별 평균 입고량
**활용**: 계절별 운영 전략 수립

#### 5. sku_clustering.png
**내용**:
- PCA 클러스터 시각화
- 수요 vs 변동계수 산점도
- 클러스터별 수요 분포
- 클러스터별 SKU 개수

**활용**: 클러스터 특성 파악 및 분류 검증

#### 6. cluster_heatmap.png
**내용**: 클러스터별 특성 프로파일 히트맵
**활용**: 클러스터 간 차이점 비교

#### 7. model_comparison.png
**내용**: 예측 모델 성능 비교 (MAE, RMSE, MAPE)
**활용**: 최적 모델 선택

#### 8. forecast_visualization.png
**내용**: 선택된 SKU의 실제값 vs 예측값
**활용**: 예측 정확도 시각적 확인

#### 9. forecast_report.csv
**구조**:
```csv
date,weekday,sku_code,temp_category,forecast_boxes
2024-01-15,월,2014728,냉장,450.2
2024-01-15,월,2014968,냉장,230.5
...
```
**활용**: 일별 예측 데이터 기반 운영 계획

#### 10. summary_report.txt
**구조**:
```
1. 데이터 기본 정보
2. 요일별 패턴 분석
3. 온도대별 패턴 분석
4. Top 5 SKU
5. 수요예측 기반 추천사항
```
**활용**: 경영진 보고용 요약 리포트

---

## 문제 해결 (Troubleshooting)

### 1. 데이터 로딩 오류

#### 오류: "파일을 찾을 수 없습니다"
```python
# 해결: 절대 경로 사용
import os
file_path = os.path.abspath('shipment.csv')
dfs.load_data(data_path=file_path)
```

#### 오류: "필수 컬럼이 없습니다"
```python
# 해결: 컬럼명 확인 및 변경
df = pd.read_csv('shipment.csv')
print(df.columns)

# 컬럼명 변경
df.rename(columns={'date_col': 'date', 'sku': 'sku_code'}, inplace=True)
dfs.load_data(df=df)
```

### 2. 메모리 오류

#### 오류: "MemoryError"
```python
# 해결: 데이터 샘플링
df = pd.read_csv('shipment.csv')

# 최근 6개월만 사용
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2024-07-01']

dfs.load_data(df=df)
```

### 3. 시각화 오류

#### 오류: "한글 폰트가 깨짐"
```python
# 해결: 폰트 설정
import matplotlib.pyplot as plt

# Windows
plt.rcParams['font.family'] = 'Malgun Gothic'

# Mac
plt.rcParams['font.family'] = 'AppleGothic'

# Linux
plt.rcParams['font.family'] = 'NanumGothic'
```

#### 오류: "Figure가 표시되지 않음"
```python
# 해결: 백엔드 변경
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
```

### 4. 클러스터링 경고

#### 경고: "클러스터 개수가 SKU 개수보다 많음"
```python
# 해결: 클러스터 개수 조정
n_skus = dfs.sku_features.shape[0]
n_clusters = min(4, n_skus - 1)
dfs.perform_sku_clustering(n_clusters=n_clusters)
```

#### 경고: "수렴하지 않음"
```python
# 해결: max_iter 증가
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, max_iter=1000, random_state=42)
```

---

## 성능 최적화

### 1. 대용량 데이터 처리

```python
# 청크 단위로 로딩
import pandas as pd

chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    # 필요한 컬럼만 선택
    chunk = chunk[['date', 'sku_code', 'degr', 'box_qty']]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
dfs.load_data(df=df)
```

### 2. 병렬 처리

```python
from multiprocessing import Pool

def process_sku(sku_code):
    # SKU별 예측 로직
    pass

# 병렬 실행
with Pool(4) as p:
    results = p.map(process_sku, sku_codes)
```

### 3. 캐싱

```python
import pickle

# 전처리 결과 저장
dfs.preprocess_data()
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(dfs, f)

# 재사용
with open('preprocessed_data.pkl', 'rb') as f:
    dfs = pickle.load(f)
```

---

## 자동화 스크립트

### 1. 일일 자동 실행

#### Linux/Mac (cron)
```bash
# crontab -e
0 6 * * * cd /path/to/ML && python main.py >> logs/forecast_$(date +\%Y\%m\%d).log 2>&1
```

#### Windows (Task Scheduler)
```powershell
# run_forecast.bat
@echo off
cd C:\path\to\ML
python main.py >> logs\forecast_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
```

### 2. 결과 자동 이메일 발송

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def send_report_email():
    # 이메일 설정
    sender = 'forecast@example.com'
    recipients = ['manager@example.com', 'ops@example.com']

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = f'수요예측 리포트 - {datetime.now().strftime("%Y-%m-%d")}'

    # 본문
    with open('outputs/summary_report.txt', 'r', encoding='utf-8') as f:
        body = f.read()
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    # 첨부파일
    files = ['forecast_report.csv', 'sku_clustering.png']
    for file in files:
        with open(f'outputs/{file}', 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={file}')
            msg.attach(part)

    # 전송
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender, 'password')
        server.send_message(msg)

# main.py 마지막에 추가
if __name__ == "__main__":
    main()
    send_report_email()
```

### 3. API 서버화

```python
from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)
dfs = DemandForecastSystem()

@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    data = request.json
    sku_code = data.get('sku_code')
    days = data.get('days', 7)

    # 예측 수행
    forecast = dfs.generate_forecast_report(forecast_days=days)
    result = forecast[forecast['sku_code'] == sku_code].to_dict('records')

    return jsonify(result)

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    return jsonify(dfs.sku_features.to_dict('records'))

if __name__ == '__main__':
    # 초기 학습
    dfs.load_data(data_path='shipment.csv')
    dfs.preprocess_data()
    dfs.create_aggregations()
    dfs.extract_sku_features()
    dfs.perform_sku_clustering()

    # 서버 시작
    app.run(host='0.0.0.0', port=5000)
```

---

## 프로덕션 체크리스트

### 배포 전 확인사항

- [ ] 모든 필수 라이브러리 설치 확인
- [ ] 데이터 형식 검증
- [ ] 출력 디렉토리 권한 확인
- [ ] 로그 시스템 구축
- [ ] 에러 핸들링 추가
- [ ] 성능 테스트 완료
- [ ] 백업 시스템 구축
- [ ] 모니터링 설정
- [ ] 문서화 완료
- [ ] 사용자 교육 완료

### 운영 시 모니터링 항목

1. **데이터 품질**
   - 일일 데이터 유입량
   - 결측치 비율
   - 이상치 탐지

2. **모델 성능**
   - 예측 정확도 (MAPE)
   - 클러스터 안정성
   - 실행 시간

3. **시스템 리소스**
   - CPU 사용률
   - 메모리 사용량
   - 디스크 공간

4. **비즈니스 지표**
   - 예측 활용률
   - 재고 회전율 변화
   - 비용 절감 효과

---

## FAQ

### Q1: 데이터가 부족할 때는?
**A**: 최소 3개월 이상의 데이터 권장. 부족 시 단순 모델(이동평균) 사용

### Q2: 새로운 SKU는 어떻게 처리?
**A**: 초기 3개월은 유사 SKU의 패턴 참고, 이후 자동 클러스터링

### Q3: 클러스터 개수는 어떻게 결정?
**A**: Elbow method 또는 Silhouette score 활용. 일반적으로 3-5개 권장

### Q4: 예측이 부정확할 때는?
**A**:
1. 더 많은 데이터 확보
2. 특성 추가 (프로모션, 재고 등)
3. 클러스터별 모델 튜닝

### Q5: 실시간 예측이 가능한가?
**A**: 일 단위 배치 처리 권장. 실시간 필요 시 API 서버 구축

---

## 추가 리소스

### 관련 문서
- `README.md`: 프로젝트 개요
- `CLUSTERING_ANALYSIS.md`: 클러스터링 상세 분석
- `demand_forecast_system.py`: 소스 코드 (주석 참고)

### 학습 자료
- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Time Series Forecasting Best Practices](https://otexts.com/fpp3/)
- [Inventory Management 101](https://www.ibm.com/topics/inventory-management)

### 커뮤니티
- GitHub Issues: 버그 리포트 및 기능 요청
- Discussion Forum: 사용 팁 공유
- Technical Blog: 최신 업데이트 및 케이스 스터디

---

## 문의

- 기술 지원: tech-support@example.com
- 비즈니스 문의: business@example.com
- 긴급 문의: +82-10-XXXX-XXXX

---

**문서 버전**: 1.0
**최종 업데이트**: 2024-01-XX
**담당자**: Logistics AI Team
