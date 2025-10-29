# 물류 수요예측 시스템 (Logistics Demand Forecast System)

## 개요

입고 데이터를 기반으로 수요를 예측하고 SKU를 클러스터링하여 물류 운영을 최적화하는 AI 시스템입니다.

## 주요 기능

### 1. 데이터 분석
- 일별/주별 입고량 집계 및 분석
- 요일별, 계절별, 온도대별 패턴 분석
- SKU별 수요 트렌드 분석

### 2. SKU 클러스터링 (NEW!)
- 시계열 특성 추출 (평균 수요, 변동성, 트렌드 등)
- K-Means 및 DBSCAN 클러스터링
- 클러스터별 특성 자동 해석
- PCA 기반 2D 시각화

### 3. 수요 예측
- 이동평균(Moving Average) 모델
- 지수평활(Exponential Smoothing) 모델
- 요일 패턴 기반 예측
- 향후 7일 수요 예측

### 4. 시각화 및 리포트
- 8가지 시각화 차트 생성
- 종합 분석 리포트 자동 생성
- CSV 형식 예측 결과 저장

## 시스템 아키텍처

```
입고 데이터 (shipment.csv)
    ↓
데이터 전처리
    ↓
집계 데이터 생성 (일별/주별)
    ↓
패턴 분석 → SKU 클러스터링 (NEW!)
    ↓
수요예측 모델 구축
    ↓
결과 시각화 및 리포트 생성
```

## 실행 방법

```bash
python main.py
```

## 필수 라이브러리

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 출력 파일

### 시각화 파일 (outputs 폴더)
- `weekday_pattern.png`: 요일별 입고 패턴
- `temperature_pattern.png`: 온도대별 패턴
- `sku_pattern.png`: SKU별 입고량 (Top 10)
- `seasonal_pattern.png`: 계절별 패턴
- `sku_clustering.png`: SKU 클러스터 분석 (NEW!)
- `cluster_heatmap.png`: 클러스터별 특성 히트맵 (NEW!)
- `model_comparison.png`: 예측 모델 성능 비교
- `forecast_visualization.png`: 예측 결과 시각화

### 데이터 파일
- `forecast_report.csv`: 향후 7일 예측 결과
- `summary_report.txt`: 종합 분석 리포트

## 파일 구조

```
ML/
├── main.py                      # 메인 실행 파일
├── demand_forecast_system.py    # 수요예측 시스템 클래스
├── shipment.csv                 # 입력 데이터
├── outputs/                     # 출력 결과 폴더
│   ├── *.png                    # 시각화 파일
│   ├── forecast_report.csv      # 예측 결과
│   └── summary_report.txt       # 종합 리포트
├── README.md                    # 프로젝트 설명
├── CLUSTERING_ANALYSIS.md       # 클러스터링 분석 상세 문서
└── USAGE_GUIDE.md              # 사용 가이드
```

## 클러스터링 특성

### 추출되는 특성
1. **기본 통계**: 평균, 표준편차, 최대/최소 수요
2. **변동성**: 변동계수(CV)
3. **패턴 강도**: 요일별, 계절별 패턴 강도
4. **트렌드**: 선형 회귀 기울기
5. **간헐성**: 수요가 0인 날의 비율
6. **주말 비율**: 주말 평균 / 평일 평균

### 클러스터 해석 예시
- **Cluster 0**: 고수요, 변동성 낮음, 안정적
- **Cluster 1**: 저수요, 변동성 높음, 증가 추세
- **Cluster 2**: 중간 수요, 변동성 중간, 감소 추세
- **Cluster 3**: 고수요, 변동성 높음, 계절성 강함

## 활용 방안

### 재고 관리
- 클러스터별 맞춤형 재고 정책 수립
- 고변동성 클러스터: 안전재고 확보
- 저변동성 클러스터: JIT(Just-In-Time) 운영

### 인력 배치
- 요일별 예측 수요에 따른 인력 스케줄링
- 피크 시즌 대비 인력 확보

### 공간 최적화
- 온도대별, SKU별 보관 공간 사전 배치
- 클러스터별 동선 최적화

## 성능 지표

- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **RMSE (Root Mean Squared Error)**: 평균 제곱근 오차
- **MAPE (Mean Absolute Percentage Error)**: 평균 절대 백분율 오차

## 개발 환경

- Python 3.8+
- scikit-learn 1.0+
- pandas 1.3+
- matplotlib 3.4+
- seaborn 0.11+

## 버전 히스토리

### v2.0 (2024-01-XX)
- SKU 클러스터링 기능 추가
- K-Means 및 DBSCAN 알고리즘 지원
- PCA 기반 시각화 추가
- 클러스터 자동 해석 기능

### v1.0 (2024-01-XX)
- 초기 수요예측 시스템 구축
- 패턴 분석 및 시각화
- 3가지 예측 모델 구현

## 참고 문서

- **CLUSTERING_ANALYSIS.md**: 클러스터링 상세 분석 및 비즈니스 활용 (발표용)
- **USAGE_GUIDE.md**: 단계별 실행 가이드 및 문제 해결
- **main.py**: 전체 파이프라인 실행 스크립트
- **demand_forecast_system.py**: 핵심 시스템 클래스 구현

## 실행 예시

```python
# Python 인터프리터에서
from demand_forecast_system import DemandForecastSystem

# 시스템 초기화
dfs = DemandForecastSystem(output_dir='outputs')

# 데이터 로딩
dfs.load_data(data_path='shipment.csv')

# 전처리 및 분석
dfs.preprocess_data()
dfs.create_aggregations()
dfs.analyze_patterns()

# 클러스터링
dfs.extract_sku_features()
dfs.perform_sku_clustering(n_clusters=4, method='kmeans')
dfs.visualize_clusters()

# 예측 및 평가
dfs.build_forecast_models()
dfs.evaluate_forecasts()
dfs.generate_forecast_report(forecast_days=7)
dfs.generate_summary_report()
```

## 주요 기능별 코드 위치

### 클러스터링 관련
- `extract_sku_features()`: demand_forecast_system.py:379
- `perform_sku_clustering()`: demand_forecast_system.py:450
- `visualize_clusters()`: demand_forecast_system.py:547

### 예측 관련
- `build_forecast_models()`: demand_forecast_system.py:664
- `evaluate_forecasts()`: demand_forecast_system.py:736
- `generate_forecast_report()`: demand_forecast_system.py:809

### 시각화 관련
- `analyze_patterns()`: demand_forecast_system.py:206
- `visualize_forecast()`: demand_forecast_system.py:863

## 문의

- 기술 지원: 프로젝트 Issues 탭
- 기능 요청: Pull Request
- 문서 개선: Contribution Welcome

---

**마지막 업데이트**: 2024-01-XX
**개발**: Logistics AI Team
**라이선스**: MIT
