# SKU 클러스터링 플로우 차트

## 전체 프로세스 플로우

```mermaid
flowchart TD
    Start([시작]) --> LoadData[출고 데이터 로드<br/>inbound.csv]
    LoadData --> Preprocess[데이터 전처리]

    Preprocess --> Aggregate[일별 SKU 집계<br/>date + sku_code + temp_category]

    Aggregate --> FeatureExtract{특성 추출<br/>7가지 핵심 특성}

    FeatureExtract --> F1[평균 수요<br/>mean_demand]
    FeatureExtract --> F2[변동계수<br/>cv]
    FeatureExtract --> F3[요일 패턴<br/>weekday_pattern]
    FeatureExtract --> F4[계절 패턴<br/>seasonal_pattern]
    FeatureExtract --> F5[트렌드<br/>trend]
    FeatureExtract --> F6[간헐 수요 비율<br/>zero_demand_ratio]
    FeatureExtract --> F7[주말 비율<br/>weekend_ratio]

    F1 --> Standardize
    F2 --> Standardize
    F3 --> Standardize
    F4 --> Standardize
    F5 --> Standardize
    F6 --> Standardize
    F7 --> Standardize

    Standardize[데이터 표준화<br/>StandardScaler] --> SelectAlgo{알고리즘 선택}

    SelectAlgo -->|K-Means| KMeans[K-Means 클러스터링<br/>n_clusters=4]
    SelectAlgo -->|DBSCAN| DBSCAN[DBSCAN 클러스터링<br/>자동 클러스터 개수]

    KMeans --> Assign[클러스터 레이블 할당]
    DBSCAN --> Assign

    Assign --> Profile[클러스터 프로파일 생성]

    Profile --> C0[Cluster 0<br/>안정적 고수요]
    Profile --> C1[Cluster 1<br/>불규칙 저수요]
    Profile --> C2[Cluster 2<br/>계절성 중수요]
    Profile --> C3[Cluster 3<br/>급증 추세]

    C0 --> Visual
    C1 --> Visual
    C2 --> Visual
    C3 --> Visual

    Visual[시각화<br/>PCA + 차트] --> Analysis[분석 및 해석]

    Analysis --> Strategy{비즈니스 전략}

    Strategy --> S1[재고 정책<br/>클러스터별 차등]
    Strategy --> S2[예측 모델<br/>맞춤형 적용]
    Strategy --> S3[운영 최적화<br/>동선 + 인력]

    S1 --> End([완료])
    S2 --> End
    S3 --> End

    style Start fill:#90EE90
    style End fill:#FFB6C1
    style FeatureExtract fill:#87CEEB
    style SelectAlgo fill:#FFD700
    style Strategy fill:#DDA0DD
    style C0 fill:#98FB98
    style C1 fill:#FFA07A
    style C2 fill:#87CEFA
    style C3 fill:#FFB6C1
```

---

## 특성 추출 상세 플로우

```mermaid
flowchart LR
    SKU[SKU 데이터<br/>시계열] --> Check{충분한<br/>데이터?}

    Check -->|Yes| F1Process[평균 수요 계산<br/>mean daily boxes]
    Check -->|No| Skip[제외]

    F1Process --> F2Process[변동계수 계산<br/>std / mean]
    F2Process --> F3Process[요일 패턴<br/>std of weekday avg]
    F3Process --> F4Process[계절 패턴<br/>std of seasonal avg]
    F4Process --> F5Process[트렌드<br/>linear regression slope]
    F5Process --> F6Process[간헐 수요<br/>zero count ratio]
    F6Process --> F7Process[주말 비율<br/>weekend / weekday]

    F7Process --> Features[7차원 특성 벡터]

    style Features fill:#90EE90
    style Skip fill:#FFB6C1
```

---

## K-Means 알고리즘 상세 플로우

```mermaid
flowchart TD
    Init[K-Means 시작<br/>n_clusters=4] --> Random[4개 중심점<br/>랜덤 초기화]

    Random --> Assign1[각 SKU를<br/>가장 가까운<br/>중심점에 할당]

    Assign1 --> Calc1[각 클러스터의<br/>중심점 재계산]

    Calc1 --> Check1{중심점<br/>변화?}

    Check1 -->|Yes| Assign1
    Check1 -->|No| Converge[수렴 완료]

    Converge --> Labels[클러스터 레이블<br/>0, 1, 2, 3]

    style Init fill:#87CEEB
    style Converge fill:#90EE90
    style Labels fill:#FFD700
```

---

## DBSCAN 알고리즘 상세 플로우

```mermaid
flowchart TD
    Init[DBSCAN 시작] --> Params[파라미터 설정<br/>eps, min_samples]

    Params --> Loop{모든 점<br/>방문?}

    Loop -->|No| Point[다음 점 선택]

    Point --> Neighbors[epsilon 반경 내<br/>이웃 점 찾기]

    Neighbors --> CheckDensity{이웃 수 >=<br/>min_samples?}

    CheckDensity -->|Yes| CorePoint[핵심 점<br/>새 클러스터 생성]
    CheckDensity -->|No| NoiseCheck{이미<br/>클러스터<br/>소속?}

    NoiseCheck -->|No| Noise[노이즈 점<br/>레이블 = -1]
    NoiseCheck -->|Yes| Border[경계 점]

    CorePoint --> Expand[클러스터 확장<br/>이웃의 이웃 탐색]
    Expand --> Loop

    Border --> Loop
    Noise --> Loop

    Loop -->|Yes| Result[클러스터 레이블<br/>0, 1, 2, ..., -1]

    style Init fill:#87CEEB
    style Result fill:#90EE90
    style Noise fill:#FFB6C1
```

---

## 클러스터 분석 및 활용 플로우

```mermaid
flowchart TD
    Clusters[클러스터링 결과] --> Analyze[클러스터 프로파일링]

    Analyze --> Metrics{평가 지표 계산}

    Metrics --> Silhouette[실루엣 점수<br/>클러스터 품질]
    Metrics --> Inertia[Inertia<br/>응집도]
    Metrics --> Size[클러스터 크기<br/>SKU 분포]

    Silhouette --> Quality{품질<br/>충분?}

    Quality -->|No| Retune[파라미터 재조정<br/>또는 알고리즘 변경]
    Retune --> Clusters

    Quality -->|Yes| Interpret[비즈니스 해석]

    Interpret --> Profile0[Cluster 0 프로파일<br/>안정적 고수요]
    Interpret --> Profile1[Cluster 1 프로파일<br/>불규칙 저수요]
    Interpret --> Profile2[Cluster 2 프로파일<br/>계절성 중수요]
    Interpret --> Profile3[Cluster 3 프로파일<br/>급증 추세]

    Profile0 --> Apply0[EOQ 재고 정책<br/>이동평균 예측]
    Profile1 --> Apply1[MTO 정책<br/>Croston 예측]
    Profile2 --> Apply2[계절 차등 재고<br/>SARIMA 예측]
    Profile3 --> Apply3[적극 재고 확보<br/>Holt-Winters 예측]

    Apply0 --> Monitor[실시간 모니터링<br/>및 업데이트]
    Apply1 --> Monitor
    Apply2 --> Monitor
    Apply3 --> Monitor

    Monitor --> Drift{클러스터<br/>이동 감지?}

    Drift -->|Yes| Reallocate[클러스터 재할당<br/>정책 업데이트]
    Drift -->|No| Continue[현재 정책 유지]

    Reallocate --> Monitor
    Continue --> Monitor

    style Quality fill:#FFD700
    style Monitor fill:#87CEEB
    style Drift fill:#DDA0DD
```

---

## 시각화 프로세스 플로우

```mermaid
flowchart LR
    Data[7차원 특성 데이터] --> PCA[PCA 변환<br/>7D → 2D]

    PCA --> PC1[PC1: 수요 규모<br/>40-60% 설명력]
    PCA --> PC2[PC2: 변동성<br/>20-30% 설명력]

    PC1 --> Plot1[Scatter Plot<br/>클러스터별 색상]
    PC2 --> Plot1

    Data --> Plot2[수요 vs 변동계수<br/>4사분면 분석]
    Data --> Plot3[박스플롯<br/>클러스터별 분포]
    Data --> Plot4[히트맵<br/>특성 프로파일]

    Plot1 --> Save[이미지 저장<br/>outputs/]
    Plot2 --> Save
    Plot3 --> Save
    Plot4 --> Save

    style PCA fill:#87CEEB
    style Save fill:#90EE90
```

---

## 의사결정 지원 플로우

```mermaid
flowchart TD
    NewSKU[신규 SKU 입고] --> Collect[초기 3개월<br/>데이터 수집]

    Collect --> Sufficient{충분한<br/>데이터?}

    Sufficient -->|No| Default[기본 정책 적용<br/>중간 안전재고]
    Sufficient -->|Yes| Extract[특성 추출]

    Extract --> Scale[표준화 변환<br/>기존 scaler 사용]

    Scale --> Predict[클러스터 예측<br/>가장 가까운 중심점]

    Predict --> C0Check{Cluster 0?<br/>안정적 고수요}
    Predict --> C1Check{Cluster 1?<br/>불규칙 저수요}
    Predict --> C2Check{Cluster 2?<br/>계절성}
    Predict --> C3Check{Cluster 3?<br/>급증 추세}

    C0Check -->|Yes| Policy0[EOQ 정책<br/>안전재고 낮음<br/>A존 배치]
    C1Check -->|Yes| Policy1[MTO 정책<br/>안전재고 최소<br/>D존 배치]
    C2Check -->|Yes| Policy2[계절 차등<br/>안전재고 중간<br/>B존 배치]
    C3Check -->|Yes| Policy3[적극 확보<br/>안전재고 높음<br/>C존 배치]

    Default --> Review[3개월 후 재평가]
    Policy0 --> Monitor[지속 모니터링]
    Policy1 --> Monitor
    Policy2 --> Monitor
    Policy3 --> Monitor

    Review --> Sufficient
    Monitor --> Adjust{정책 조정<br/>필요?}

    Adjust -->|Yes| Sufficient
    Adjust -->|No| Monitor

    style NewSKU fill:#90EE90
    style Monitor fill:#87CEEB
    style Adjust fill:#FFD700
```

---

## 평가 및 개선 플로우

```mermaid
flowchart TD
    Result[클러스터링 결과] --> Eval{평가}

    Eval --> Internal[내부 평가<br/>Silhouette, Inertia]
    Eval --> External[외부 평가<br/>비즈니스 지표]

    Internal --> SI[실루엣 점수<br/>0.5-0.7 목표]
    Internal --> CH[Calinski-Harabasz<br/>분리도]
    Internal --> DB[Davies-Bouldin<br/>응집도]

    External --> Accuracy[예측 정확도<br/>MAPE 개선]
    External --> Inventory[재고 회전율<br/>향상]
    External --> Efficiency[운영 효율<br/>증대]

    SI --> Score{목표<br/>달성?}
    CH --> Score
    DB --> Score

    Accuracy --> Business{비즈니스<br/>목표 달성?}
    Inventory --> Business
    Efficiency --> Business

    Score -->|No| Improve1[개선 방안]
    Business -->|No| Improve2[개선 방안]

    Improve1 --> I1[클러스터 개수 조정]
    Improve1 --> I2[알고리즘 변경<br/>GMM, Hierarchical]
    Improve1 --> I3[특성 재설계<br/>새로운 변수 추가]

    Improve2 --> I4[정책 미세 조정<br/>안전재고 레벨]
    Improve2 --> I5[예측 모델 개선<br/>앙상블 적용]
    Improve2 --> I6[동선 최적화<br/>재배치]

    I1 --> Rerun[재실행]
    I2 --> Rerun
    I3 --> Rerun
    I4 --> Rerun
    I5 --> Rerun
    I6 --> Rerun

    Rerun --> Result

    Score -->|Yes| Deploy[배포]
    Business -->|Yes| Deploy

    Deploy --> Production[프로덕션 환경<br/>적용]

    Production --> ContinuousMonitor[지속적 모니터링<br/>월별 재평가]

    style Deploy fill:#90EE90
    style ContinuousMonitor fill:#87CEEB
    style Improve1 fill:#FFB6C1
    style Improve2 fill:#FFB6C1
```

---

## 시스템 아키텍처

```mermaid
flowchart TD
    subgraph Input["📥 입력 레이어"]
        CSV[inbound.csv<br/>입고 데이터]
        Config[설정 파일<br/>n_clusters, method]
    end

    subgraph Processing["⚙️ 처리 레이어"]
        Load[데이터 로더]
        Prep[전처리 엔진]
        Feature[특성 추출 엔진]
        ML[ML 엔진<br/>K-Means / DBSCAN]
    end

    subgraph Analysis["📊 분석 레이어"]
        Profile[프로파일링]
        Visual[시각화]
        Metrics[평가 지표]
    end

    subgraph Output["📤 출력 레이어"]
        Reports[보고서<br/>PDF/HTML]
        Charts[차트<br/>PNG]
        Data[결과 데이터<br/>CSV]
        API[API<br/>실시간 조회]
    end

    CSV --> Load
    Config --> ML

    Load --> Prep
    Prep --> Feature
    Feature --> ML

    ML --> Profile
    ML --> Visual
    ML --> Metrics

    Profile --> Reports
    Visual --> Charts
    Metrics --> Data
    Profile --> API

    style Input fill:#E6F3FF
    style Processing fill:#FFF4E6
    style Analysis fill:#E8F5E9
    style Output fill:#F3E5F5
```

---

## 실시간 모니터링 플로우

```mermaid
flowchart LR
    Live[실시간 입고 데이터] --> Stream[데이터 스트림]

    Stream --> Update{기존 SKU?}

    Update -->|Yes| AddData[데이터 누적<br/>특성 재계산]
    Update -->|No| NewSKU[신규 SKU<br/>초기 클러스터 할당]

    AddData --> Recalc[특성 벡터<br/>업데이트]

    Recalc --> Distance[클러스터 중심까지<br/>거리 계산]

    Distance --> Threshold{이동<br/>임계값<br/>초과?}

    Threshold -->|Yes| Alert[알림 발생<br/>클러스터 이동 감지]
    Threshold -->|No| Stay[현재 클러스터 유지]

    Alert --> Review[관리자 검토]

    Review --> Decision{재할당?}

    Decision -->|Yes| Reassign[클러스터 재할당<br/>정책 변경]
    Decision -->|No| Override[수동 오버라이드]

    Reassign --> Update
    Override --> Stay
    Stay --> Live
    NewSKU --> Live

    style Alert fill:#FFB6C1
    style Review fill:#FFD700
    style Reassign fill:#90EE90
```

---

## 사용 방법

### 1. 기본 플로우 확인
전체 프로세스 플로우를 통해 시스템의 전반적인 흐름을 이해합니다.

### 2. 알고리즘별 상세 플로우
K-Means 또는 DBSCAN의 내부 동작 원리를 확인합니다.

### 3. 비즈니스 활용 플로우
클러스터링 결과를 실제 비즈니스에 적용하는 과정을 파악합니다.

### 4. 의사결정 지원 플로우
신규 SKU에 대한 자동 정책 할당 프로세스를 확인합니다.

---

## Mermaid 차트 렌더링 방법

### VS Code
1. "Markdown Preview Mermaid Support" 확장 설치
2. 마크다운 파일 열기
3. `Cmd/Ctrl + Shift + V`로 미리보기

### GitHub
- GitHub에서 자동으로 Mermaid 차트를 렌더링합니다

### 온라인 에디터
- https://mermaid.live/ 에서 코드 복사 후 렌더링

### Jupyter Notebook
```python
from IPython.display import display, Markdown

mermaid_code = """
```mermaid
flowchart TD
    A --> B
```
"""

display(Markdown(mermaid_code))
```

---

**문서 버전**: 1.0
**최종 업데이트**: 2024-01-XX
**작성자**: Logistics AI Team
