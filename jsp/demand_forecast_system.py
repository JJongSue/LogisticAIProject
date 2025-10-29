"""
입고 데이터 기반 수요예측 시스템
==================================
- 일별/주별 출하량 예측 (sku_code, 온도대별)
- 계절성 및 요일별 패턴 분석
- 온도대별(degr) 수요 변동 예측
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 8)

class DemandForecastSystem:
    """수요예측 시스템 메인 클래스"""

    def __init__(self, data_path=None, output_dir='outputs'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df_raw = None
        self.df_processed = None
        self.daily_agg = None
        self.weekly_agg = None
        self.forecasts = {}
        self.sku_features = None
        self.sku_clusters = None
        self.cluster_labels = None

        # 출력 디렉토리 생성
        import os
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✓ 출력 디렉토리 생성: {self.output_dir}")

    # ==================== Phase 1: 데이터 로딩 및 전처리 ====================

    def load_data(self, df=None, data_path=None):
        """데이터 로딩"""
        # data_path가 파라미터로 전달되면 우선 사용
        if data_path:
            self.data_path = data_path

        if df is not None:
            self.df_raw = df.copy()
        elif self.data_path:
            # CSV, Excel, TSV 등 지원
            if self.data_path.endswith('.csv'):
                self.df_raw = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.xlsx'):
                self.df_raw = pd.read_excel(self.data_path)
            elif self.data_path.endswith('.txt'):
                self.df_raw = pd.read_csv(self.data_path, sep='\t')
            else:
                raise ValueError("지원하지 않는 파일 형식입니다.")
        else:
            raise ValueError("df 또는 data_path를 제공해야 합니다.")

        print(f"✓ 데이터 로딩 완료: {len(self.df_raw)} rows")
        return self

    def preprocess_data(self):
        """데이터 전처리 및 특성 엔지니어링"""
        df = self.df_raw.copy()

        # 1. 날짜 변환
        if df['date'].dtype == 'object':
            df['date'] = pd.to_datetime(df['date'].str.replace('.', '-'))
        else:
            df['date'] = pd.to_datetime(df['date'])

        # 2. 시계열 특성 생성
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek  # 0=월요일, 6=일요일
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter

        # 요일명 (한글)
        weekday_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
        df['weekday_kr'] = df['dayofweek'].map(weekday_map)

        # 3. 계절 분류
        def get_season(month):
            if month in [3, 4, 5]:
                return '봄'
            elif month in [6, 7, 8]:
                return '여름'
            elif month in [9, 10, 11]:
                return '가을'
            else:
                return '겨울'

        df['season'] = df['month'].apply(get_season)

        # 4. 주말/평일 구분
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['day_type'] = df['is_weekend'].map({0: '평일', 1: '주말'})

        # 5. 월초/월말 구분
        df['is_month_start'] = (df['day'] <= 7).astype(int)
        df['is_month_end'] = (df['day'] >= 24).astype(int)

        # 6. 온도대 표준화
        df['temp_category'] = df['degr'].str.strip()

        # 7. 숫자형 데이터 정리
        # 필수 컬럼
        required_numeric_cols = ['box_qty']
        # 선택 컬럼
        optional_numeric_cols = ['pcs_change', 'in_qty', 'plt_change']

        # 필수 컬럼 처리
        for col in required_numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')
            else:
                raise ValueError(f"필수 컬럼 '{col}'이(가) 없습니다.")

        # 선택 컬럼 처리 (없으면 0으로 채움)
        for col in optional_numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')
                df[col] = df[col].fillna(0)
            else:
                df[col] = 0  # 컬럼이 없으면 0으로 생성

        self.df_processed = df
        print(f"✓ 데이터 전처리 완료")
        print(f"  - 기간: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  - SKU 종류: {df['sku_code'].nunique()}개")
        print(f"  - 온도대: {df['temp_category'].unique()}")

        return self

    def create_aggregations(self):
        """일별, 주별 집계 데이터 생성"""
        df = self.df_processed

        # 집계할 컬럼 동적 결정
        agg_cols = {'box_qty': 'sum'}
        if 'pcs_change' in df.columns:
            agg_cols['pcs_change'] = 'sum'
        if 'in_qty' in df.columns:
            agg_cols['in_qty'] = 'sum'
        if 'plt_change' in df.columns:
            agg_cols['plt_change'] = 'sum'

        # 일별 집계 (SKU + 온도대별)
        self.daily_agg = df.groupby(['date', 'sku_code', 'temp_category',
                                     'dayofweek', 'weekday_kr', 'season',
                                     'is_weekend', 'day_type']).agg(agg_cols).reset_index()

        # 컬럼명 변경
        rename_dict = {
            'box_qty': 'daily_boxes',
            'pcs_change': 'daily_pieces',
            'in_qty': 'daily_qty',
            'plt_change': 'daily_pallets'
        }
        self.daily_agg.rename(columns=rename_dict, inplace=True)

        # 주별 집계
        df_weekly = df.copy()
        df_weekly['year_week'] = df_weekly['date'].dt.strftime('%Y-W%W')

        weekly_agg_cols = {'box_qty': 'sum', 'date': 'min'}
        if 'pcs_change' in df.columns:
            weekly_agg_cols['pcs_change'] = 'sum'
        if 'in_qty' in df.columns:
            weekly_agg_cols['in_qty'] = 'sum'

        self.weekly_agg = df_weekly.groupby(['year_week', 'sku_code', 'temp_category']).agg(
            weekly_agg_cols).reset_index()

        # 컬럼명 변경
        weekly_rename_dict = {
            'box_qty': 'weekly_boxes',
            'pcs_change': 'weekly_pieces',
            'in_qty': 'weekly_qty',
            'date': 'week_start_date'
        }
        self.weekly_agg.rename(columns=weekly_rename_dict, inplace=True)

        print(f"✓ 집계 데이터 생성 완료")
        print(f"  - 일별 데이터: {len(self.daily_agg)} rows")
        print(f"  - 주별 데이터: {len(self.weekly_agg)} rows")

        return self

    # ==================== Phase 2: 탐색적 데이터 분석 ====================

    def analyze_patterns(self):
        """패턴 분석 및 시각화"""
        print("\n" + "="*60)
        print("📊 패턴 분석 시작")
        print("="*60)

        # 1. 요일별 패턴
        self._analyze_weekday_pattern()

        # 2. 온도대별 패턴
        self._analyze_temperature_pattern()

        # 3. SKU별 패턴
        self._analyze_sku_pattern()

        # 4. 계절성 패턴
        self._analyze_seasonal_pattern()

        return self

    def _analyze_weekday_pattern(self):
        """요일별 패턴 분석"""
        print("\n[1] 요일별 입고 패턴")

        # 집계할 컬럼 동적 결정
        agg_dict = {'daily_boxes': ['mean', 'sum', 'std']}
        if 'daily_qty' in self.daily_agg.columns:
            agg_dict['daily_qty'] = ['mean', 'sum']

        weekday_pattern = self.daily_agg.groupby('weekday_kr').agg(agg_dict).round(2)

        print(weekday_pattern)

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 평균 박스 수
        weekday_order = ['월', '화', '수', '목', '금', '토', '일']
        weekday_avg = self.daily_agg.groupby('weekday_kr')['daily_boxes'].mean().reindex(weekday_order)

        axes[0].bar(weekday_avg.index, weekday_avg.values, color='steelblue', alpha=0.7)
        axes[0].set_title('Weekday Pattern - Average Boxes', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Day of Week (KR)')
        axes[0].set_ylabel('Average Boxes')
        axes[0].grid(axis='y', alpha=0.3)

        # 평일 vs 주말
        day_type_avg = self.daily_agg.groupby('day_type')['daily_boxes'].mean()
        axes[1].bar(day_type_avg.index, day_type_avg.values, color=['coral', 'lightgreen'], alpha=0.7)
        axes[1].set_title('Weekday vs Weekend', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Day Type')
        axes[1].set_ylabel('Average Boxes')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/weekday_pattern.png', dpi=300, bbox_inches='tight')
        print("  ✓ 시각화 저장: weekday_pattern.png")
        plt.close()

    def _analyze_temperature_pattern(self):
        """온도대별 패턴 분석"""
        print("\n[2] 온도대별 입고 패턴")

        # 집계할 컬럼 동적 결정
        agg_dict = {'daily_boxes': ['sum', 'mean', 'std']}
        if 'daily_qty' in self.daily_agg.columns:
            agg_dict['daily_qty'] = ['sum', 'mean']

        temp_pattern = self.daily_agg.groupby('temp_category').agg(agg_dict).round(2)

        print(temp_pattern)

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 총 박스 수
        temp_total = self.daily_agg.groupby('temp_category')['daily_boxes'].sum().sort_values(ascending=False)
        axes[0].bar(range(len(temp_total)), temp_total.values, color='teal', alpha=0.7)
        axes[0].set_xticks(range(len(temp_total)))
        axes[0].set_xticklabels(temp_total.index, rotation=0)
        axes[0].set_title('Temperature Category - Total Boxes', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Total Boxes')
        axes[0].grid(axis='y', alpha=0.3)

        # 평균 박스 수
        temp_avg = self.daily_agg.groupby('temp_category')['daily_boxes'].mean().sort_values(ascending=False)
        axes[1].bar(range(len(temp_avg)), temp_avg.values, color='darkseagreen', alpha=0.7)
        axes[1].set_xticks(range(len(temp_avg)))
        axes[1].set_xticklabels(temp_avg.index, rotation=0)
        axes[1].set_title('Temperature Category - Average Boxes per Day', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Average Boxes')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/temperature_pattern.png', dpi=300, bbox_inches='tight')
        print("  ✓ 시각화 저장: temperature_pattern.png")
        plt.close()

    def _analyze_sku_pattern(self):
        """SKU별 패턴 분석"""
        print("\n[3] SKU별 입고 패턴 (Top 10)")

        # 집계할 컬럼 동적 결정
        agg_dict = {'daily_boxes': ['sum', 'mean', 'count']}
        if 'daily_qty' in self.daily_agg.columns:
            agg_dict['daily_qty'] = 'sum'

        sku_pattern = self.daily_agg.groupby('sku_code').agg(agg_dict).round(2)

        # 컬럼명 정리
        if 'daily_qty' in self.daily_agg.columns:
            sku_pattern.columns = ['total_boxes', 'avg_boxes', 'days_count', 'total_qty']
        else:
            sku_pattern.columns = ['total_boxes', 'avg_boxes', 'days_count']

        sku_pattern = sku_pattern.sort_values('total_boxes', ascending=False).head(10)

        print(sku_pattern)

        # 시각화
        fig, ax = plt.subplots(figsize=(12, 6))

        top_skus = sku_pattern.head(10)
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_skus)))

        bars = ax.barh(range(len(top_skus)), top_skus['total_boxes'].values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_skus)))
        ax.set_yticklabels(top_skus.index)
        ax.set_xlabel('Total Boxes')
        ax.set_title('Top 10 SKUs by Total Boxes', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # 값 표시
        for i, (idx, row) in enumerate(top_skus.iterrows()):
            ax.text(row['total_boxes'], i, f"  {int(row['total_boxes'])}",
                    va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sku_pattern.png', dpi=300, bbox_inches='tight')
        print("  ✓ 시각화 저장: sku_pattern.png")
        plt.close()

    def _analyze_seasonal_pattern(self):
        """계절별 패턴 분석"""
        print("\n[4] 계절별 입고 패턴")

        seasonal_pattern = self.daily_agg.groupby('season').agg({
            'daily_boxes': ['sum', 'mean', 'count'],
            'daily_qty': 'sum'
        }).round(2)

        print(seasonal_pattern)

        # 시각화
        fig, ax = plt.subplots(figsize=(10, 6))

        season_order = ['봄', '여름', '가을', '겨울']
        season_avg = self.daily_agg.groupby('season')['daily_boxes'].mean().reindex(season_order)

        colors = ['lightgreen', 'gold', 'orange', 'lightblue']
        ax.bar(season_avg.index, season_avg.values, color=colors, alpha=0.7)
        ax.set_title('Seasonal Pattern - Average Boxes', fontsize=14, fontweight='bold')
        ax.set_xlabel('Season')
        ax.set_ylabel('Average Boxes per Day')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/seasonal_pattern.png', dpi=300, bbox_inches='tight')
        print("  ✓ 시각화 저장: seasonal_pattern.png")
        plt.close()

    # ==================== Phase 2.5: SKU 클러스터링 ====================

    def extract_sku_features(self):
        """SKU별 시계열 특성 추출"""
        print("\n" + "="*60)
        print("🔍 SKU별 특성 추출")
        print("="*60)

        sku_features_list = []

        for sku in self.daily_agg['sku_code'].unique():
            sku_data = self.daily_agg[self.daily_agg['sku_code'] == sku].copy()
            sku_data = sku_data.sort_values('date')

            # 기본 통계량
            mean_demand = sku_data['daily_boxes'].mean()
            std_demand = sku_data['daily_boxes'].std()
            max_demand = sku_data['daily_boxes'].max()
            min_demand = sku_data['daily_boxes'].min()
            total_demand = sku_data['daily_boxes'].sum()

            # 변동성 계수 (CV = std / mean)
            cv = std_demand / mean_demand if mean_demand > 0 else 0

            # 요일별 패턴 강도
            weekday_std = sku_data.groupby('dayofweek')['daily_boxes'].mean().std()

            # 계절성 패턴 강도
            seasonal_std = sku_data.groupby('season')['daily_boxes'].mean().std()

            # 트렌드 계산 (선형 회귀 기울기)
            if len(sku_data) > 1:
                x = np.arange(len(sku_data))
                y = sku_data['daily_boxes'].values
                trend = np.polyfit(x, y, 1)[0]
            else:
                trend = 0

            # 간헐적 수요 비율 (0인 날의 비율)
            zero_demand_ratio = (sku_data['daily_boxes'] == 0).sum() / len(sku_data)

            # 주말 vs 평일 비율
            weekend_avg = sku_data[sku_data['is_weekend']==1]['daily_boxes'].mean()
            weekday_avg = sku_data[sku_data['is_weekend']==0]['daily_boxes'].mean()
            weekend_ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 0

            # 온도대 정보
            primary_temp = sku_data['temp_category'].mode()[0] if len(sku_data) > 0 else 'Unknown'

            sku_features_list.append({
                'sku_code': sku,
                'mean_demand': mean_demand,
                'std_demand': std_demand,
                'max_demand': max_demand,
                'min_demand': min_demand,
                'total_demand': total_demand,
                'cv': cv,
                'weekday_pattern_strength': weekday_std,
                'seasonal_pattern_strength': seasonal_std,
                'trend': trend,
                'zero_demand_ratio': zero_demand_ratio,
                'weekend_ratio': weekend_ratio,
                'primary_temp': primary_temp,
                'num_days': len(sku_data)
            })

        self.sku_features = pd.DataFrame(sku_features_list)
        print(f"✓ {len(self.sku_features)}개 SKU 특성 추출 완료")
        print("\n주요 특성:")
        print(self.sku_features[['sku_code', 'mean_demand', 'cv', 'trend']].head(10).to_string(index=False))

        return self

    def perform_sku_clustering(self, n_clusters=4, method='kmeans'):
        """SKU 클러스터링 수행"""
        print("\n" + "="*60)
        print(f"📦 SKU 클러스터링 ({method.upper()})")
        print("="*60)

        if self.sku_features is None:
            print("❌ 먼저 extract_sku_features()를 실행하세요.")
            return self

        # 클러스터링에 사용할 특성 선택
        feature_cols = ['mean_demand', 'cv', 'weekday_pattern_strength',
                        'seasonal_pattern_strength', 'trend', 'zero_demand_ratio',
                        'weekend_ratio']

        X = self.sku_features[feature_cols].fillna(0)

        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 클러스터링 수행
        if method.lower() == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(X_scaled)
        elif method.lower() == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            labels = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        else:
            print(f"❌ 지원하지 않는 방법: {method}")
            return self

        self.cluster_labels = labels
        self.sku_features['cluster'] = labels

        # 클러스터별 통계
        print(f"\n✓ 클러스터링 완료: {n_clusters}개 클러스터")
        print("\n클러스터별 SKU 수:")
        print(self.sku_features['cluster'].value_counts().sort_index())

        # 클러스터별 특성 요약
        print("\n클러스터별 평균 특성:")
        cluster_summary = self.sku_features.groupby('cluster')[feature_cols].mean().round(2)
        print(cluster_summary)

        # 클러스터 해석
        self._interpret_clusters()

        return self

    def _interpret_clusters(self):
        """클러스터 특성 해석"""
        print("\n" + "-"*60)
        print("클러스터 해석:")
        print("-"*60)

        for cluster_id in sorted(self.sku_features['cluster'].unique()):
            cluster_data = self.sku_features[self.sku_features['cluster'] == cluster_id]

            print(f"\n[Cluster {cluster_id}] - {len(cluster_data)}개 SKU")

            avg_demand = cluster_data['mean_demand'].mean()
            avg_cv = cluster_data['cv'].mean()
            avg_trend = cluster_data['trend'].mean()
            avg_zero_ratio = cluster_data['zero_demand_ratio'].mean()

            # 클러스터 특성 분류
            if avg_demand > self.sku_features['mean_demand'].median():
                demand_level = "고수요"
            else:
                demand_level = "저수요"

            if avg_cv > 1.0:
                variability = "변동성 높음"
            elif avg_cv > 0.5:
                variability = "변동성 중간"
            else:
                variability = "변동성 낮음"

            if avg_trend > 0:
                trend_type = "증가 추세"
            elif avg_trend < 0:
                trend_type = "감소 추세"
            else:
                trend_type = "안정적"

            print(f"  특성: {demand_level}, {variability}, {trend_type}")
            print(f"  평균 수요: {avg_demand:.1f} boxes/day")
            print(f"  변동계수: {avg_cv:.2f}")
            print(f"  트렌드: {avg_trend:.3f}")
            print(f"  간헐적 수요 비율: {avg_zero_ratio:.1%}")

            # 대표 SKU
            representative_skus = cluster_data.nlargest(3, 'total_demand')['sku_code'].tolist()
            print(f"  대표 SKU: {', '.join(map(str, representative_skus))}")

    def visualize_clusters(self):
        """클러스터 시각화"""
        if self.sku_features is None or 'cluster' not in self.sku_features.columns:
            print("❌ 먼저 클러스터링을 수행하세요.")
            return self

        print("\n클러스터 시각화 생성 중...")

        # PCA로 2D 축소
        feature_cols = ['mean_demand', 'cv', 'weekday_pattern_strength',
                        'seasonal_pattern_strength', 'trend', 'zero_demand_ratio',
                        'weekend_ratio']

        X = self.sku_features[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        self.sku_features['pca1'] = X_pca[:, 0]
        self.sku_features['pca2'] = X_pca[:, 1]

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. PCA 클러스터 시각화
        scatter = axes[0, 0].scatter(self.sku_features['pca1'],
                                     self.sku_features['pca2'],
                                     c=self.sku_features['cluster'],
                                     cmap='viridis',
                                     s=100,
                                     alpha=0.6,
                                     edgecolors='black')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0, 0].set_title('SKU Clustering (PCA)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 평균 수요 vs 변동계수
        for cluster_id in sorted(self.sku_features['cluster'].unique()):
            cluster_data = self.sku_features[self.sku_features['cluster'] == cluster_id]
            axes[0, 1].scatter(cluster_data['mean_demand'],
                              cluster_data['cv'],
                              label=f'Cluster {cluster_id}',
                              s=80,
                              alpha=0.6)
        axes[0, 1].set_xlabel('Average Demand (boxes/day)')
        axes[0, 1].set_ylabel('Coefficient of Variation')
        axes[0, 1].set_title('Demand vs Variability', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 클러스터별 박스플롯 (평균 수요)
        cluster_demands = [self.sku_features[self.sku_features['cluster']==c]['mean_demand'].values
                          for c in sorted(self.sku_features['cluster'].unique())]
        bp = axes[1, 0].boxplot(cluster_demands,
                                labels=[f'C{c}' for c in sorted(self.sku_features['cluster'].unique())],
                                patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Average Demand (boxes/day)')
        axes[1, 0].set_title('Demand Distribution by Cluster', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. 클러스터별 SKU 개수
        cluster_counts = self.sku_features['cluster'].value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
        axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of SKUs')
        axes[1, 1].set_title('SKU Count per Cluster', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)

        # 값 표시
        for i, (cluster, count) in enumerate(cluster_counts.items()):
            axes[1, 1].text(cluster, count, f'  {count}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sku_clustering.png', dpi=300, bbox_inches='tight')
        print("✓ 클러스터 시각화 저장: sku_clustering.png")
        plt.close()

        # 클러스터별 특성 히트맵
        self._plot_cluster_heatmap()

        return self

    def _plot_cluster_heatmap(self):
        """클러스터별 특성 히트맵"""
        feature_cols = ['mean_demand', 'cv', 'weekday_pattern_strength',
                        'seasonal_pattern_strength', 'trend', 'zero_demand_ratio',
                        'weekend_ratio']

        cluster_summary = self.sku_features.groupby('cluster')[feature_cols].mean()

        # 정규화 (0-1 스케일)
        cluster_summary_norm = (cluster_summary - cluster_summary.min()) / (cluster_summary.max() - cluster_summary.min())

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(cluster_summary_norm.T, annot=True, fmt='.2f', cmap='YlGnBu',
                    cbar_kws={'label': 'Normalized Value'},
                    linewidths=0.5, linecolor='gray')
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title('Cluster Feature Heatmap (Normalized)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cluster_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ 클러스터 히트맵 저장: cluster_heatmap.png")
        plt.close()

    # ==================== Phase 3: 수요예측 모델링 ====================

    def build_forecast_models(self):
        """수요예측 모델 구축"""
        print("\n" + "="*60)
        print("🔮 수요예측 모델 구축")
        print("="*60)

        # 1. 이동평균 기반 예측
        self._forecast_moving_average()

        # 2. 지수평활 기반 예측
        self._forecast_exponential_smoothing()

        # 3. 요일 패턴 기반 예측
        self._forecast_weekday_pattern()

        return self

    def _forecast_moving_average(self, window=7):
        """이동평균 기반 예측"""
        print(f"\n[1] 이동평균 예측 (Window={window}일)")

        # SKU별로 예측
        forecasts = []

        for sku in self.daily_agg['sku_code'].unique()[:5]:  # 상위 5개 SKU만
            sku_data = self.daily_agg[self.daily_agg['sku_code'] == sku].copy()
            sku_data = sku_data.sort_values('date')

            # 이동평균 계산
            sku_data['ma_7d'] = sku_data['daily_boxes'].rolling(window=window, min_periods=1).mean()
            sku_data['forecast'] = sku_data['ma_7d'].shift(1)

            forecasts.append(sku_data)

        self.forecasts['moving_average'] = pd.concat(forecasts, ignore_index=True)
        print("  ✓ 이동평균 예측 완료")

    def _forecast_exponential_smoothing(self, alpha=0.3):
        """지수평활 기반 예측"""
        print(f"\n[2] 지수평활 예측 (Alpha={alpha})")

        forecasts = []

        for sku in self.daily_agg['sku_code'].unique()[:5]:
            sku_data = self.daily_agg[self.daily_agg['sku_code'] == sku].copy()
            sku_data = sku_data.sort_values('date')

            # 지수평활
            sku_data['ema'] = sku_data['daily_boxes'].ewm(alpha=alpha, adjust=False).mean()
            sku_data['forecast'] = sku_data['ema'].shift(1)

            forecasts.append(sku_data)

        self.forecasts['exponential_smoothing'] = pd.concat(forecasts, ignore_index=True)
        print("  ✓ 지수평활 예측 완료")

    def _forecast_weekday_pattern(self):
        """요일 패턴 기반 예측"""
        print("\n[3] 요일 패턴 기반 예측")

        # 각 SKU + 요일 조합의 평균 사용
        weekday_avg = self.daily_agg.groupby(['sku_code', 'dayofweek'])['daily_boxes'].mean().reset_index()
        weekday_avg.columns = ['sku_code', 'dayofweek', 'weekday_forecast']

        # 원본 데이터에 병합
        forecast_df = self.daily_agg.merge(weekday_avg, on=['sku_code', 'dayofweek'], how='left')

        self.forecasts['weekday_pattern'] = forecast_df
        print("  ✓ 요일 패턴 예측 완료")

    # ==================== Phase 4: 결과 평가 및 시각화 ====================

    def evaluate_forecasts(self):
        """예측 성능 평가"""
        print("\n" + "="*60)
        print("📈 예측 성능 평가")
        print("="*60)

        results = []

        for model_name, forecast_df in self.forecasts.items():
            if 'forecast' in forecast_df.columns:
                # 결측치 제거
                eval_df = forecast_df.dropna(subset=['forecast', 'daily_boxes'])

                if len(eval_df) > 0:
                    actual = eval_df['daily_boxes']
                    predicted = eval_df['forecast']

                    # 평가지표 계산
                    mae = np.mean(np.abs(actual - predicted))
                    rmse = np.sqrt(np.mean((actual - predicted)**2))
                    mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100

                    results.append({
                        'Model': model_name,
                        'MAE': round(mae, 2),
                        'RMSE': round(rmse, 2),
                        'MAPE(%)': round(mape, 2)
                    })

            elif 'weekday_forecast' in forecast_df.columns:
                eval_df = forecast_df.dropna(subset=['weekday_forecast', 'daily_boxes'])

                if len(eval_df) > 0:
                    actual = eval_df['daily_boxes']
                    predicted = eval_df['weekday_forecast']

                    mae = np.mean(np.abs(actual - predicted))
                    rmse = np.sqrt(np.mean((actual - predicted)**2))
                    mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100

                    results.append({
                        'Model': model_name,
                        'MAE': round(mae, 2),
                        'RMSE': round(rmse, 2),
                        'MAPE(%)': round(mape, 2)
                    })

        results_df = pd.DataFrame(results)
        print("\n모델 성능 비교:")
        print(results_df.to_string(index=False))

        # 성능 비교 시각화
        if len(results_df) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            metrics = ['MAE', 'RMSE', 'MAPE(%)']
            colors = ['skyblue', 'lightcoral', 'lightgreen']

            for idx, (metric, color) in enumerate(zip(metrics, colors)):
                axes[idx].bar(results_df['Model'], results_df[metric], color=color, alpha=0.7)
                axes[idx].set_title(f'{metric} Comparison', fontweight='bold')
                axes[idx].set_xlabel('Model')
                axes[idx].set_ylabel(metric)
                axes[idx].tick_params(axis='x', rotation=15)
                axes[idx].grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
            print("\n✓ 모델 비교 시각화 저장: model_comparison.png")
            plt.close()

        return results_df

    def generate_forecast_report(self, forecast_days=7):
        """미래 예측 리포트 생성"""
        print("\n" + "="*60)
        print(f"📋 향후 {forecast_days}일 수요 예측")
        print("="*60)

        # 최근 데이터 기반으로 예측
        latest_date = self.daily_agg['date'].max()

        # 요일 패턴 기반 예측
        weekday_avg = self.daily_agg.groupby(['sku_code', 'temp_category', 'dayofweek']).agg({
            'daily_boxes': 'mean',
            'daily_qty': 'mean'
        }).reset_index()

        # 미래 날짜 생성
        future_dates = pd.date_range(start=latest_date + timedelta(days=1),
                                     periods=forecast_days, freq='D')

        forecast_list = []

        for sku in self.daily_agg['sku_code'].unique()[:3]:  # Top 3 SKU
            for temp in self.daily_agg[self.daily_agg['sku_code']==sku]['temp_category'].unique():
                for date in future_dates:
                    dow = date.dayofweek

                    # 해당 SKU+온도대+요일의 평균 사용
                    forecast_val = weekday_avg[
                        (weekday_avg['sku_code'] == sku) &
                        (weekday_avg['temp_category'] == temp) &
                        (weekday_avg['dayofweek'] == dow)
                        ]['daily_boxes'].values

                    if len(forecast_val) > 0:
                        weekday_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
                        forecast_list.append({
                            'date': date,
                            'weekday': weekday_map[dow],
                            'sku_code': sku,
                            'temp_category': temp,
                            'forecast_boxes': round(forecast_val[0], 1)
                        })

        forecast_df = pd.DataFrame(forecast_list)

        print(f"\n향후 {forecast_days}일 예측 (Top 3 SKU):")
        print(forecast_df.head(20).to_string(index=False))

        # 예측 결과 저장
        forecast_df.to_csv(f'{self.output_dir}/forecast_report.csv', index=False, encoding='utf-8-sig')
        print(f"\n✓ 예측 리포트 저장: forecast_report.csv")

        return forecast_df

    def visualize_forecast(self, sku_code=None, days_back=30):
        """예측 결과 시각화"""
        if sku_code is None:
            # 가장 많이 입고된 SKU 선택
            sku_code = self.daily_agg.groupby('sku_code')['daily_boxes'].sum().idxmax()

        print(f"\n시각화 대상 SKU: {sku_code}")

        # 해당 SKU 데이터
        sku_data = self.daily_agg[self.daily_agg['sku_code'] == sku_code].copy()
        sku_data = sku_data.sort_values('date')

        # 최근 N일 데이터
        recent_date = sku_data['date'].max()
        cutoff_date = recent_date - timedelta(days=days_back)
        sku_recent = sku_data[sku_data['date'] >= cutoff_date]

        # 이동평균 추가
        sku_recent['ma_7d'] = sku_recent['daily_boxes'].rolling(window=7, min_periods=1).mean()

        # 시각화
        fig, ax = plt.subplots(figsize=(15, 6))

        ax.plot(sku_recent['date'], sku_recent['daily_boxes'],
                marker='o', label='Actual', linewidth=2, markersize=6, color='steelblue')
        ax.plot(sku_recent['date'], sku_recent['ma_7d'],
                label='7-day MA', linewidth=2, linestyle='--', color='coral')

        ax.set_title(f'Demand Forecast for SKU {sku_code} (Recent {days_back} days)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Daily Boxes')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/forecast_visualization.png', dpi=300, bbox_inches='tight')
        print("✓ 예측 시각화 저장: forecast_visualization.png")
        plt.close()

    def generate_summary_report(self):
        """종합 분석 리포트 생성"""
        print("\n" + "="*60)
        print("📄 종합 분석 리포트")
        print("="*60)

        summary = []

        # 1. 기본 통계
        summary.append("=" * 60)
        summary.append("1. 데이터 기본 정보")
        summary.append("=" * 60)
        summary.append(f"분석 기간: {self.df_processed['date'].min()} ~ {self.df_processed['date'].max()}")
        summary.append(f"총 데이터 수: {len(self.df_processed):,} rows")
        summary.append(f"SKU 종류: {self.df_processed['sku_code'].nunique()} 개")
        summary.append(f"온도대: {', '.join(self.df_processed['temp_category'].unique())}")
        summary.append(f"총 입고 박스: {self.df_processed['box_qty'].sum():,.0f} boxes")
        summary.append("")

        # 2. 요일별 인사이트
        summary.append("=" * 60)
        summary.append("2. 요일별 패턴 분석")
        summary.append("=" * 60)
        weekday_avg = self.daily_agg.groupby('weekday_kr')['daily_boxes'].mean()
        max_day = weekday_avg.idxmax()
        min_day = weekday_avg.idxmin()
        summary.append(f"최대 입고 요일: {max_day} ({weekday_avg[max_day]:.1f} boxes/day)")
        summary.append(f"최소 입고 요일: {min_day} ({weekday_avg[min_day]:.1f} boxes/day)")

        weekend_avg = self.daily_agg[self.daily_agg['is_weekend']==1]['daily_boxes'].mean()
        weekday_avg_val = self.daily_agg[self.daily_agg['is_weekend']==0]['daily_boxes'].mean()
        summary.append(f"평일 평균: {weekday_avg_val:.1f} boxes/day")
        summary.append(f"주말 평균: {weekend_avg:.1f} boxes/day")
        summary.append("")

        # 3. 온도대별 인사이트
        summary.append("=" * 60)
        summary.append("3. 온도대별 패턴 분석")
        summary.append("=" * 60)
        temp_total = self.daily_agg.groupby('temp_category')['daily_boxes'].sum().sort_values(ascending=False)
        for temp, val in temp_total.items():
            pct = (val / temp_total.sum()) * 100
            summary.append(f"{temp}: {val:,.0f} boxes ({pct:.1f}%)")
        summary.append("")

        # 4. Top SKU
        summary.append("=" * 60)
        summary.append("4. Top 5 SKU")
        summary.append("=" * 60)
        top_skus = self.daily_agg.groupby('sku_code')['daily_boxes'].sum().sort_values(ascending=False).head(5)
        for rank, (sku, val) in enumerate(top_skus.items(), 1):
            summary.append(f"{rank}. SKU {sku}: {val:,.0f} boxes")
        summary.append("")

        # 5. 추천사항
        summary.append("=" * 60)
        summary.append("5. 수요예측 기반 추천사항")
        summary.append("=" * 60)
        summary.append(f"• {max_day}요일은 입고량이 가장 많으므로 충분한 인력 및 공간 확보 필요")
        summary.append(f"• 주말 입고량이 평일 대비 {'높으므로' if weekend_avg > weekday_avg_val else '낮으므로'} 주말 운영 전략 검토")
        summary.append(f"• {temp_total.index[0]} 상품이 전체의 {(temp_total.iloc[0]/temp_total.sum()*100):.1f}%를 차지하므로 별도 관리 필요")
        summary.append("")

        # 리포트 출력 및 저장
        report_text = "\n".join(summary)
        print(report_text)

        with open(f'{self.output_dir}/summary_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print("\n✓ 종합 리포트 저장: summary_report.txt")

        return report_text


# ==================== 메인 실행 함수 ====================

def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("🚀 수요예측 시스템 시작")
    print("="*60)

    # 1. 시스템 초기화
    dfs = DemandForecastSystem()

    # 2. 데이터 로딩 (실제 파일 경로로 변경 필요)
    # dfs.load_data(data_path='your_data.csv')

    print("\n샘플 데이터로 시연합니다.")
    print("실제 데이터를 사용하려면 load_data() 메서드에 파일 경로를 지정하세요.")

    return dfs


if __name__ == "__main__":
    dfs = main()
    print("\n사용 예시:")
    print("  dfs.load_data(df=your_dataframe)")
    print("  dfs.preprocess_data()")
    print("  dfs.create_aggregations()")
    print("  dfs.analyze_patterns()")
    print("  dfs.build_forecast_models()")
    print("  dfs.evaluate_forecasts()")
    print("  dfs.generate_forecast_report()")
    print("  dfs.generate_summary_report()")