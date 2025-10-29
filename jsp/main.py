"""
로컬 환경에서 수요예측 시스템 실행
사용법: python main.py
"""

from demand_forecast_system import DemandForecastSystem
import pandas as pd
import sys
import os

def main():
    print("="*70)
    print("🚀 입고 데이터 수요예측 시스템")
    print("="*70)

    # 1. 데이터 파일 확인
    data_file = 'shipment.csv'  # 또는 'your_data.csv'

    if not os.path.exists(data_file):
        print(f"\n❌ 오류: '{data_file}' 파일을 찾을 수 없습니다.")
        print("\n다음 중 하나를 수행하세요:")
        print("  1. 'shipment.csv' 파일을 현재 디렉토리에 복사")
        print("  2. 아래 코드에서 data_file 변수를 실제 파일명으로 변경")
        print("\n필수 컬럼: date, sku_code, degr, box_qty")
        sys.exit(1)

    print(f"\n✓ 데이터 파일 발견: {data_file}")

    # 2. 시스템 초기화
    print("\n" + "="*70)
    print("Step 1: 시스템 초기화")
    print("="*70)

    # output_dir을 'outputs'로 설정 (현재 디렉토리에 outputs 폴더 생성)
    dfs = DemandForecastSystem(output_dir='outputs')

    # 3. 데이터 로딩
    print("\n" + "="*70)
    print("Step 2: 데이터 로딩")
    print("="*70)

    try:
        dfs.load_data(data_path=data_file)
    except Exception as e:
        print(f"\n❌ 데이터 로딩 오류: {e}")
        print("\n데이터 파일 형식을 확인하세요:")
        print("  - CSV: .csv 확장자")
        print("  - Excel: .xlsx 확장자")
        print("  - TSV: .txt 확장자")
        sys.exit(1)

    # 4. 데이터 전처리
    print("\n" + "="*70)
    print("Step 3: 데이터 전처리")
    print("="*70)

    try:
        dfs.preprocess_data()
    except Exception as e:
        print(f"\n❌ 전처리 오류: {e}")
        print("\n필수 컬럼을 확인하세요: date, sku_code, degr, box_qty")
        sys.exit(1)

    # 5. 집계 데이터 생성
    print("\n" + "="*70)
    print("Step 4: 집계 데이터 생성")
    print("="*70)
    dfs.create_aggregations()

    # 6. 패턴 분석
    print("\n" + "="*70)
    print("Step 5: 패턴 분석 및 시각화")
    print("="*70)
    dfs.analyze_patterns()

    # 7. SKU 클러스터링
    print("\n" + "="*70)
    print("Step 6: SKU 클러스터링 분석")
    print("="*70)
    dfs.extract_sku_features()
    dfs.perform_sku_clustering(n_clusters=4, method='kmeans')
    dfs.visualize_clusters()

    # 8. 예측 모델 구축
    print("\n" + "="*70)
    print("Step 7: 수요예측 모델 구축")
    print("="*70)
    dfs.build_forecast_models()

    # 9. 모델 평가
    print("\n" + "="*70)
    print("Step 8: 모델 성능 평가")
    print("="*70)
    results = dfs.evaluate_forecasts()

    # 10. 미래 예측
    print("\n" + "="*70)
    print("Step 9: 미래 수요 예측 (향후 7일)")
    print("="*70)
    forecast = dfs.generate_forecast_report(forecast_days=7)

    # 11. 시각화
    print("\n" + "="*70)
    print("Step 10: 예측 결과 시각화")
    print("="*70)
    dfs.visualize_forecast()

    # 12. 종합 리포트
    print("\n" + "="*70)
    print("Step 11: 종합 분석 리포트 생성")
    print("="*70)
    dfs.generate_summary_report()

    # 완료
    print("\n" + "="*70)
    print("✅ 수요예측 완료!")
    print("="*70)
    print(f"\n📂 생성된 파일 위치: ./{dfs.output_dir}/")
    print("\n생성된 파일 목록:")
    print("  📊 시각화 파일:")
    print("    - weekday_pattern.png")
    print("    - temperature_pattern.png")
    print("    - sku_pattern.png")
    print("    - seasonal_pattern.png")
    print("    - sku_clustering.png (NEW!)")
    print("    - cluster_heatmap.png (NEW!)")
    print("    - model_comparison.png")
    print("    - forecast_visualization.png")
    print("\n  📄 데이터 파일:")
    print("    - forecast_report.csv")
    print("    - summary_report.txt")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자가 중단했습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류 발생: {e}")
        print("\n문제가 지속되면 다음을 확인하세요:")
        print("  1. 데이터 파일 형식")
        print("  2. 필수 컬럼 존재 여부")
        print("  3. Python 라이브러리 설치 (pandas, numpy, matplotlib)")
        import traceback
        traceback.print_exc()
        sys.exit(1)