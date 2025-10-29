"""
title: WMS Pipeline (Final Fix v3)
author: Your Name
version: 1.0.4
license: MIT
description: 물류 관리 시스템 파이프라인 (valves를 클래스 변수로 초기화)
requirements: pydantic, requests
"""

from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel
import os
import sys
import json
from datetime import datetime, timedelta
import random
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- 1. Valves 정의 (UI 생성용) ---
# 이 클래스는 Open Web UI 설정 화면을 만드는데 사용됩니다.
class Valves(BaseModel):
    """파이프라인 설정값"""
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-oss:latest")
    TEMPERATURE: float = 0.1
    ENABLE_WMS_PROCESSING: bool = True


class Pipeline:
    """WMS 처리 파이프라인"""

    # --- 2. [핵심 수정] valves를 클래스 변수로 직접 생성 ---
    # __init__을 사용하지 않고, 프레임워크 주입에 의존하지도 않고,
    # 클래스가 로드될 때 'valves' 인스턴스를 직접 생성합니다.
    # os.getenv()는 docker-compose.yml의 값을 올바르게 읽어옵니다.
    valves = Valves(
        OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
        MODEL_NAME=os.getenv("MODEL_NAME", "gpt-oss:latest"),
        TEMPERATURE=float(os.getenv("TEMPERATURE", 0.1)),
        ENABLE_WMS_PROCESSING=bool(os.getenv("ENABLE_WMS_PROCESSING", True))
    )

    # --- 3. __init__ 생성자 제거 (필수) ---
    # def __init__(self): ... (이 줄이 없어야 합니다)

    # --- 4. WMS 데이터 로딩 (클래스 변수) ---
    @staticmethod
    def _load_shipment_data():
        """shipment.csv에서 전체 데이터 로드"""
        try:
            # CSV 파일 경로 설정
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "input",
                "shipment.csv"
            )

            # CSV 로드
            df = pd.read_csv(csv_path)
            df['OUTB_DATE'] = pd.to_datetime(df['OUTB_DATE'])

            # 데이터 변환: CSV 형식을 기존 구조와 유사하게 변환
            all_data = []
            for _, row in df.iterrows():
                all_data.append({
                    "order_id": str(row['ORDER_NO']),
                    "date": row['OUTB_DATE'].strftime("%Y-%m-%d"),
                    "datetime": row['OUTB_DATE'].strftime("%Y-%m-%d %H:%M:%S"),
                    "quantity": int(row['JOB_QTY']),
                    "item_code": str(row['ITEM_CD']),
                    "shipto_id": str(row['SHIPTO_ID']),
                    "outbound_type": str(row['OUTB_TCD']),
                    "storage_status": str(row['STRG_STAT']),
                    "status": "completed"  # 기본값
                })

            print(f"[Pipeline] shipment.csv 로드 완료: {len(all_data)}건")
            return all_data

        except Exception as e:
            print(f"[Pipeline] shipment.csv 로드 실패: {e}")
            # 실패 시 빈 리스트 반환
            return []

    # 전체 데이터를 클래스 변수로 저장
    wms_data = _load_shipment_data()


    def pipes(self) -> List[dict]:
        """사용 가능한 파이프라인 목록"""
        return [
            {
                "id": "wms_pipeline",
                "name": "WMS Assistant",
            }
        ]

    # --- 5. [핵심 수정] pipe 시그니처 원복 ---
    # valves 인자 없이, 원래 프레임워크가 호출하는 형태로 되돌립니다.
    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        메시지 처리 파이프라인
        """

        print(f"\n[Pipeline] 메시지 수신: {user_message}")

        # --- 6. [핵심 수정] self.valves 사용 ---
        # 이제 self.valves는 2번 단계에서 생성한 '클래스 변수'를 가리킵니다.
        if self.valves.ENABLE_WMS_PROCESSING and self._is_wms_query(user_message):
            print("[Pipeline] WMS 쿼리 감지")
            wms_data = self._fetch_wms_data(user_message)
            enhanced_message = self._enhance_prompt(user_message, wms_data)
            messages[-1]["content"] = enhanced_message
            print("[Pipeline] WMS 데이터 추가 완료")

        # _call_ollama 호출 시에도 클래스 변수인 self.valves를 전달합니다.
        return self._call_ollama(messages, body, self.valves)

    # --- (헬퍼 함수들, 변경 없음) ---
    def _is_wms_query(self, message: str) -> bool:
        keywords = ["출고", "주문", "배송", "재고", "입고", "이상", "현황", "확인"]
        return any(keyword in message for keyword in keywords)

    def _fetch_wms_data(self, message: str) -> str:
        """
        사용자 메시지에서 날짜 또는 날짜 범위를 추출하고 해당 출고 데이터를 반환
        """
        date_info = self._extract_date(message)
        print(f"[Pipeline] 추출된 날짜 정보: {date_info}")

        # 날짜 필터링
        if isinstance(date_info, dict) and date_info.get("type") == "range":
            # 날짜 범위 필터링
            start_date = date_info["start"]
            end_date = date_info["end"]
            filtered_data = [
                order for order in self.wms_data
                if start_date <= order["date"] <= end_date
            ]
            query_description = date_info["description"]
            target_date = f"{start_date} ~ {end_date}"
        elif date_info == "week":
            # 일주일 데이터 (전체 데이터 반환)
            filtered_data = self.wms_data[-5000:]  # 최근 5000건으로 제한
            query_description = "최근 일주일"
            target_date = "week"
        else:
            # 단일 날짜 필터링
            target_date = self._parse_date_string(date_info)
            filtered_data = [order for order in self.wms_data if order["date"] == target_date]
            query_description = target_date

        print(f"[Pipeline] 필터링된 데이터 건수: {len(filtered_data)}")

        # 통계 계산
        total_quantity = sum(order["quantity"] for order in filtered_data)
        unique_orders = len(set(order["order_id"] for order in filtered_data))
        unique_items = len(set(order["item_code"] for order in filtered_data))
        unique_customers = len(set(order["shipto_id"] for order in filtered_data))

        # 상품별 출고 수량 집계 (상위 10개)
        item_quantities = {}
        for order in filtered_data:
            item_code = order["item_code"]
            item_quantities[item_code] = item_quantities.get(item_code, 0) + order["quantity"]

        top_items = sorted(item_quantities.items(), key=lambda x: x[1], reverse=True)[:10]

        # 이상 탐지
        anomalies = self._detect_anomalies(filtered_data)

        # 결과 구성
        result = {
            "query_description": query_description,
            "query_date": date_info if isinstance(date_info, str) else date_info.get("description"),
            "target_date": target_date,
            "summary": {
                "total_records": len(filtered_data),
                "unique_orders": unique_orders,
                "total_quantity": total_quantity,
                "unique_items": unique_items,
                "unique_customers": unique_customers
            },
            "top_items": [
                {"item_code": item, "total_quantity": qty}
                for item, qty in top_items
            ],
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:5] if anomalies else [],  # 최대 5건만 표시
            "sample_data": filtered_data[:10]  # 샘플 데이터 10건
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _detect_anomalies(self, data: List[dict]) -> List[dict]:
        anomalies = []
        for order in data:
            issues = []
            if order["quantity"] < 0: issues.append("수량이 음수입니다")
            if order["status"] == "delayed": issues.append("배송이 지연되었습니다")
            if issues: anomalies.append({**order, "issues": issues})
        return anomalies

    def _extract_date(self, query: str) -> Union[str, dict]:
        """
        사용자 쿼리에서 날짜 또는 날짜 범위 추출
        지원 형식:
        - 오늘, 어제
        - 일주일, 7일
        - YYYY-MM-DD (2025-06-01)
        - M/D, MM/DD (6/1, 06/01)
        - M월 D일 (6월 1일)
        - M월 전체 (6월, 6월달, 6월 데이터)
        - 기간 범위 (6월 1일부터 6월 30일까지, 6/1~6/30)
        """
        query_lower = query.lower()

        # 상대적 날짜
        if "오늘" in query or "today" in query_lower:
            return "today"
        if "어제" in query or "yesterday" in query_lower:
            return "yesterday"
        if "일주일" in query or "7일" in query or "week" in query_lower:
            return "week"

        # M월 전체 데이터 요청 (예: "6월", "6월달", "6월 데이터")
        match = re.search(r'(\d{1,2})월(?:달|간|\s+데이터|\s+중)?', query)
        if match:
            month = int(match.group(1))
            year = 2025  # 기본 연도
            # 해당 월의 첫날과 마지막 날 계산
            from calendar import monthrange
            last_day = monthrange(year, month)[1]
            return {
                "type": "range",
                "start": f"{year}-{month:02d}-01",
                "end": f"{year}-{month:02d}-{last_day:02d}",
                "description": f"{year}년 {month}월 전체"
            }

        # 날짜 범위 (부터~까지, ~)
        # 패턴 1: "M월 D일부터 M월 D일까지"
        match = re.search(r'(\d{1,2})월\s*(\d{1,2})일?(?:부터|에서)?\s*(?:~|-)?\s*(\d{1,2})월\s*(\d{1,2})일?(?:까지)?', query)
        if match:
            start_month, start_day, end_month, end_day = match.groups()
            year = 2025
            return {
                "type": "range",
                "start": f"{year}-{int(start_month):02d}-{int(start_day):02d}",
                "end": f"{year}-{int(end_month):02d}-{int(end_day):02d}",
                "description": f"{start_month}월 {start_day}일부터 {end_month}월 {end_day}일까지"
            }

        # 패턴 2: "M/D ~ M/D"
        match = re.search(r'(\d{1,2})/(\d{1,2})\s*(?:~|-)\s*(\d{1,2})/(\d{1,2})', query)
        if match:
            start_month, start_day, end_month, end_day = match.groups()
            year = 2025
            return {
                "type": "range",
                "start": f"{year}-{int(start_month):02d}-{int(start_day):02d}",
                "end": f"{year}-{int(end_month):02d}-{int(end_day):02d}",
                "description": f"{start_month}/{start_day} ~ {end_month}/{end_day}"
            }

        # YYYY-MM-DD 형식
        match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', query)
        if match:
            year, month, day = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"

        # M/D 또는 MM/DD 형식 (예: 6/1, 06/01)
        match = re.search(r'(\d{1,2})/(\d{1,2})', query)
        if match:
            month, day = match.groups()
            year = 2025  # 기본 연도
            return f"{year}-{int(month):02d}-{int(day):02d}"

        # M월 D일 형식 (예: 6월 1일)
        match = re.search(r'(\d{1,2})월\s*(\d{1,2})일?', query)
        if match:
            month, day = match.groups()
            year = 2025  # 기본 연도
            return f"{year}-{int(month):02d}-{int(day):02d}"

        return "today"

    def _parse_date_string(self, date_str: str) -> str:
        """날짜 문자열을 YYYY-MM-DD 형식으로 변환"""
        if date_str == "today":
            return datetime.now().strftime("%Y-%m-%d")
        elif date_str == "yesterday":
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        elif date_str == "week":
            return "week"
        return date_str

    def _enhance_prompt(self, user_message: str, wms_data: str) -> str:
        return f"""
당신은 물류 데이터 분석 전문가입니다.
사용자 질문: {user_message}
WMS 데이터:
{wms_data}
위 데이터를 바탕으로 사용자의 질문에 구체적이고 명확하게 답변하세요.
이상이 있다면 각 이상 사항(anomalies)을 자세히 설명하고, 없다면 정상이라고 알려주세요.
"""

    # --- 7. _call_ollama (valves를 인자로 받음) ---
    def _call_ollama(
            self, messages: List[dict], body: dict, valves: Valves
    ) -> Generator:
        """Ollama API 호출 (OpenAI 호환 스트리밍으로 변환)"""
        import requests

        url = f"{valves.OLLAMA_BASE_URL}/api/chat"
        model_id_from_body = body.get("model", "wms_pipeline")
        chunk_id = f"chatcmpl-{model_id_from_body}-{int(time.time())}"

        payload = {
            "model": valves.MODEL_NAME,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": valves.TEMPERATURE
            }
        }

        print(f"[Pipeline] Ollama 호출 (Streaming): {url} (모델: {valves.MODEL_NAME})")

        # 1. 어시스턴트 역할(role) 청크 전송
        try:
            chunk_role = {
                "id": chunk_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id_from_body,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk_role)}\n\n"
        except Exception as e:
            print(f"[Pipeline] Role Chunk Error: {e}")
            yield "data: [DONE]\n\n"
            return

            # 2. Ollama 응답 스트리밍 및 변환
        try:
            with requests.post(url, json=payload, stream=True, timeout=120) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line: continue

                    try:
                        chunk = json.loads(line)

                        if chunk.get("done"):
                            chunk_stop = {
                                "id": chunk_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id_from_body,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                            }
                            yield f"data: {json.dumps(chunk_stop)}\n\n"
                            break

                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            if content:
                                response_chunk = {
                                    "id": chunk_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id_from_body,
                                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(response_chunk)}\n\n"

                    except json.JSONDecodeError:
                        print(f"[Pipeline] Invalid JSON line: {line.decode('utf-8')}")
                        continue

        except Exception as e:
            print(f"[Pipeline 오류] Ollama 호출 실패: {e}")
            error_content = f"Ollama API 호출 중 오류가 발생했습니다: {str(e)}"
            chunk_error = {
                "id": chunk_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id_from_body,
                "choices": [{"index": 0, "delta": {"content": error_content}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(chunk_error)}\n\n"

        # 3. 최종 [DONE] 신호
        yield "data: [DONE]\n\n"