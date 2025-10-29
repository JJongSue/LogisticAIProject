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
    def _load_wms_data():
        """input 폴더의 모든 CSV 파일 로드"""
        input_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")

        data = {
            "inbound": None,
            "inventory": None,
            "outbound": None
        }

        try:
            # 1. Inbound 데이터 로드
            inbound_path = os.path.join(input_dir, "inbound.csv")
            if os.path.exists(inbound_path):
                df_inbound = pd.read_csv(inbound_path, encoding='utf-8-sig')
                df_inbound['date'] = pd.to_datetime(df_inbound['date'])
                data["inbound"] = df_inbound
                print(f"[Pipeline] inbound.csv 로드 완료: {len(df_inbound)}건")

            # 2. Inventory 데이터 로드
            inventory_path = os.path.join(input_dir, "inventory.csv")
            if os.path.exists(inventory_path):
                df_inventory = pd.read_csv(inventory_path, encoding='utf-8-sig')
                data["inventory"] = df_inventory
                print(f"[Pipeline] inventory.csv 로드 완료: {len(df_inventory)}건")

            # 3. Outbound 데이터 로드
            outbound_path = os.path.join(input_dir, "outbound.csv")
            if os.path.exists(outbound_path):
                df_outbound = pd.read_csv(outbound_path, encoding='utf-8-sig')
                df_outbound['date'] = pd.to_datetime(df_outbound['date'])
                data["outbound"] = df_outbound
                print(f"[Pipeline] outbound.csv 로드 완료: {len(df_outbound)}건")

            return data

        except Exception as e:
            print(f"[Pipeline] CSV 파일 로드 실패: {e}")
            return data

    # 전체 데이터를 클래스 변수로 저장
    wms_data = _load_wms_data()


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
        사용자 메시지에서 날짜 또는 날짜 범위를 추출하고 해당 데이터를 반환
        데이터가 많을 경우 요약해서 반환
        """
        date_info = self._extract_date(message)
        print(f"[Pipeline] 추출된 날짜 정보: {date_info}")

        result = {
            "query_description": "",
            "query_date": date_info if isinstance(date_info, str) else date_info.get("description"),
            "target_date": "",
            "inbound": {},
            "inventory": {},
            "outbound": {},
            "data_summary_note": ""
        }

        # 날짜 파싱
        if isinstance(date_info, dict) and date_info.get("type") == "range":
            start_date = pd.to_datetime(date_info["start"])
            end_date = pd.to_datetime(date_info["end"])
            result["query_description"] = date_info["description"]
            result["target_date"] = f"{date_info['start']} ~ {date_info['end']}"
        else:
            target_date = self._parse_date_string(date_info)
            start_date = end_date = pd.to_datetime(target_date) if target_date != "week" else None
            result["query_description"] = target_date
            result["target_date"] = target_date

        # --- Inbound 데이터 처리 ---
        if self.wms_data["inbound"] is not None:
            df_in = self.wms_data["inbound"]
            if start_date is not None:
                df_in_filtered = df_in[(df_in['date'] >= start_date) & (df_in['date'] <= end_date)]
            else:
                df_in_filtered = df_in.tail(5000)  # 최근 5000건

            # 데이터가 많을 경우 샘플 제한
            sample_limit = 3 if len(df_in_filtered) > 10000 else 5

            # 샘플 데이터 준비 (날짜를 문자열로 변환)
            sample_data = []
            if not df_in_filtered.empty:
                df_sample = df_in_filtered.head(sample_limit).copy()
                df_sample['date'] = df_sample['date'].dt.strftime('%Y-%m-%d')
                sample_data = df_sample.to_dict('records')

            # Top SKU도 제한 (데이터 많을 경우 5개로 축소)
            top_limit = 5 if len(df_in_filtered) > 10000 else 10

            result["inbound"] = {
                "total_records": len(df_in_filtered),
                "total_quantity": int(df_in_filtered['in_qty'].sum()) if not df_in_filtered.empty else 0,
                "unique_skus": int(df_in_filtered['sku_code'].nunique()) if not df_in_filtered.empty else 0,
                "unique_containers": int(df_in_filtered['in_cntr'].nunique()) if not df_in_filtered.empty else 0,
                "top_skus": df_in_filtered.groupby('sku_code')['in_qty'].sum().nlargest(top_limit).to_dict() if not df_in_filtered.empty else {},
                "sample_data": sample_data
            }
            print(f"[Pipeline] Inbound 필터링: {len(df_in_filtered)}건")

        # --- Inventory 데이터 처리 ---
        if self.wms_data["inventory"] is not None:
            df_inv = self.wms_data["inventory"]
            # Inventory는 크기가 작으므로 샘플 5개로 제한
            result["inventory"] = {
                "total_skus": len(df_inv),
                "columns": df_inv.columns.tolist(),
                "sample_data": df_inv.head(5).to_dict('records')
            }
            print(f"[Pipeline] Inventory: {len(df_inv)}개 SKU")

        # --- Outbound 데이터 처리 ---
        if self.wms_data["outbound"] is not None:
            df_out = self.wms_data["outbound"]
            if start_date is not None:
                df_out_filtered = df_out[(df_out['date'] >= start_date) & (df_out['date'] <= end_date)]
            else:
                df_out_filtered = df_out.tail(5000)

            # 데이터가 많을 경우 샘플 제한
            sample_limit_out = 3 if len(df_out_filtered) > 10000 else 5

            # 샘플 데이터 준비 (날짜를 문자열로 변환)
            sample_data_out = []
            if not df_out_filtered.empty:
                df_sample_out = df_out_filtered.head(sample_limit_out).copy()
                df_sample_out['date'] = df_sample_out['date'].dt.strftime('%Y-%m-%d')
                sample_data_out = df_sample_out.to_dict('records')

            # 출고 수량 계산 (box_qty 또는 ea_qty 사용)
            total_box_qty = 0
            total_ea_qty = 0
            top_skus = {}

            # Top SKU도 제한 (데이터 많을 경우 5개로 축소)
            top_limit_out = 5 if len(df_out_filtered) > 10000 else 10

            if not df_out_filtered.empty:
                if 'box_qty' in df_out_filtered.columns:
                    total_box_qty = int(df_out_filtered['box_qty'].sum())
                if 'ea_qty' in df_out_filtered.columns:
                    total_ea_qty = int(df_out_filtered['ea_qty'].sum())

                # SKU별 출고 수량 (box_qty 기준)
                if 'sku_code' in df_out_filtered.columns and 'box_qty' in df_out_filtered.columns:
                    top_skus = df_out_filtered.groupby('sku_code')['box_qty'].sum().nlargest(top_limit_out).to_dict()

            result["outbound"] = {
                "total_records": len(df_out_filtered),
                "total_box_qty": total_box_qty,
                "total_ea_qty": total_ea_qty,
                "unique_skus": int(df_out_filtered['sku_code'].nunique()) if not df_out_filtered.empty and 'sku_code' in df_out_filtered.columns else 0,
                "unique_channels": int(df_out_filtered['channel'].nunique()) if not df_out_filtered.empty and 'channel' in df_out_filtered.columns else 0,
                "top_skus": top_skus,
                "sample_data": sample_data_out
            }
            print(f"[Pipeline] Outbound 필터링: {len(df_out_filtered)}건")

        # 데이터 요약 노트 추가
        total_records = result["inbound"].get("total_records", 0) + result["outbound"].get("total_records", 0)
        if total_records > 20000:
            result["data_summary_note"] = f"데이터가 많아 ({total_records:,}건) 통계 요약과 상위 항목만 제공합니다. 샘플 데이터는 축소되었습니다."

        return json.dumps(result, ensure_ascii=False, indent=2)


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

        # 먼저 연도가 명시되어 있는지 확인
        year_match = re.search(r'(\d{4})년', query)
        default_year = int(year_match.group(1)) if year_match else 2024  # 기본 연도를 2024로 변경

        # 날짜 범위 (부터~까지, ~) - M월 전체보다 먼저 체크해야 함!
        # 패턴 1: "M월 D일부터 M월 D일까지" (다른 월 사이 - 먼저 체크)
        match = re.search(r'(\d{1,2})월\s*(\d{1,2})일?\s*(?:부터|에서)\s*(?:~|-)?\s*(\d{1,2})월\s*(\d{1,2})일?(?:까지)?', query)
        if match:
            start_month, start_day, end_month, end_day = match.groups()
            year = default_year
            return {
                "type": "range",
                "start": f"{year}-{int(start_month):02d}-{int(start_day):02d}",
                "end": f"{year}-{int(end_month):02d}-{int(end_day):02d}",
                "description": f"{start_month}월 {start_day}일부터 {end_month}월 {end_day}일까지"
            }

        # 패턴 2: "M월 D일~D일" (같은 월 내)
        match = re.search(r'(\d{1,2})월\s*(\d{1,2})일?\s*(?:~|-)\s*(\d{1,2})일?(?:까지)?', query)
        if match:
            start_month = int(match.group(1))
            start_day = int(match.group(2))
            end_day = int(match.group(3))
            year = default_year
            return {
                "type": "range",
                "start": f"{year}-{start_month:02d}-{start_day:02d}",
                "end": f"{year}-{start_month:02d}-{end_day:02d}",
                "description": f"{start_month}월 {start_day}일부터 {end_day}일까지"
            }

        # 패턴 3: "M월 D일부터 D일까지" (같은 월, 부터~까지 명시)
        match = re.search(r'(\d{1,2})월\s*(\d{1,2})일?\s*부터\s*(\d{1,2})일?(?:까지)?', query)
        if match:
            start_month = int(match.group(1))
            start_day = int(match.group(2))
            end_day = int(match.group(3))
            year = default_year
            return {
                "type": "range",
                "start": f"{year}-{start_month:02d}-{start_day:02d}",
                "end": f"{year}-{start_month:02d}-{end_day:02d}",
                "description": f"{start_month}월 {start_day}일부터 {end_day}일까지"
            }

        # 패턴 4: "M/D ~ M/D"
        match = re.search(r'(\d{1,2})/(\d{1,2})\s*(?:~|-)\s*(\d{1,2})/(\d{1,2})', query)
        if match:
            start_month, start_day, end_month, end_day = match.groups()
            year = default_year
            return {
                "type": "range",
                "start": f"{year}-{int(start_month):02d}-{int(start_day):02d}",
                "end": f"{year}-{int(end_month):02d}-{int(end_day):02d}",
                "description": f"{start_month}/{start_day} ~ {end_month}/{end_day}"
            }

        # M월 전체 데이터 요청 (예: "6월", "6월달", "6월 데이터") - 범위 체크 후 마지막에!
        match = re.search(r'(\d{1,2})월(?:달|간|\s+데이터|\s+중)?', query)
        if match:
            month = int(match.group(1))
            year = default_year
            # 해당 월의 첫날과 마지막 날 계산
            from calendar import monthrange
            last_day = monthrange(year, month)[1]
            return {
                "type": "range",
                "start": f"{year}-{month:02d}-01",
                "end": f"{year}-{month:02d}-{last_day:02d}",
                "description": f"{year}년 {month}월 전체"
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
            year = default_year
            return f"{year}-{int(month):02d}-{int(day):02d}"

        # M월 D일 형식 (예: 6월 1일) - 단일 날짜
        match = re.search(r'(\d{1,2})월\s*(\d{1,2})일?', query)
        if match:
            month, day = match.groups()
            year = default_year
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

WMS 데이터 (Inbound, Inventory, Outbound):
{wms_data}

위 데이터를 바탕으로 사용자의 질문에 구체적이고 명확하게 답변하세요.
- Inbound: 입고 데이터 (입고 수량, SKU, 컨테이너 정보 등)
- Inventory: 재고 현황 데이터 (월별 SKU 재고)
- Outbound: 출고 데이터 (출고 수량, SKU 정보 등)

데이터 분석 시 다음을 포함하세요:
1. 주요 통계 요약
2. 상위 SKU 분석
3. 이상 패턴이나 특이사항 (있는 경우)
4. 추세 및 인사이트
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