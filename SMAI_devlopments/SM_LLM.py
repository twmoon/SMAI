
# -*- coding: utf-8 -*-
import os
import pandas as pd
from datetime import datetime
from rapidfuzz import process, fuzz
from metaphone import doublemetaphone
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


class AcademicChatbot:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.current_date = datetime(2025, 2, 23)
        self.synonym_map = {
            '등록금': ['등록'],
            '졸업식': ['학위수여식'],
            '수강신청': ['장바구니', '교차수강', '수강정정'],
            '개강총회': ['개강', '학사일정회의'],
            '중간고사': ['중간시험', '중간평가'],
            '기말고사': ['기말시험', '기말평가']
        }

        self._preprocess_dates()
        self._add_status_column()

    def _preprocess_dates(self):
        """날짜 형식 변환 및 기간 계산"""
        self.df['Start'] = pd.to_datetime(self.df['Start'])
        self.df['End'] = pd.to_datetime(self.df['End'])
        self.df['Duration'] = (self.df['End'] - self.df['Start']).dt.days + 1

    def _add_status_column(self):
        """현재 날짜 기준 상태 컬럼 추가"""
        self.df['Status'] = self.df.apply(
            lambda row: self._get_schedule_status(row['Start'], row['End']), axis=1
        )

    def _get_schedule_status(self, start, end):
        """일정 상태 판별 로직"""
        if self.current_date < start:
            delta = (start - self.current_date).days
            return f"D-{delta} 예정"
        elif start <= self.current_date <= end:
            return "진행 중"
        else:
            delta = (self.current_date - end).days
            return f"종료 (D+{delta})"

    def _expand_query(self, query):
        """동적 쿼리 확장 시스템"""
        # 1단계: 직접 매칭
        if query in self.synonym_map:
            return self.synonym_map[query]

        # 2단계: 음운론적 유사도
        terms = list(self.synonym_map.keys()) + [item for sublist in self.synonym_map.values() for item in sublist]
        best_match = process.extractOne(query, terms, scorer=fuzz.WRatio)

        return self.synonym_map.get(best_match[0], [query]) if best_match[1] > 80 else [query]

    def _find_related_events(self, terms):
        """관련 이벤트 검색 엔진"""
        results = pd.DataFrame()
        for term in terms:
            mask = self.df['Title'].str.contains(term)
            results = pd.concat([results, self.df[mask]])
        return results.drop_duplicates().sort_values(by='Start')

    def _format_response(self, events):
        """이벤트 포맷팅"""
        response = []
        for _, row in events.iterrows():
            emoji = "🟢" if "예정" in row['Status'] else "🟡" if "진행" in row['Status'] else "🔴"
            response.append(
                f"{emoji} {row['Title']}\n"
                f"   ▸ 기간: {row['Start'].strftime('%Y-%m-%d')} ~ {row['End'].strftime('%Y-%m-%d')}\n"
                f"   ▸ 상태: {row['Status']}\n"
            )
        return "\n".join(response)

    def generate_answer(self, query):
        """Ollama 기반 지능형 응답 생성"""
        # 1. 쿼리 확장
        expanded_terms = self._expand_query(query)

        # 2. 이벤트 검색
        events = self._find_related_events(expanded_terms)

        # 3. 응답 생성
        if not events.empty:
            formatted_events = self._format_response(events)
            prompt = f"""
            [현재 날짜] 2025-02-23
            [사용자 질문] {query}
            [검색 결과]
            {formatted_events}

            [생성 규칙]
            1. 친절한 어조로 반말 금지
            2. 모든 이벤트 번호 없이 자연스럽게 나열
            3. 가장 가까운 일정은 💡로 강조
            4. 지난 일정은 회색 이모지 사용
            5. 정확하지 않은 정보는 학사행정팀 안내
            """

            llm = ChatOllama(
                model="exaone3.5:latest",
                temperature=0.2,
                num_ctx=4096
            )
            return llm.invoke(prompt).content

        return f"⚠️ 관련 일정을 찾을 수 없습니다. 학사행정팀(02-1234-5678)으로 문의해주세요."

    def run(self):
        """실행 인터페이스"""
        print("상명대학교 학사안내 챗봇 서비스 시작\n")
        while True:
            try:
                query = input("질문을 입력하세요 (종료: exit): ")
                if query.lower() == 'exit':
                    break
                print(f"\n{self.generate_answer(query)}\n")
            except Exception as e:
                print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    chatbot = AcademicChatbot("hagsailjeong.csv")
    chatbot.run()
