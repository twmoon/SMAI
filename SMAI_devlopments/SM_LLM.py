import pandas as pd
from datetime import datetime
import time
import re
from sentence_transformers import SentenceTransformer, util
from kiwipiepy import Kiwi
from SynonymPattern import SynonymManager

class AcademicCalendarRAG:
    def __init__(self, csv_path='hagsailjeong.csv'):
        self.synonym_mgr = SynonymManager()
        self._wait_for_synonyms()
        self.term_db = self.synonym_mgr.get_synonyms()
        self.synonym_map = {term: data['synonyms'] for term, data in self.term_db.items()}
        self.df = self._load_data(csv_path)
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.kiwi = Kiwi()
        self._prepare_embeddings()
        self.current_date = pd.Timestamp(datetime.today().date())

    def _wait_for_synonyms(self):
        retry_count = 0
        while not self.synonym_mgr.synonym_path.exists():
            if retry_count == 0:
                print("동의어 사전 생성 중... (최대 2분 소요)")
                #self.synonym_mgr._generate_synonyms() #함수명 변경됨
                self.synonym_mgr._generate_terms()
            if retry_count >= 12:
                raise FileNotFoundError("동의어 사전 생성 실패")
            time.sleep(10)
            retry_count += 1
        print("동의어 사전 로드 완료\n")

    def _load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])
        current_date = pd.Timestamp(datetime.today().date())
        print("현재 날짜:", current_date)
        df = df[df['Start'].dt.year >= current_date.year]
        df['document'] = df['Title']
        return df

    def _prepare_embeddings(self):
        self.document_embeddings = self.model.encode(
            self.df['document'].tolist(),
            convert_to_tensor=True,
            show_progress_bar=True
        )

    def _extract_keywords(self, query):
        tokens = self.kiwi.tokenize(query)
        return [token.form for token in tokens if token.tag.startswith('NN')] or [query]

    def _get_relevant_documents(self, query, top_k=10, similarity_threshold=0.3):
        query_keywords = self._extract_keywords(query)
        q_lower = query.lower().replace(" ", "")
        # 유사도 계산
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.document_embeddings)[0].cpu().numpy()
        # 개선된 가중치 시스템
        for i in range(len(similarities)):
            title = self.df.iloc[i]['Title'].lower().replace(" ", "")
            # 긍정 가중치 조건
            exact_match = any(kw.lower() == title for kw in query_keywords)
            synonym_match = any(syn in title for syn in self.synonym_map.get(query, []))
            # 부정 가중치 조건
            negative_match = any(nt in title for nt in self.negative_terms) and '수강신청' in q_lower
            if exact_match:
                similarities[i] *= 2.0
            elif synonym_match:
                similarities[i] *= 1.5
            elif any(kw in title for kw in query_keywords):
                similarities[i] *= 1.2
            if negative_match:
                similarities[i] *= 0.3  # 강한 패널티 적용
            elif "장바구니" in title and "수강신청" in q_lower:
                similarities[i] *= 0.5
        # 결과 필터링
        top_indices = similarities.argsort()[-top_k:][::-1]
        relevant_docs = self.df.iloc[top_indices]
        filtered_docs = relevant_docs[similarities[top_indices] > similarity_threshold]
        return filtered_docs if not filtered_docs.empty else pd.DataFrame()

    def _extract_semester(self, events):
        future_events = events[events['Start'] > self.current_date]
        if future_events.empty:
            return None
        closest_event = future_events.sort_values('Start').iloc[0]
        start_year = closest_event['Start'].year
        if re.search(r'\d-학기', closest_event['Title']):
            return f"{start_year}-{re.search(r'\d-학기', closest_event['Title']).group()}"
        return f"{start_year}-{'1' if closest_event['Start'].month <= 6 else '2'}학기"

    def _format_response(self, events, query):
        response = ["안녕하세요! 관련 학사일정 안내드립니다.\n"]
        # 학기 필터링
        semester = self._extract_semester(events)
        if semester:
            events = events[events['Title'].str.contains(semester, na=False)]
        # 시간대 분류
        past = events[events['End'] < self.current_date]
        future = events[events['Start'] > self.current_date]
        # 과거 일정 처리
        if not past.empty:
            response.append("### 과거 일정")
            for _, row in past.iterrows():
                days_passed = (self.current_date - row['End']).days
                response.append(
                    f"- 🔴 {row['Title']}\n"
                    f" ▸ 기간: {row['Start'].date()} ~ {row['End'].date()}\n"
                    f" ▸ 상태: 종료 (D+{days_passed})\n"
                )
        # 미래 일정 처리
        if not future.empty:
            response.append("### 미래 일정")
            sorted_future = future.sort_values('Start')
            closest = sorted_future.iloc[0]
            for _, row in sorted_future.iterrows():
                days_remaining = (row['Start'] - self.current_date).days
                icon = "💡" if row.equals(closest) else "🟢"
                response.append(
                    f"- {icon} {row['Title']}\n"
                    f" ▸ 기간: {row['Start'].date()} ~ {row['End'].date()}\n"
                    f" ▸ 상태: D-{days_remaining} 예정\n"
                )
        response.append("\n※ 정확한 정보는 학사운영팀(📞 02-2287-7077)으로 문의 바랍니다.")
        return "\n".join(response)

    def get_answer(self, question):
        results = self._get_relevant_documents(question)
        return self._format_response(results, question) if not results.empty else (
            f"⚠️ '{question}' 관련 일정을 찾지 못했습니다.\n"
            f"학사운영팀(☎ 02-2287-7077)으로 문의주시기 바랍니다."
        )

    def main(self):
        print("\n🔍 학사일정 조회 시스템 작동 중...")
        while True:
            query = input("\n질문 (종료: quit): ")
            if query.lower() in ['quit', 'exit']:
                break
            start = time.time()
            print(f"\n{self.get_answer(query)}")
            print(f"\n⏱️ 처리 시간: {time.time() - start:.2f}초")

if __name__ == "__main__":
    AcademicCalendarRAG().main()
