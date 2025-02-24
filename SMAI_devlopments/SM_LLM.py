import pandas as pd
from datetime import datetime
import time
import re
from sentence_transformers import SentenceTransformer, util
from kiwipiepy import Kiwi
from SynonymPattern import SynonymManager

# 동의어, 배제어, 범주 데이터를 매핑
class AcademicCalendarRAG:
    def __init__(self, csv_path='hagsailjeong.csv'):
        self.synonym_mgr = SynonymManager()
        self._wait_for_synonyms()
        self.term_db = self.synonym_mgr.get_synonyms()
        self.synonym_map = {term: data['synonyms'] for term, data in self.term_db.items()}
        self.negative_map = {term: data['negatives'] for term, data in self.term_db.items()}
        self.category_map = {term: data.get('category', 'general') for term, data in self.term_db.items()}
        self.df = self._load_data(csv_path)
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.kiwi = Kiwi()
        self._prepare_embeddings()
        self.current_date = pd.Timestamp(datetime.today().date())

    # 동의어 사전 파일이 생성될 때까지 대기
    def _wait_for_synonyms(self):
        retry_count = 0
        while not self.synonym_mgr.synonym_path.exists():
            if retry_count >= 12:
                raise FileNotFoundError("동의어 사전 생성 실패")
            time.sleep(10)
            retry_count += 1

    # CSV 파일에서 학사일정 데이터를 로드하고 현재 연도 이후 데이터만 필터링
    def _load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])
        current_date = pd.Timestamp(datetime.today().date())
        df = df[df['Start'].dt.year >= current_date.year]
        df['document'] = df['Title']
        return df

    # 문서 제목을 임베딩으로 변환
    def _prepare_embeddings(self):
        self.document_embeddings = self.model.encode(
            self.df['document'].tolist(),
            convert_to_tensor=True,
            show_progress_bar=True
        )

    # 질문에서 명사 키워드 추출
    def _extract_keywords(self, query):
        tokens = self.kiwi.tokenize(query)
        keywords = [token.form for token in tokens if token.tag.startswith('NN')] or [query]
        return keywords

    # 질문에서 범주(enrollment, grade 등)를 추론
    def _infer_category(self, query, query_keywords):
        for term, category in self.category_map.items():
            if term in query or any(kw in term for kw in query_keywords):
                return category
        return 'general'

    # 질문과 관련된 문서를 검색하고 유사도 계산
    def _get_relevant_documents(self, query, top_k=10, similarity_threshold=0.3):
        query_keywords = self._extract_keywords(query)
        inferred_category = self._infer_category(query, query_keywords)
        q_lower = query.lower().replace(" ", "")
        query_key = " ".join(query_keywords)

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.document_embeddings)[0].cpu().numpy()

        for i in range(len(similarities)):
            title = self.df.iloc[i]['Title'].lower().replace(" ", "")
            doc_category = self.category_map.get(self.df.iloc[i]['Title'], 'general')

            # 범주가 다르면 유사도에 패널티 적용
            if doc_category != inferred_category:
                similarities[i] *= 0.1
                continue

            # 유사도 가중치 조정
            exact_match = any(kw.lower() == title for kw in query_keywords)
            synonym_match = any(syn in title for syn in self.synonym_map.get(query_key, []))
            keyword_match = any(kw in title for kw in query_keywords)
            negative_terms = self.negative_map.get(query_key, [])
            negative_match = any(nt in title for nt in negative_terms) and '수강신청' in q_lower
            basket_confusion = "장바구니" in title and "수강신청" in q_lower

            if exact_match:
                similarities[i] *= 2.0
            elif synonym_match:
                similarities[i] *= 1.5
            elif keyword_match:
                similarities[i] *= 1.2
            if negative_match:
                similarities[i] *= 0.3
            elif basket_confusion:
                similarities[i] *= 0.5

            # 학기별 월 별로 가중치 적용
            current_month = self.current_date.month
            if "1학기" in self.df.iloc[i]['Title'] and 2 <= current_month <= 7:
                similarities[i] *= 1.02
            elif "2학기" in self.df.iloc[i]['Title'] and (8 <= current_month <= 12 or current_month == 1):
                similarities[i] *= 1.02

        # 상위 문서 필터링
        top_indices = similarities.argsort()[-top_k:][::-1]
        relevant_docs = self.df.iloc[top_indices]
        filtered_docs = relevant_docs[similarities[top_indices] > similarity_threshold]
        return filtered_docs if not filtered_docs.empty else pd.DataFrame()

    # 가장 가까운 이벤트에서 학기 정보 추출
    def _extract_semester(self, events):
        if events.empty:
            return None
        closest_event = events.loc[(events['Start'] - self.current_date).abs().idxmin()]
        start_year = closest_event['Start'].year
        if re.search(r'\d-학기', closest_event['Title']):
            semester = f"{start_year}-{re.search(r'\d-학기', closest_event['Title']).group()}"
        else:
            semester = f"{start_year}-{'1' if closest_event['Start'].month <= 6 else '2'}학기"
        return semester

    # 검색 결과를 시간대별로 포맷팅
    def _format_response(self, events, query):
        response = ["안녕하세요! 관련 학사일정 안내드립니다.\n"]
        semester = self._extract_semester(events)
        if semester:
            events = events[events['Title'].str.contains(semester, na=False)]

        ongoing = events[(events['Start'] <= self.current_date) & (self.current_date <= events['End'])]
        past = events[events['End'] < self.current_date]
        future = events[events['Start'] > self.current_date]

        if not ongoing.empty:
            response.append("### 진행 중인 일정")
            for _, row in ongoing.iterrows():
                response.append(
                    f"- 💡 {row['Title']}\n"
                    f" ▸ 기간: {row['Start'].date()} ~ {row['End'].date()}\n"
                    f" ▸ 상태: 진행 중\n"
                )

        if not past.empty:
            response.append("### 과거 일정")
            for _, row in past.iterrows():
                days_passed = (self.current_date - row['End']).days
                response.append(
                    f"- 🔴 {row['Title']}\n"
                    f" ▸ 기간: {row['Start'].date()} ~ {row['End'].date()}\n"
                    f" ▸ 상태: 종료 (D+{days_passed})\n"
                )

        if not future.empty:
            response.append("### 미래 일정")
            sorted_future = future.sort_values('Start')
            use_highlight = ongoing.empty
            for i, (_, row) in enumerate(sorted_future.iterrows()):
                days_remaining = (row['Start'] - self.current_date).days
                icon = "💡" if use_highlight and i == 0 else "🟢"
                response.append(
                    f"- {icon} {row['Title']}\n"
                    f" ▸ 기간: {row['Start'].date()} ~ {row['End'].date()}\n"
                    f" ▸ 상태: D-{days_remaining} 예정\n"
                )

        response.append("\n※ 정확한 정보는 학사운영팀(📞 02-2287-7077)으로 문의 바랍니다.")
        return "\n".join(response)

    # 질문에 대한 최종 답변 생성
    def get_answer(self, query):
        results = self._get_relevant_documents(query)
        return self._format_response(results, query) if not results.empty else (
            f"⚠️ '{query}' 관련 일정을 찾지 못했습니다.\n"
            f"오타가 있는지 확인해 주시거나 자세한 사항은 학사운영팀(☎ 02-2287-7077)으로 문의해 주세요."
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