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
        self.negative_map = {term: data['negatives'] for term, data in self.term_db.items()}
        self.df = self._load_data(csv_path)
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.kiwi = Kiwi()
        self._prepare_embeddings()
        self.current_date = pd.Timestamp(datetime.today().date())
        print(f"초기화 완료 - 현재 날짜: {self.current_date}")

    def _wait_for_synonyms(self):
        retry_count = 0
        while not self.synonym_mgr.synonym_path.exists():
            if retry_count == 0:
                print("동의어 사전 생성 중... (최대 2분 소요)")
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
        df = df[df['Start'].dt.year >= current_date.year]
        df['document'] = df['Title']
        print(f"데이터 로드 완료 - {len(df)}개의 항목")
        return df

    def _prepare_embeddings(self):
        self.document_embeddings = self.model.encode(
            self.df['document'].tolist(),
            convert_to_tensor=True,
            show_progress_bar=True
        )
        print("임베딩 준비 완료")

    def _extract_keywords(self, query):
        tokens = self.kiwi.tokenize(query)
        keywords = [token.form for token in tokens if token.tag.startswith('NN')] or [query]
        print(f"추출된 키워드: {keywords}")
        return keywords

    def _get_relevant_documents(self, query, top_k=10, similarity_threshold=0.3):
        print(f"\n=== 질문 처리 시작: '{query}' ===")
        query_keywords = self._extract_keywords(query)
        q_lower = query.lower().replace(" ", "")
        print(f"소문자 변환 질문: {q_lower}")
        query_key = " ".join(query_keywords)
        print(f"쿼리 키: {query_key}")
        print(f"동의어: {self.synonym_map.get(query_key, [])}")
        print(f"배제어: {self.negative_map.get(query_key, [])}")

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.document_embeddings)[0].cpu().numpy()
        print(f"초기 유사도 계산 완료 - {len(similarities)}개 문서")

        for i in range(len(similarities)):
            title = self.df.iloc[i]['Title'].lower().replace(" ", "")
            original_similarity = similarities[i]
            print(f"\n문서 {i}: {self.df.iloc[i]['Title']}")
            print(f"  초기 유사도: {original_similarity:.4f}")

            exact_match = any(kw.lower() == title for kw in query_keywords)
            synonym_match = any(syn in title for syn in self.synonym_map.get(query_key, []))
            keyword_match = any(kw in title for kw in query_keywords)
            negative_terms = self.negative_map.get(query_key, [])
            negative_match = any(nt in title for nt in negative_terms) and '수강신청' in q_lower
            basket_confusion = "장바구니" in title and "수강신청" in q_lower

            if exact_match:
                similarities[i] *= 2.0
                print(f"  정확 매칭 적용 (×2.0): {similarities[i]:.4f}")
            elif synonym_match:
                similarities[i] *= 1.5
                print(f"  동의어 매칭 적용 (×1.5): {similarities[i]:.4f}")
            elif keyword_match:
                similarities[i] *= 1.2
                print(f"  키워드 포함 적용 (×1.2): {similarities[i]:.4f}")
            if negative_match:
                similarities[i] *= 0.3
                print(f"  배제어 패널티 적용 (×0.3): {similarities[i]:.4f}")
            elif basket_confusion:
                similarities[i] *= 0.5
                print(f"  장바구니 혼동 패널티 적용 (×0.5): {similarities[i]:.4f}")

            current_month = self.current_date.month
            if "1학기" in self.df.iloc[i]['Title'] and 2 <= current_month <= 7:
                similarities[i] *= 1.02
                print(f"  1학기 가중치 적용 (×1.02, {current_month}월): {similarities[i]:.4f}")
            elif "2학기" in self.df.iloc[i]['Title'] and (8 <= current_month <= 12 or current_month == 1):
                similarities[i] *= 1.02
                print(f"  2학기 가중치 적용 (×1.02, {current_month}월): {similarities[i]:.4f}")

            print(f"  최종 유사도: {similarities[i]:.4f}")

        top_indices = similarities.argsort()[-top_k:][::-1]
        print(f"\n상위 {top_k}개 인덱스: {top_indices}")
        print(f"상위 유사도: {[f'{similarities[idx]:.4f}' for idx in top_indices]}")
        relevant_docs = self.df.iloc[top_indices]
        filtered_docs = relevant_docs[similarities[top_indices] > similarity_threshold]
        print(f"필터링된 문서 수: {len(filtered_docs)} (임계값: {similarity_threshold})")
        for _, row in filtered_docs.iterrows():
            print(f"  - {row['Title']} (유사도: {similarities[top_indices[list(relevant_docs.index).index(row.name)]]:.4f})")
        print("=== 질문 처리 종료 ===\n")
        return filtered_docs if not filtered_docs.empty else pd.DataFrame()

    def _extract_semester(self, events):
        if events.empty:
            return None
        closest_event = events.loc[(events['Start'] - self.current_date).abs().idxmin()]
        start_year = closest_event['Start'].year
        if re.search(r'\d-학기', closest_event['Title']):
            semester = f"{start_year}-{re.search(r'\d-학기', closest_event['Title']).group()}"
        else:
            semester = f"{start_year}-{'1' if closest_event['Start'].month <= 6 else '2'}학기"
        print(f"추출된 학기: {semester}")
        return semester

    def _format_response(self, events, query):
        response = ["안녕하세요! 관련 학사일정 안내드립니다.\n"]
        semester = self._extract_semester(events)
        if semester:
            events = events[events['Title'].str.contains(semester, na=False)]
            print(f"학기 필터링 적용: {semester}")

        # 진행 중: 현재 날짜가 Start와 End 사이에 포함
        ongoing = events[(events['Start'] <= self.current_date) & (self.current_date <= events['End'])]
        # 과거: 종료일이 현재 날짜보다 이전
        past = events[events['End'] < self.current_date]
        # 미래: 시작일이 현재 날짜보다 이후
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
            # 진행 중인 일정이 있으면 💡 사용 안 함
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

    def get_answer(self, question):
        results = self._get_relevant_documents(question)
        return self._format_response(results, question) if not results.empty else (
            f"⚠️ '{question}' 관련 일정을 찾지 못했습니다.\n"
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