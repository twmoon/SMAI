import pandas as pd
from datetime import datetime
import ollama
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from SynonymPattern import SynonymManager


class AcademicCalendarRAG:
    def __init__(self, csv_path='hagsailjeong.csv'):
        # SynonymManager 인스턴스 생성
        self.synonym_mgr = SynonymManager()
        # synonyms.json 파일 생성/로딩 완료 여부를 기다림
        self._wait_for_synonyms()
        self.synonym_map = self.synonym_mgr.get_synonyms()
        self.df = self._load_data(csv_path)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        self._prepare_vectors()

    def _wait_for_synonyms(self):
        retry_count = 0
        # synonyms.json 파일이 생성될 때까지 대기 (최대 60번, 10초 간격 → 최대 10분, 필요에 따라 조정)
        while not self.synonym_mgr.synonym_path.exists():
            if retry_count == 0:
                print("동의어 사전을 생성 중입니다. 최대 2분 소요될 수 있습니다...")
                # 처음 한 번 생성 시도
                self.synonym_mgr._generate_synonyms()
            if retry_count >= 12:  # 12회 × 10초 = 120초 (2분 타임아웃)
                raise FileNotFoundError("동의어 사전 생성 실패")
            if not self.synonym_mgr.synonym_path.exists():
                print("생성 진행 중... (10초 후 재확인)")
                time.sleep(10)
                retry_count += 1
                continue
            break
        print("동의어 사전이 정상적으로 로드되었습니다.\n")

    def _load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])
        current_date = datetime.today().date()
        print("현재 날짜:", current_date)
        df = df[df['Start'].dt.year >= current_date.year]

        # 질의에 활용할 태그 추출: 간단 키워드 기반
        keywords = ['수강신청', '등록', '성적 입력', '성적입력', '계절수업']

        def extract_tags(title):
            tags = []
            for kw in keywords:
                if kw in title:
                    tags.append(kw)
            return ", ".join(tags) if tags else "일반"

        df['tags'] = df['Title'].apply(extract_tags)
        df['document'] = df.apply(lambda row:
                                  f"{row['Title']} ({row['tags']}) 일정은 {row['Start'].strftime('%Y년 %m월 %d일')}부터 {row['End'].strftime('%Y년 %m월 %d일')}까지입니다.",
                                  axis=1)
        return df

    def _prepare_vectors(self):
        docs = self.df['document'].tolist()
        self.document_vectors = self.vectorizer.fit_transform(docs)
        tokenized_documents = [doc.split(" ") for doc in self.df['document']]
        self.bm25 = BM25Okapi(tokenized_documents)

    def _get_relevant_documents(self, query, top_k=3):
        # TF-IDF를 이용한 기본 검색
        query_vector = self.vectorizer.transform([query])
        similarities_all = cosine_similarity(query_vector, self.document_vectors).flatten()
        q_lower = query.lower()
        adjusted_similarities = similarities_all.copy()
        # 예시: '수강신청' 키워드가 있으면 장바구니 문서 페널티, 아니면 보너스를 적용
        for i in range(len(adjusted_similarities)):
            title = self.df.iloc[i]['Title'].lower()
            if "수강신청" in q_lower:
                if "장바구니" in title:
                    adjusted_similarities[i] *= 0.5
                else:
                    adjusted_similarities[i] *= 1.2
            if ("등록" in q_lower or "등록금" in q_lower) and "등록" in title:
                adjusted_similarities[i] *= 1.1
        top_indices = adjusted_similarities.argsort()[-top_k:][::-1]
        relevant_docs = []
        for idx in top_indices:
            if adjusted_similarities[idx] > 0:
                relevant_docs.append(self.df['document'].iloc[idx])
        return "\n".join(relevant_docs) if relevant_docs else "관련 정보를 찾을 수 없습니다."

    def _get_registration_document(self):
        filtered_data = self.df[self.df['Title'].str.contains('등록', case=False, na=False)]
        if not filtered_data.empty:
            filtered_data = filtered_data.sort_values(by='Start')
            return "\n".join(
                filtered_data.apply(
                    lambda
                        row: f"{row['Title']} 일정은 {row['Start'].strftime('%Y년 %m월 %d일')}부터 {row['End'].strftime('%Y년 %m월 %d일')}까지입니다.",
                    axis=1
                ).tolist()
            )
        return "등록 관련 정보를 찾을 수 없습니다."

    def get_answer(self, question):
        # 질문에 '등록' 계열 단어가 포함되면 등록 관련 정보를 우선 사용
        if "등록" in question or "등록금" in question:
            relevant_context = self._get_registration_document()
        else:
            relevant_context = self._get_relevant_documents(question)
        current_date = datetime.today().date()
        prompt = f"""당신은 상명대학교 학생들을 위한 챗봇입니다. 다음 규칙에 따라 답변해주세요.
0. 현재 날짜: {current_date}
1. 학사일정 중 질문과 관련된 정보를 기반으로 답변할 것.
2. 현재 날짜에서 가장 가까운 일정을 출력할 것.
아래는 질문과 관련된 학사일정 정보입니다:
{relevant_context}
질문: {question}
위 정보를 참고하여 답변해주세요. 날짜 정보는 정확하게 포함시키고, 관련 정보를 찾지 못한 경우 유사한 정보를 바탕으로 대답해주세요."""
        try:
            response = ollama.chat(
                model='exaone3.5:latest',
                messages=[{'role': 'user', 'content': prompt}]
            )
            content = response['message']['content']
            cleaned_content = re.sub(r'\*{2,}', '', content).strip()
            return cleaned_content
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"


def main():
    rag_system = AcademicCalendarRAG()
    print("학사일정 RAG Load 완료.")
    print("종료하려면 'quit' 또는 'exit'를 입력.")
    while True:
        question = input("\n질문: ")
        if question.lower() in ['quit', 'exit']:
            break
        start_time = time.time()
        answer = rag_system.get_answer(question)
        print("\n답변:", answer)
        elapsed_time = time.time() - start_time
        print("Time Elapsed: {:.2f} Sec".format(elapsed_time))


if __name__ == "__main__":
    main()
