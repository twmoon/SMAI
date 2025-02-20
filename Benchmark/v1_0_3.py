import pandas as pd
from datetime import datetime
import ollama
import numpy as np
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process  # 퍼지 매칭을 위한 라이브러리
import unicodedata

class AcademicCalendarRAG:
    def __init__(self, csv_path='hagsailjeong.csv'):
        self.df = self._load_data(csv_path)
        # 제목 정규화를 적용하여 새로운 열 생성 (소문자 변환 + 공백 제거)
        self.df['normalized_title'] = self.df['Title'].apply(self._normalize_text)
        # 문자 기반 n-gram 분석 적용: 문자 단위 분석, ngram 범위 2~4
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        self.document_vectors = None
        self._prepare_vectors()

    def _load_data(self, csv_path):
        # CSV 파일 로드 및 날짜 데이터형 변환
        df = pd.read_csv(csv_path)
        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])
        today = datetime.today()
        # 현재 연도 기준 이전의 일정은 제외
        df = df[df['Start'].dt.year >= today.year]
        # 각 이벤트를 하나의 문서 문자열로 구성
        df['document'] = df.apply(lambda row:
                                  f"{row['Title']} 일정은 {row['Start'].strftime('%Y년 %m월 %d일')}부터 "
                                  f"{row['End'].strftime('%Y년 %m월 %d일')}까지입니다.",
                                  axis=1)
        return df

    def _normalize_text(self, text):
        # 유니코드 정규화 수행한 후 소문자로 변환하고 공백 제거
        norm_text = unicodedata.normalize('NFC', text)
        return norm_text.lower().replace(" ", "")

    def _prepare_vectors(self):
        # 모든 문서를 TF-IDF 벡터로 변환
        self.document_vectors = self.vectorizer.fit_transform(self.df['document'])

    def _get_relevant_documents(self, query, top_k=3):
        today = datetime.today()
        # 질문을 벡터화
        query_vector = self.vectorizer.transform([query])
        similarities_all = cosine_similarity(query_vector, self.document_vectors).flatten()
        # 오늘 이후의 이벤트만 고려하도록 필터링
        mask = self.df['Start'] >= today
        similarities = np.where(mask, similarities_all, 0)
        top_indices = similarities.argsort()[-top_k:][::-1]
        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] > 0:
                relevant_docs.append(self.df['document'].iloc[idx])
        return "\n".join(relevant_docs) if relevant_docs else "관련 정보를 찾을 수 없습니다."

    def _find_best_fuzzy_match(self, query, threshold=80):
        # 질문도 동일하게 정규화한 후 퍼지 매칭 수행
        normalized_query = self._normalize_text(query)
        choices = self.df['normalized_title'].tolist()
        best_match, score = process.extractOne(normalized_query, choices)
        if score >= threshold:
            # 유사도 기준을 만족하는 경우 해당 행 반환
            row = self.df[self.df['normalized_title'] == best_match]
            if not row.empty:
                return row.iloc[0]
        return None

    def get_answer(self, question):
        today = datetime.today()
        # 먼저 퍼지 매칭을 시도하여 질문과 가장 유사한 제목을 찾음
        match = self._find_best_fuzzy_match(question)
        if match is not None:
            title = match['Title']
            start_date = match['Start'].strftime('%Y년 %m월 %d일')
            end_date = match['End'].strftime('%Y년 %m월 %d일')
            answer_text = f"{title} 일정은 {start_date}부터 {end_date}까지입니다."
        else:
            # 퍼지 매칭으로 찾지 못한 경우, TF-IDF 기반 결과 사용
            answer_text = self._get_relevant_documents(question)

        prompt = f"""당신은 상명대학교 학생들을 위한 챗봇입니다. 다음 규칙에 따라 답변해주세요.
1. 현재 연도 기준 이전의 일정은 제외할 것.
2. 학사일정 중 질문과 관련된 정보를 기반으로 답변할 것.
3. 현재 날짜에서 가장 가까운 일정을 출력할 것.
현재 날짜: {today.strftime('%Y년 %m월 %d일')}
아래는 질문과 관련된 학사일정 정보입니다:
{answer_text}
질문: {question}
위 정보를 참고하여 답변해주세요. 날짜 정보는 정확하게 포함시키고, 관련 정보를 찾지 못한 경우 유사한 정보를 바탕으로 대답해주세요."""
        try:
            response = ollama.chat(
                model='exaone3.5:latest',
                messages=[{'role': 'user', 'content': prompt}]
            )
            content = response['message']['content']
            # "<think> ... </think>" 구문 제거
            cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return cleaned_content.strip()
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
