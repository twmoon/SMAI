import pandas as pd
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


class AcademicCalendarRAG:
    def __init__(self, csv_path='hagsailjeong.csv'):
        self.df = self._load_data(csv_path)
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        self._prepare_vectors()

    def _load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])
        # 각 이벤트를 문서화
        df['document'] = df.apply(lambda row:
                                  f"{row['Title']} 일정은 {row['Start'].strftime('%Y년 %m월 %d일')}부터 "
                                  f"{row['End'].strftime('%Y년 %m월 %d일')}까지입니다.", axis=1)
        return df

    def _prepare_vectors(self):
        # 문서들을 TF-IDF 벡터로 변환
        self.document_vectors = self.vectorizer.fit_transform(self.df['document'])

    def _get_relevant_documents(self, query, top_k=3):
        # 쿼리를 벡터화
        query_vector = self.vectorizer.transform([query])

        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()

        # 상위 k개의 관련 문서 선택
        top_indices = similarities.argsort()[-top_k:][::-1]

        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 유사도가 0보다 큰 문서만 선택
                relevant_docs.append(self.df['document'].iloc[idx])

        return "\n".join(relevant_docs)

    def get_answer(self, question):
        # 오늘 날짜
        today = datetime.today()

        # 관련 문서 검색
        relevant_context = self._get_relevant_documents(question)

        # 프롬프트 구성
        prompt = f"""당신은 상명대학교 학생들을 위한 챗봇입니다. 다음과 같은 규칙을 바탕으로 답변해주세요.
        1. 현재 연도를 기준으로 이전의 내용은 답변에서 제외할 것.
        2. 학사일정 중 질문과 관련된 정보를 기반으로 답변할 것.
        다음은 현재 연도입니다: {today.strftime('%Y')}
        다음은 학교 학사일정 중 질문과 관련된 정보입니다: {relevant_context}
        질문: {question}
        위 정보를 바탕으로 질문에 답변해주세요. 날짜 정보는 정확하게 포함해주시고, 
        찾은 정보가 질문과 관련이 없다면 "관련 정보를 찾을 수 없습니다"라고 답변해주세요."""

        try:
            response = ollama.chat(
                model='exaone3.5:latest',
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"


def main():
    # RAG 시스템 초기화
    rag_system = AcademicCalendarRAG()

    print("학사일정 RAG Load 완료.")
    print("종료하려면 'quit' 또는 'exit'를 입력.")

    while True:
        question = input("\n질문: ")

        if question.lower() in ['quit', 'exit']:
            break

        answer = rag_system.get_answer(question)
        print("\n답변:", answer)


if __name__ == "__main__":
    main()