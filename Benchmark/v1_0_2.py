import pandas as pd
from datetime import datetime
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama  # Added missing import


class AcademicCalendarRAG:
    def __init__(self, csv_path='hagsailjeong.csv'):
        self.df = self._load_data(csv_path)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        self.document_vectors = None
        self._prepare_vectors()

    def _load_data(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
            df['End'] = pd.to_datetime(df['End'], errors='coerce')

            today = datetime.today()
            df = df[df['Start'].dt.year >= today.year].copy()

            df['document'] = df.apply(
                lambda row: (
                    f"{row['Title']} 일정은 "
                    f"{row['Start'].strftime('%Y년 %m월 %d일')}부터 "
                    f"{row['End'].strftime('%Y년 %m월 %d일')}까지입니다."
                ) if pd.notnull(row['Start']) and pd.notnull(row['End']) else "",
                axis=1
            )
            return df.dropna(subset=['document'])
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

    def _prepare_vectors(self):
        self.document_vectors = self.vectorizer.fit_transform(
            self.df['document'].values
        )

    def _get_relevant_documents(self, query, top_k=3):
        today = datetime.today()
        query_vector = self.vectorizer.transform([query])

        # Filter documents before calculation
        valid_mask = (self.df['Start'] >= today).values
        valid_vectors = self.document_vectors[valid_mask]

        if valid_vectors.shape[0] == 0:
            return "관련 정보를 찾을 수 없습니다."

        similarities = cosine_similarity(query_vector, valid_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        relevant_docs = [
            self.df.iloc[valid_mask.nonzero()[0][idx]]['document']
            for idx in top_indices
            if similarities[idx] > 0
        ]

        return "\n".join(relevant_docs) if relevant_docs else "관련 정보를 찾을 수 없습니다."

    def get_answer(self, question):
        try:
            today = datetime.today()
            relevant_context = self._get_relevant_documents(question)

            prompt = f"""당신은 상명대학교 학생들을 위한 챗봇입니다. 다음 규칙에 따라 답변해주세요.
1. 현재 연도 기준 이전의 일정은 제외할 것.
2. 학사일정 중 질문과 관련된 정보를 기반으로 답변할 것.
3. 현재 날짜에서 가장 가까운 일정을 출력할 것.
현재 날짜: {today.strftime('%Y년 %m월 %d일')}
아래는 질문과 관련된 학사일정 정보입니다:
{relevant_context}
질문: {question}
위 정보를 참고하여 답변해주세요. 날짜 정보는 정확하게 포함시켜주시고, 관련 정보를 찾지 못한 경우 유사한 정보를 바탕으로 대답해주세요."""

            response = ollama.chat(
                model='exaone3.5:latest',
                messages=[{'role': 'user', 'content': prompt}]
            )
            content = response['message']['content']
            cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return cleaned_content.strip()

        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"


def main():
    try:
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
            print(f"Time Elapsed: {elapsed_time:.2f} Sec")

    except Exception as e:
        print(f"시스템 초기화 실패: {str(e)}")


if __name__ == "__main__":
    main()
