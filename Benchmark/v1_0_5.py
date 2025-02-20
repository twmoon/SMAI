import pandas as pd
from datetime import datetime
import ollama
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AcademicCalendarRAG:
    def __init__(self, csv_path='hagsailjeong.csv'):
        self.df = self._load_data(csv_path)
        # 문자 기반 n-gram 분석 적용: 문자 단위 분석, ngram 범위 2~4
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        self.document_vectors = None
        self._prepare_vectors()

    def _load_data(self, csv_path):
        # CSV 파일 로드 및 날짜 데이터형 변환
        df = pd.read_csv(csv_path)
        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])

        # 현재 연도 기준 이전의 일정은 제외
        current_date = datetime.today().date()
        # print("현재 날짜:", current_date)
        df = df[df['Start'].dt.year >= current_date.year]

        # 데이터 전처리: Title 열에서 주요 키워드를 기반으로 태그 추가
        keywords = ['수강신청', '등록', '성적 입력', '성적입력', '계절수업']

        def extract_tags(title):
            tags = []
            for kw in keywords:
                if kw in title:
                    tags.append(kw)
            return ", ".join(tags) if tags else "일반"

        df['tags'] = df['Title'].apply(extract_tags)

        # 각 이벤트를 하나의 문서 문자열로 구성 (태그 정보를 포함)
        df['document'] = df.apply(lambda row:
                                  f"{row['Title']} ({row['tags']}) 일정은 {row['Start'].strftime('%Y년 %m월 %d일')}부터 {row['End'].strftime('%Y년 %m월 %d일')}까지입니다.",
                                  axis=1)
        return df

    def _prepare_vectors(self):
        # 모든 문서를 TF-IDF 벡터로 변환
        self.document_vectors = self.vectorizer.fit_transform(self.df['document'])

    def _get_relevant_documents(self, query, top_k=3):
        # 기본적으로 TF-IDF를 이용해 질문과 유사한 문서를 검색
        query_vector = self.vectorizer.transform([query])
        similarities_all = cosine_similarity(query_vector, self.document_vectors).flatten()

        # 가중치 조정을 위한 추가 단계
        q_lower = query.lower()
        adjusted_similarities = similarities_all.copy()
        # 가중치 조정 예시: "수강신청"이 질문에 있으면, "장바구니"가 포함된 일정은 페널티, 포함되지 않은 경우 보너스를 적용
        for i in range(len(adjusted_similarities)):
            title = self.df.iloc[i]['Title'].lower()
            # 수강신청 관련 가중치 조정
            if "수강신청" in q_lower:
                if "장바구니" in title:
                    adjusted_similarities[i] *= 0.5  # 장바구니 문서는 점수를 절반으로 낮춤
                else:
                    adjusted_similarities[i] *= 1.2  # 보너스 점수 부여
            # 등록 관련 질문이 있을 경우 (예: "등록", "등록금")
            if ("등록" in q_lower or "등록금" in q_lower) and "등록" in title:
                adjusted_similarities[i] *= 1.1

        top_indices = adjusted_similarities.argsort()[-top_k:][::-1]
        relevant_docs = []
        for idx in top_indices:
            if adjusted_similarities[idx] > 0:
                relevant_docs.append(self.df['document'].iloc[idx])
        return "\n".join(relevant_docs) if relevant_docs else "관련 정보를 찾을 수 없습니다."

    def _get_registration_document(self):
        # '등록' 관련 데이터를 필터링 (대소문자 구분 없이 검색)
        filtered_data = self.df[self.df['Title'].str.contains('등록', case=False, na=False)]
        if not filtered_data.empty:
            filtered_data = filtered_data.sort_values(by='Start')
            return "\n".join(
                filtered_data.apply(
                    lambda row: f"{row['Title']} 일정은 {row['Start'].strftime('%Y년 %m월 %d일')}부터 {row['End'].strftime('%Y년 %m월 %d일')}까지입니다.",
                    axis=1)
            )
        return "등록 관련 정보를 찾을 수 없습니다."

    def get_answer(self, question):
        # 질문에 "등록" 또는 "등록금" 키워드가 있을 경우 등록 관련 정보를 우선 검색
        if "등록" in question or "등록금" in question:
            relevant_context = self._get_registration_document()
        else:
            # 그 외의 경우 가중치가 조정된 TF-IDF 기반의 관련 문서를 검색
            relevant_context = self._get_relevant_documents(question)

        current_date = datetime.today().date()
        prompt = f"""당신은 상명대학교 학생들을 위한 챗봇입니다. 다음 규칙에 따라 답변해주세요.
0. 현재 날짜는 {current_date}야.
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
