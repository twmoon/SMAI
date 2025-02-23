import json
import pandas as pd
from pathlib import Path
from kiwipiepy import Kiwi
import ollama
import time


class SynonymManager:
    def __init__(self, csv_path='hagsailjeong.csv', synonym_path='synonyms.json'):
        self.csv_path = Path(csv_path)
        self.synonym_path = Path(synonym_path)
        self.kiwi = Kiwi()
        self.synonyms = self._initialize_synonyms()

    def _initialize_synonyms(self):
        if self.synonym_path.exists():
            return self._load_synonyms()
        return self._generate_synonyms()

    def _generate_synonyms(self):
        start_time = time.time()
        df = pd.read_csv(self.csv_path)
        terms = df['Title'].unique().tolist()
        total_terms = len(terms)
        terms_processed = 0
        dot_count = 0

        print("\n동의어 사전 생성이 시작되었습니다")

        # 진행 상태 초기화
        last_print_time = start_time
        progress_template = "\r진행률: [%s%s] %d초 남음"
        max_dots = 10

        synonym_map = {}
        for term in terms:
            # 동의어 생성
            synonyms = self._get_llm_synonyms(term)
            synonym_map[term] = synonyms

            # 진행률 계산
            terms_processed += 1
            current_time = time.time()
            elapsed = current_time - start_time
            avg_time = elapsed / terms_processed
            remaining_time = avg_time * (total_terms - terms_processed)

            # 2초마다 진행 상태 업데이트
            if current_time - last_print_time >= 2:
                filled_dots = min(dot_count + 1, max_dots)
                empty_dots = max_dots - filled_dots
                progress_bar = "•" * filled_dots + " " * empty_dots
                print(progress_template % (progress_bar, "", int(remaining_time)), end="", flush=True)
                dot_count += 1
                last_print_time = current_time

        # 생성 완료 메시지
        print(f"\r동의어 사전 생성 완료 (총 {time.time() - start_time:.1f}초 소요)           ")
        self._save_synonyms(synonym_map)
        return synonym_map

    def _get_llm_synonyms(self, term, num_synonyms=5):
        try:
            response = ollama.chat(
                model='exaone3.5:latest',
                messages=[{
                    'role': 'user',
                    'content': f"'{term}'의 학사일정 동의어 {num_synonyms}개를 쉼표로 구분해주세요. 예시: 수강신청,강좌등록"
                }]
            )
            return [s.strip() for s in response['message']['content'].split(',')][:num_synonyms]
        except Exception as e:
            print(f"\n오류 발생: {str(e)}")
            return []

    def _save_synonyms(self, data):
        with open(self.synonym_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_synonyms(self):
        with open(self.synonym_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_synonyms(self):
        return self.synonyms
