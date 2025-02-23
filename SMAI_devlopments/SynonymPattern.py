import json
import pandas as pd
from pathlib import Path
from kiwipiepy import Kiwi
import ollama
import time
from tqdm import tqdm
import sys  # Import sys for flushing


class SynonymManager:
    def __init__(self, csv_path='hagsailjeong.csv', synonym_path='academic_terms_bllossom.json'):
        self.csv_path = Path(csv_path)
        self.synonym_path = Path(synonym_path)
        self.kiwi = Kiwi()
        self.term_data = self._initialize_data()

    def _initialize_data(self):
        if self.synonym_path.exists():
            return self._load_data()
        return self._generate_terms()

    def _generate_terms(self):
        df = pd.read_csv(self.csv_path)
        terms = df['Title'].unique().tolist()
        term_db = {}

        print("\n▶ 학술 용어 데이터베이스 생성 시작", flush=True)
        start_time = time.time()

        for term in tqdm(terms, desc="학술 용어 생성", unit="term"):
            term_db[term] = {
                'synonyms': self._get_llm_terms(term, '동의어'),
                'negatives': self._get_llm_terms(term, '배제용어')
            }
            sys.stdout.flush()  # Flush output to force display

        self._save_data(term_db)
        elapsed = time.time() - start_time
        print(f"\n✓ 데이터베이스 생성 완료 ({elapsed:.1f}초)")
        return term_db

    def _get_llm_terms(self, term, term_type, syno_num=5, nega_num=30):
        prompt_map = {
            '동의어': f"'{term}'의 학사일정 동의어 {syno_num}개를 생성. 예시: 수강신청, 강좌등록. Let's think step by step.",
            '배제용어': f"'{term}' 검색 시 혼동되지만 관련 없는 학사용어 {nega_num}개를 생성. 예시: 이의신청, 성적정정. Let's think step by step."
        }
        try:
            response = ollama.chat(
                model='bllossom:8b',
                messages=[{'role': 'user', 'content': prompt_map[term_type]}]
            )
            # 용어 타입에 따라 사용할 개수 설정
            num_terms = syno_num if term_type == '동의어' else nega_num
            return [t.strip() for t in response['message']['content'].split(',')[:num_terms]]
        except Exception as e:
            print(f"\n⚠️ 오류: {str(e)}")
            return []

    def _save_data(self, data):
        with open(self.synonym_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_data(self):
        with open(self.synonym_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_term_info(self, term):
        return self.term_data.get(term, {'synonyms': [], 'negatives': []})

    def get_synonyms(self):
        return self.term_data


if __name__ == '__main__':
    sm = SynonymManager()
