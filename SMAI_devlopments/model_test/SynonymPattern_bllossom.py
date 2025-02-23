import json
import pandas as pd
from pathlib import Path
from kiwipiepy import Kiwi
import ollama
import time
from tqdm import tqdm
import sys
import re
import random

class SynonymManager:
    def __init__(self, csv_path='hagsailjeong.csv', synonym_path='academic_terms_bllossom_8b.json'):
        self.csv_path = Path(csv_path)
        self.synonym_path = Path(synonym_path)
        self.kiwi = Kiwi()
        self.term_data = self._initialize_data()

    def _initialize_data(self):
        """데이터 초기화: 기존 파일이 있으면 로드, 없으면 생성"""
        if self.synonym_path.exists():
            return self._load_data()
        return self._generate_terms()

    def _generate_terms(self):
        """CSV에서 항목을 읽어 동의어와 배제어 생성"""
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
            sys.stdout.flush()

        self._save_data(term_db)
        elapsed = time.time() - start_time
        print(f"\n✓ 데이터베이스 생성 완료 ({elapsed:.1f}초)")
        return term_db

    def _get_llm_terms(self, term, term_type, syno_num=5, nega_num=15):
        """LLM을 통해 동의어 또는 배제어 생성"""
        prompt_map = {
            '동의어': f"'{term}'의 학사일정 동의어 {syno_num}개를 쉼표로 구분해 나열하세요. 한국어로만, 설명 없이, '{term}'과 중복되지 않는 구체적이고 다양한 학사 용어만 제공하세요. 예: '수강신청'이면 '강좌등록, 수업 선택, 장바구니 등록, 학기 과목 신청, 시간표 작성'",
            '배제용어': f"'{term}' 검색 시 혼동되지만 관련 없는 한국어 학사용어 {nega_num}개를 쉼표로 구분해 나열하세요. 설명 없이, '{term}'과 직접 관련된 용어는 제외하고, 혼동될 수 있는 학사 용어만 제공하세요. 예: '수강신청'이면 '수강 정정, 수강 포기, 성적 정정, 이의 제기, 휴학 신청'"
        }
        try:
            response = ollama.chat(
                model='bllossom:8b',
                messages=[{'role': 'user', 'content': prompt_map[term_type]}]
            )
            # 응답 정제: 숫자, 특수문자, 줄바꿈 제거
            cleaned_response = re.sub(r'[\d*()\n\r]', '', response['message']['content'])
            raw_terms = [t.strip() for t in cleaned_response.split(',')]
            # 유효 용어 필터링: 중복 제거, 학사 용어 확인
            valid_terms = [t for t in raw_terms if t and len(t.split()) <= 3 and term not in t and all(word not in t for word in term.split()) and self._is_academic_term(t)]
            valid_terms = list(dict.fromkeys(valid_terms))  # 중복 제거
            # 항목 유형별 기본값
            term_type_key = self._get_term_type(term)
            default_synonyms = self._get_default_synonyms(term_type_key)
            default_negatives = self._get_default_negatives(term_type_key)
            num_terms = syno_num if term_type == '동의어' else nega_num
            # 부족한 경우 기본값으로 채움
            if len(valid_terms) < num_terms:
                defaults = default_synonyms if term_type == '동의어' else default_negatives
                valid_terms.extend(random.sample(defaults, min(num_terms - len(valid_terms), len(defaults))))
            return valid_terms[:num_terms]
        except Exception as e:
            print(f"\n⚠️ 오류: {term} - {term_type} 생성 중 오류 발생: {str(e)}")
            # 오류 시 기본값 반환
            return random.sample(self._get_default_synonyms(term_type_key), syno_num) if term_type == '동의어' else random.sample(self._get_default_negatives(term_type_key), nega_num)

    def _get_term_type(self, term):
        """항목 유형 식별"""
        if '수강신청' in term:
            return 'enrollment'
        elif '성적' in term:
            return 'grade'
        elif '입학식' in term:
            return 'admission'
        return 'general'

    def _get_default_synonyms(self, term_type_key):
        """유형별 기본 동의어 제공"""
        defaults = {
            'enrollment': ["강좌등록", "수업 선택", "장바구니 등록", "학기 과목 신청", "시간표 작성", "수강 예약", "과목 추가"],
            'grade': ["성적 확인", "학점 조회", "성적 증명", "평가 결과", "학업 성취"],
            'admission': ["입학 행사", "신입생 환영", "입학 오리엔테이션", "새내기 행사", "입학 절차"],
            'general': ["학사 일정", "교육 과정", "학기 계획", "수업 일정", "강의 계획"]
        }
        return defaults.get(term_type_key, defaults['general'])

    def _get_default_negatives(self, term_type_key):
        """유형별 기본 배제어 제공"""
        defaults = {
            'enrollment': ["수강 정정", "수강 포기", "성적 정정", "이의 제기", "휴학 신청", "졸업 인증", "재입학 신청", "전과 신청", "편입 상담", "등록금 납부", "학점 계산", "출석 확인", "기말고사", "성적 조회", "중간고사"],
            'grade': ["수강 등록", "강좌 선택", "시간표 조정", "수업 신청", "장바구니 추가", "학점 인정", "전공 변경", "학사 경고", "기숙사 신청", "교과목 변경", "수업 평가", "출석 확인", "등록금 납부", "휴학 신청", "졸업 인증"],
            'admission': ["수강 신청", "강좌 등록", "성적 확인", "학점 조회", "졸업 인증", "휴학 신청", "재입학 신청", "전과 신청", "편입 상담", "등록금 납부", "학점 계산", "출석 확인", "기말고사", "성적 조회", "중간고사"],
            'general': ["이의 신청", "성적 정정", "장학금 신청", "휴학 신청", "졸업 인증", "재입학 신청", "전과 신청", "편입 상담", "등록금 납부", "학점 계산", "출석 확인", "기말고사", "성적 조회", "중간고사", "수업 평가"]
        }
        return defaults.get(term_type_key, defaults['general'])

    def _is_academic_term(self, term):
        """학사 용어 여부 확인"""
        tokens = self.kiwi.tokenize(term)
        non_academic = {"입니다", "가 있습니다", "반박", "항변", "소청", "논평", "견해"}
        return not any(token.form in non_academic for token in tokens)

    def _save_data(self, data):
        """데이터 저장"""
        with open(self.synonym_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_data(self):
        """데이터 로드"""
        with open(self.synonym_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_term_info(self, term):
        """특정 항목 정보 반환"""
        return self.term_data.get(term, {'synonyms': [], 'negatives': []})

    def get_synonyms(self):
        """전체 동의어 데이터 반환"""
        return self.term_data

if __name__ == '__main__':
    sm = SynonymManager()