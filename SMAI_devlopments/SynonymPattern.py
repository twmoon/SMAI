import json
import pandas as pd
from pathlib import Path
from kiwipiepy import Kiwi
import ollama
import time
from tqdm import tqdm
import re
import random


class SynonymManager:
    def __init__(self, csv_path='hagsailjeong.csv', synonym_path='academic_terms.json'):
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

        tqdm.write("▶ 학술 용어 데이터베이스 생성 시작")
        start_time = time.time()

        for term in tqdm(terms, desc="학술 용어 생성", unit="term"):
            term_db[term] = {
                'synonyms': self._get_llm_terms(term, '동의어'),
                'negatives': self._get_llm_terms(term, '배제용어'),
                'category': self._get_term_category(term)  # 범주 추가
            }

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
                model='exaone3.5:32b',
                messages=[{'role': 'user', 'content': prompt_map[term_type]}]
            )
            cleaned_response = re.sub(r'[\d*()\n\r]', '', response['message']['content'])
            raw_terms = [t.strip() for t in cleaned_response.split(',')]
            valid_terms = [
                t for t in raw_terms
                if t and len(t.split()) <= 3 and term not in t and all(word not in t for word in term.split())
                and self._is_academic_term(t)
            ]
            valid_terms = list(dict.fromkeys(valid_terms))  # 중복 제거

            term_category = self._get_term_category(term)
            default_synonyms = self._get_default_synonyms(term_category)
            default_negatives = self._get_default_negatives(term_category)
            num_terms = syno_num if term_type == '동의어' else nega_num

            if len(valid_terms) < num_terms:
                defaults = default_synonyms if term_type == '동의어' else default_negatives
                valid_terms.extend(random.sample(defaults, min(num_terms - len(valid_terms), len(defaults))))
            return valid_terms[:num_terms]
        except Exception as e:
            print(f"\n⚠️ 오류: {term} - {term_type} 생성 중 오류 발생: {str(e)}")
            term_category = self._get_term_category(term)
            if term_type == '동의어':
                return random.sample(self._get_default_synonyms(term_category), syno_num)
            else:
                return random.sample(self._get_default_negatives(term_category), nega_num)

    def _get_term_category(self, term):
        """항목의 범주 식별"""
        if '수강신청' in term or '장바구니' in term or '교차수강신청' in term or '정정 및 취소' in term or '수강포기' in term or '폐강 공고' in term:
            return 'enrollment'
        elif '등록' in term and '성적' not in term:
            return 'registration'
        elif '성적' in term or '이의신청' in term:
            return 'grade'
        elif '입학식' in term or '학위수여식' in term:
            return 'admission_graduation'
        elif '개강' in term or '강의평가' in term or '고사' in term or '자율보강' in term:
            return 'lecture_exam'
        elif '방학' in term or '계절수업' in term:
            return 'vacation_seasonal'
        elif '학적' in term:
            return 'status'
        return 'general'

    def _get_default_synonyms(self, term_category):
        """범주별 기본 동의어 제공"""
        defaults = {
            'enrollment': ["강좌등록", "수업 선택", "장바구니 등록", "학기 과목 신청", "시간표 작성", "수강 예약", "과목 추가"],
            'registration': ["학기 등록", "수강료 납부", "등록 절차", "납부 기간", "학비 납입"],
            'grade': ["성적 확인", "학점 조회", "성적 증명", "평가 결과", "학업 성취"],
            'admission_graduation': ["입학 행사", "신입생 환영", "입학 오리엔테이션", "졸업식", "학위 수여"],
            'lecture_exam': ["학기 시작", "수업 평가", "시험 기간", "강의 계획", "보강 일정"],
            'vacation_seasonal': ["방학 개시", "계절 강좌", "단기 수업", "휴가 기간", "특별 과정"],
            'status': ["학적 변경", "상태 변동", "학사 관리", "등록 상태", "학생 기록"],
            'general': ["학사 일정", "교육 과정", "학기 계획", "수업 일정", "강의 계획"]
        }
        return defaults.get(term_category, defaults['general'])

    def _get_default_negatives(self, term_category):
        """범주별 기본 배제어 제공 (다른 범주에서 혼동될 수 있는 용어 포함)"""
        defaults = {
            'enrollment': ["등록금 납부", "성적 확인", "입학식", "학위수여식", "개강", "방학 시작", "학적 변동"],
            'registration': ["수강신청", "성적 입력", "입학 오리엔테이션", "졸업식", "중간고사", "계절수업", "학적 변경"],
            'grade': ["강좌등록", "학기 등록", "신입생 환영", "학위 수여", "기말강의평가", "하계방학", "학적 관리"],
            'admission_graduation': ["수업 선택", "수강료 납부", "학점 조회", "시험 기간", "동계방학", "학적 변동"],
            'lecture_exam': ["장바구니 등록", "등록 절차", "성적 증명", "입학 행사", "계절 강좌", "상태 변동"],
            'vacation_seasonal': ["학기 과목 신청", "납부 기간", "평가 결과", "졸업식", "수업 평가", "학적 기록"],
            'status': ["시간표 작성", "학비 납입", "학업 성취", "신입생 환영", "보강 일정", "단기 수업"],
            'general': ["이의 신청", "성적 정정", "장학금 신청", "휴학 신청", "졸업 인증", "재입학 신청", "전과 신청"]
        }
        all_negatives = []
        for cat, negs in defaults.items():
            if cat != term_category:
                all_negatives.extend(negs[:3])
        category_negatives = defaults.get(term_category, defaults['general'])
        return list(set(category_negatives + all_negatives))[:15]

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
        return self.term_data.get(term, {'synonyms': [], 'negatives': [], 'category': 'general'})

    def get_synonyms(self):
        """전체 동의어 데이터 반환"""
        return self.term_data


if __name__ == '__main__':
    sm = SynonymManager()
