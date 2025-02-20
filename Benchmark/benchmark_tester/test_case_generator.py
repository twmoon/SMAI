# test_case_generator.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from faker import Faker
import random
import re

class EnhancedTestCaseGenerator:
    def __init__(self, seed=42):
        self.faker = Faker('ko_KR')
        random.seed(seed)
        np.random.seed(seed)
        self.keyword_map = {
            '수강신청': ['수강', '강의등록', '과목선택', '강좌추가', '장바구니'],
            '등록금': ['등록', '학비', '납부', '결제', '장학금'],
            '성적입력': ['성적', '점수', '평가', '학점', 'GPA'],
            '계절학기': ['계절', '여름학기', '겨울학기', '방학강의', '단기과정']
        }

    def _generate_high_ambiguity_questions(self):
        return [
            ("기간 알려줘", ['수강신청', '등록금', '계절학기']),
            ("언제 시작해?", ['수강신청', '등록금', '성적입력']),
            ("마감일이 언제야?", ['수강신청', '등록금', '계절학기']),
            ("신청 관련 정보", ['수강신청', '등록금', '성적입력']),
            ("처리해야 하는 일", ['수강신청', '등록금', '계절학기', '성적입력'])
        ]

    def _generate_tricky_invalid_cases(self):
        return [
            ("수강신청 기간에 학식 메뉴 알려줘", None),
            ("등록금 납부하고 도서관 좌석 현황", None),
            ("성적 입력할 때 총장실 전화번호", None),
            ("2024년 계절학기 일정과 교수진 정보", None),
            ("장바구니에 넣은 과목의 강의실 위치", None)
        ]

    def _generate_variations(self, base_question, num_variants=5):
        modifiers = [
            lambda s: s + '?'*random.randint(1,3),
            lambda s: s.replace(' ', '  '),
            lambda s: re.sub(r'[가-힣]', lambda x: x.group()*2, s),
            lambda s: s.upper(),
            lambda s: s.lower(),
            lambda s: s[:-1] if s.endswith('?') else s
        ]
        return [self._apply_modifiers(base_question, modifiers, random.randint(2,4))
                for _ in range(num_variants)]

    def _apply_modifiers(self, text, modifiers, times):
        for _ in range(times):
            text = random.choice(modifiers)(text)
        return text

    def generate_test_suite(self):
        test_suite = {'exact_match': [], 'synonym': [], 'unstructured': [], 'ambiguous': [], 'invalid': []}

        # 모호성 케이스
        for q, kws in self._generate_high_ambiguity_questions():
            test_suite['ambiguous'].extend([(self._add_noise(q), kws) for _ in range(20)])

        # 무효 케이스
        invalid_samples = self._generate_tricky_invalid_cases()
        test_suite['invalid'] = [(self._add_noise(q), None) for q, _ in invalid_samples for _ in range(10)]

        # 정확 매칭 케이스
        for kw in self.keyword_map:
            base = f"{self.faker.year()}년 {kw} 일정을 알려주세요"
            test_suite['exact_match'].extend([(self._add_noise(v), kw)
                for v in self._generate_variations(base)])

        # 동의어 케이스
        for _ in range(300):
            target_kw = random.choice(list(self.keyword_map.keys()))
            synonym = random.choice(self.keyword_map[target_kw])
            base = f"{synonym} 관련 일정이 어떻게 되나요?"
            test_suite['synonym'].append((self._add_noise(base), target_kw))

        # 비정형 케이스
        unstructured_samples = [
            ("수강신청언제임?ㅠㅠ 급해요!!!", "수강신청"),
            ("등록금내야하는데기간모르겠어", "등록금"),
            ("성적입력하다가오류났어요도와줘", "성적입력")
        ]
        for q, kw in unstructured_samples:
            test_suite['unstructured'].extend([(self._add_noise(q), kw) for _ in range(7)])

        return test_suite

    def _add_noise(self, text):
        noise_types = [
            lambda t: t + '!' * random.randint(1, 3),
            lambda t: t.replace('?', '?!'*random.randint(1,2)),
            lambda t: ''.join([c.upper() if i%2==0 else c.lower() for i,c in enumerate(t)]),
            lambda t: re.sub(r'[가-힣]', lambda x: x.group()*random.randint(1,3), t),
            lambda t: t.replace(' ', ''),
            lambda t: t + ' ' + random.choice(['알려주세요', '빨리요', '제발', 'ㅠㅠ'])
        ]
        for _ in range(random.randint(1,3)):
            text = random.choice(noise_types)(text)
        return text

if __name__ == "__main__":
    generator = EnhancedTestCaseGenerator()
    test_suite = generator.generate_test_suite()
    total_cases = sum(len(v) for v in test_suite.values())
    print(f"생성된 테스트 케이스: {total_cases}개")
