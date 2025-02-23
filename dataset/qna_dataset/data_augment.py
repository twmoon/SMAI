import pandas as pd
import random
import re
import os
import time
from tqdm import tqdm  # 프로그레스 바를 위한 라이브러리

# 1. 노이즈 문장 정의
NOISE_PHRASES = [
    "알려주세요.", "도와주세요.", "확인 부탁드립니다.", "답변 기다리겠습니다.",
    "설명해주실 수 있나요?", "자세히 알려주세요.", "궁금합니다.", "방법을 모르겠어요.",
    "가능한가요?", "처리 방법을 알려주세요.", "안내 부탁드립니다.", "해결책이 있을까요?",
    "정보가 필요합니다.", "절차를 알려주세요.", "어떻게 해야 하나요?", "예외 사항은 없나요?",
    "추가 설명 부탁드립니다.", "관련 규정을 알려주세요.", "변경 가능한가요?", "확인이 필요합니다.",
    "지침을 알려주세요.", "신청 방법을 모르겠어요.", "기간을 확인하고 싶어요.", "차이가 있나요?",
    "주의사항이 있나요?", "예시를 들어주세요.", "재확인 부탁드립니다.", "공식 문서를 참고해야 하나요?",
    "특별한 절차가 있나요?", "제한 조건이 있나요?", "시스템 오류인가요?", "연장 가능한가요?",
    "사전 준비가 필요할까요?", "기본 규칙을 알려주세요.", "공지된 내용이 있나요?", "학칙에 따라 다른가요?",
    "담당 부서는 어디인가요?", "즉시 처리 가능한가요?", "포털에서 확인할 수 있나요?", "오류가 발생했어요.",
    "최신 정보인가요?", "혼동스러워요.", "예약이 필요한가요?", "서류 제출이 필요한가요?",
    "유의할 점이 있나요?", "승인 절차가 있나요?", "기본 가이드라인을 알려주세요.", "공식 일정과 다른가요?",
    "필수 항목인가요?", "추가 문의처가 있나요?", "자동 처리되나요?", "수정이 불가능한가요?",
    "이전 사례가 있나요?", "기준을 알려주세요.", "우회 방법이 있나요?", "확인 후 답변 주실 수 있나요?",
    "시스템 상의 문제인가요?", "즉시 반영되나요?", "차후 변경 가능한가요?", "기본 설정은 어떻게 되나요?",
    "공식 답변을 받고 싶어요.", "예외 처리는 어떻게 하나요?", "표준 절차를 알려주세요.", "일괄 처리 가능한가요?",
    "추가 비용이 발생하나요?", "지연 사유가 있나요?", "재검토 가능한가요?", "공식적인 근거가 있나요?",
    "양식이 있나요?", "참고 자료를 알려주세요.", "담당자 연락처를 알려주세요.", "즉시 조치 가능한가요?",
    "시스템 이용 시간이 정해져 있나요?", "부분 수정이 가능한가요?", "이전 안내와 다른 점이 있나요?",
    "공식 채널을 통해 확인해야 하나요?", "자세한 매뉴얼이 있나요?", "예상 소요 시간을 알려주세요.",
    "별도 신청이 필요한가요?", "기본 권한이 필요한가요?", "접수 확인은 어떻게 하나요?", "공지 예정인가요?",
    "일시적인 오류인가요?", "재확인 방법을 알려주세요.", "추가 안내가 필요해요.", "공식 기준을 확인할 수 있을까요?",
    "시스템 이용 제한이 있나요?", "부분 취소 가능한가요?", "담당자 확인이 필요한가요?", "기본값은 어떻게 설정되나요?",
    "즉시 반영되지 않나요?", "사유를 명시해야 하나요?", "공식 절차에 따라야 하나요?", "별도 문의가 필요한가요?",
    "재신청 기간이 있나요?", "기본 제공되는 서비스인가요?", "자동화된 시스템인가요?", "일괄 문의 가능한가요?",
    "공식 문서 링크를 알려주세요.", "추가 지원이 필요하면 어디로 연락하나요?"
]

CSV_PATH = "qna_data.csv"  # 입력 파일
OUTPUT_PATH = "augmented_qna_data.csv"  # 출력 파일

# 2. CSV 파일 읽기 (첫 번째 줄 무시)
try:
    df = pd.read_csv(
        CSV_PATH,
        sep=",",
        quotechar='"',
        header=0,  # 첫 번째 줄을 헤더로 간주
        names=["질문", "답변"],  # 헤더 이름 강제 지정
        engine="python",
        skiprows=1  # 첫 번째 줄(헤더)을 건너뛰고 데이터부터 읽기
    )
    print(f"✅ CSV 로드 성공! 총 {len(df)}개의 데이터 (첫 번째 줄 제외)")
except Exception as e:
    print(f"❌ CSV 읽기 실패: {str(e)}")
    exit()

# 3. 텍스트 변형 함수 정의
def add_typo(text):
    typo_map = {'요': '용', '부탁': '부탁드려용', '확인': '확인해주세용'}
    for k, v in typo_map.items():
        if random.random() < 0.3:
            text = text.replace(k, v)
    return text

def modify_punctuation(text):
    punct = ['???', '!', '..', '#', '^^', '~~']  # 제공된 샘플과 일치
    text = re.sub(r'[?.!]$', '', text)  # 기존 구두점 제거
    return text + random.choice(punct)

def add_special_char(text):
    chars = ['#학사문의', '@급해', '!!', '^^', '~', '...']  # 샘플에 나온 특수문자 포함
    if random.random() < 0.4:
        return text + random.choice(chars)
    return text

def modify_spacing(text):
    words = text.split()
    if len(words) < 2: return text
    if random.random() < 0.5:
        idx = random.randint(0, len(words)-2)
        words[idx] += words[idx+1][0]
        words[idx+1] = words[idx+1][1:]
    return ' '.join(words)

def apply_noise(question):
    transforms = [add_typo, modify_punctuation, add_special_char, modify_spacing]
    for func in random.sample(transforms, k=random.randint(1, 3)):
        question = func(question)
    return question

# 4. 데이터 증강 (프로그레스 바 추가)
augmented_data = []
TEXT_TRANSFORM_COUNT = 15  # 샘플 데이터를 기준으로 적당히 조정 (질문당 50번 변형)

print("🔄 데이터 증강 시작...")
start_time = time.time()

total_tasks = len(df) * (1 + 2 * len(NOISE_PHRASES) + TEXT_TRANSFORM_COUNT)
with tqdm(total=total_tasks, desc="진행 중", unit="task") as pbar:
    for _, row in df.iterrows():
        original_q = row["질문"]
        answer = row["답변"]

        # 원본 저장
        augmented_data.append({"질문": original_q, "답변": answer})
        pbar.update(1)

        # 노이즈 문장 추가
        for phrase in NOISE_PHRASES:
            augmented_data.append({"질문": f"{phrase} {original_q}", "답변": answer})
            pbar.update(1)
            augmented_data.append({"질문": f"{original_q} {phrase}", "답변": answer})
            pbar.update(1)

        # 텍스트 변형 추가
        for _ in range(TEXT_TRANSFORM_COUNT):
            noisy_q = apply_noise(original_q)
            augmented_data.append({"질문": noisy_q, "답변": answer})
            pbar.update(1)

end_time = time.time()
elapsed_time = end_time - start_time

# 5. 결과 저장
pd.DataFrame(augmented_data).to_csv(OUTPUT_PATH, index=False)

print(f"""
✅ 증강 완료!
   - 원본 데이터: {len(df):,}개
   - 노이즈 문장 증강: {(len(NOISE_PHRASES) * 2 * len(df)):,}개
   - 텍스트 변형 증강: {(TEXT_TRANSFORM_COUNT * len(df)):,}개
   - 총 데이터셋: {len(augmented_data):,}개
   - 저장 위치: {os.path.abspath(OUTPUT_PATH)}
   - 소요 시간: {elapsed_time:.2f}초
""")