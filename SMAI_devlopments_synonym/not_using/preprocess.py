# preprocess.py (신규 생성)

import csv
import json
import re
from datetime import datetime


def preprocess_csv(input_path, output_dir):
    chunks = []

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 날짜 형식 변환: 2025-02-05 → 2025년 02월 05일
            start_date = datetime.strptime(row['Start'], "%Y-%m-%d").strftime("%Y년 %m월 %d일")
            end_date = datetime.strptime(row['End'], "%Y-%m-%d").strftime("%Y년 %m월 %d일")

            # 자연어 형식의 청크 생성
            chunk = f"{row['Title']}: {start_date}부터 {end_date}까지"
            chunks.append(chunk)

    # 전처리 결과 저장
    with open(f'{output_dir}/preprocessed_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False)


if __name__ == "__main__":
    preprocess_csv('../hagsailjeong.csv', 'preprocessing')
