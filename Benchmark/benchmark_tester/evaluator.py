#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import time
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import os
import sys

try:
    import matplotlib.font_manager as fm

    font_path = 'C:/Windows/Fonts/malgun.ttf'
    font_prop = fm.FontProperties(fname=font_path, size=12)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    logging.warning(f"한글 폰트 설정 실패: {str(e)}")
    font_prop = None


class AdvancedBenchmark:
    def __init__(self, versions):
        self.logger = logging.getLogger("Benchmark")
        self.versions = versions
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._initialize()

    def _initialize(self):
        self.logger.info("성능 평가 시스템 초기화 시작")
        self._load_test_cases()
        self.logger.info("테스트 케이스 생성 완료")

    def _load_test_cases(self):
        from test_case_generator import EnhancedTestCaseGenerator
        self.generator = EnhancedTestCaseGenerator()
        self.test_suite = self.generator.generate_test_suite()

    def _load_rag(self, version):
        try:
            file_path = os.path.join(self.base_path, f"{version}.py")
            self.logger.debug(f"로드 시도 파일: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} not found")

            module_name = f"rag_{version}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module.AcademicCalendarRAG()
        except Exception as e:
            self.logger.error(f"모듈 로드 실패: {version}", exc_info=True)
            raise

    def _evaluate_single(self, args):
        version, category, query, expected = args
        try:
            rag = self._load_rag(version)
            start = time.time()
            answer = rag.get_answer(query)
            elapsed = time.time() - start

            if expected:
                if isinstance(expected, list):
                    correct = any(kw in answer for kw in expected)
                else:
                    correct = expected in answer
            else:
                correct = "정보" not in answer

            return {
                'version': version,
                'category': category,
                'query': query,
                'answer': answer,
                'time': elapsed,
                'correct': correct,
                'error': None
            }
        except Exception as e:
            self.logger.error(f"평가 실패: {query}", exc_info=True)
            return {
                'version': version,
                'category': category,
                'query': query,
                'answer': None,
                'time': None,
                'correct': False,
                'error': str(e)
            }

    def run_benchmark(self):
        self.logger.info("벤치마크 실행 시작")
        tasks = [(version, category, query, expected)
                 for version in self.versions
                 for category, cases in self.test_suite.items()
                 for (query, expected) in cases]

        with Pool(cpu_count()) as pool:
            results = []
            with tqdm(total=len(tasks), desc="전체 진행률", ncols=80,
                      bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for result in pool.imap_unordered(self._evaluate_single, tasks):
                    results.append(result)
                    pbar.update()

        results_df = pd.DataFrame(results)
        self._save_test_logs(results_df)
        return results_df

    def _save_test_logs(self, df, filename="test_logs"):
        save_dir = os.path.join(self.base_path, "benchmark_results")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        try:
            csv_path = os.path.join(save_dir, f"{filename}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"테스트 로그 CSV 저장 완료: {csv_path}")

            excel_path = os.path.join(save_dir, f"{filename}.xlsx")
            df.to_excel(excel_path, index=False)
            self.logger.info(f"테스트 로그 Excel 저장 완료: {excel_path}")
        except Exception as e:
            self.logger.error(f"로그 저장 실패: {str(e)}")

    def analyze_results(self, df):
        report = df.groupby('version').agg(
            Accuracy=('correct', 'mean'),
            AvgTime=('time', 'mean'),
            ErrorRate=('error', lambda x: x.notna().mean())
        ).rename(columns={
            'correct': '정확도',
            'time': '평균응답시간',
            'error': '에러율'
        }).fillna(0)

        category_report = df.groupby(['version', 'category']).agg(
            SuccessRate=('correct', 'mean'),
            AvgTime=('time', 'mean'),
            Count=('correct', 'count')
        ).reset_index()

        # 개선된 점수 계산 로직 (랭크 기반)
        success_rate = df.groupby('version')['correct'].mean()

        # 정확도 랭크 계산
        accuracy_rank = report['Accuracy'].rank(ascending=False, method='min')
        accuracy_scores = (60 - (accuracy_rank - 1) * 5).astype(int)

        # 성공률 랭크 계산 (3차 함수 적용 후 랭킹)
        success_rate_values = (success_rate ** 3) * 30
        success_rank = success_rate_values.rank(ascending=False, method='min')
        success_scores = (30 - (success_rank - 1) * 5).astype(int)

        # 응답시간 랭크 계산
        time_rank = report['AvgTime'].rank(ascending=True, method='min')
        time_scores = (10 - (time_rank - 1) * 2).astype(int)

        # 최종 점수
        final_scores = pd.DataFrame({
            '정확도 점수 (60%)': accuracy_scores,
            '성공률 점수 (30%)': success_scores,
            '실행 시간 점수 (10%)': time_scores,
            '최종 점수': (accuracy_scores + success_scores + time_scores)
        })

        plt.figure(figsize=(20, 15))
        if font_prop:
            plt.suptitle("성능 벤치마크 결과", fontsize=14, fontproperties=font_prop)
        else:
            plt.suptitle("성능 벤치마크 결과", fontsize=14)

        # 시각화 1: 정확도 비교
        plt.subplot(3, 2, 1)
        sns.barplot(x=report.index, y='Accuracy', data=report, palette="Blues_d")
        plt.ylim(0, 1)
        if font_prop:
            plt.title("전체 정확도 비교", fontproperties=font_prop)
            plt.ylabel("정확도", fontproperties=font_prop)
        else:
            plt.title("전체 정확도 비교")
            plt.ylabel("정확도")

        # 시각화 2: 평균 응답시간
        plt.subplot(3, 2, 2)
        sns.lineplot(x=report.index, y='AvgTime', data=report, marker="o", markersize=8, linewidth=2.5)
        if font_prop:
            plt.title("전체 평균 응답시간 (초)", fontproperties=font_prop)
        else:
            plt.title("전체 평균 응답시간 (초)")

        # 시각화 3: 정답 분포
        plt.subplot(3, 2, 3)
        crosstab = pd.crosstab(df['version'], df['correct'])
        sns.heatmap(crosstab, annot=True, fmt='d', cmap="YlGnBu")
        if font_prop:
            plt.title("정답 분포 (True/False)", fontproperties=font_prop)
        else:
            plt.title("정답 분포 (True/False)")

        # 시각화 4: 카테고리별 성공률
        plt.subplot(3, 2, 4)
        sns.barplot(x='version', y='SuccessRate', hue='category', data=category_report, palette="viridis")
        plt.ylim(0, 1)
        if font_prop:
            plt.title("버전별 케이스 유형 성공률", fontproperties=font_prop)
        else:
            plt.title("버전별 케이스 유형 성공률")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 시각화 5: 성공률 히트맵
        plt.subplot(3, 2, 5)
        pivot_table = category_report.pivot(index="version", columns="category", values="SuccessRate")
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
        if font_prop:
            plt.title("케이스 유형별 성공률 히트맵", fontproperties=font_prop)
        else:
            plt.title("케이스 유형별 성공률 히트맵")

        # 시각화 6: 최종 점수 표
        plt.subplot(3, 2, 6)
        cell_text = final_scores.values.tolist()
        columns = final_scores.columns.tolist()
        rows = final_scores.index.tolist()
        table = plt.table(cellText=cell_text, colLabels=columns, rowLabels=rows,
                          loc='center', cellLoc='center',
                          colColours=['#f3f3f3'] * 4, rowColours=['#f3f3f3'] * len(rows))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.axis('off')
        if font_prop:
            plt.title("최종 점수 비교", fontproperties=font_prop)
        else:
            plt.title("최종 점수 비교")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_dir = os.path.join(self.base_path, "benchmark_results")
        plt.savefig(os.path.join(save_dir, 'benchmark_results.png'), dpi=300, bbox_inches='tight')

        return report, category_report, final_scores


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.FileHandler('benchmark.log'), logging.StreamHandler()]
    )

    versions = ['v1_0_0', 'v1_0_1', 'v1_0_2', 'v1_0_3', 'v1_0_4', 'v1_0_5']
    runner = AdvancedBenchmark(versions)

    print("벤치마크 실행 중...")
    results_df = runner.run_benchmark()

    print("\n결과 분석 중...")
    final_report, category_report, final_scores = runner.analyze_results(results_df)

    print("\n최종 성적표:")
    print(final_scores.to_markdown(tablefmt="grid"))