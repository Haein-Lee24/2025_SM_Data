# -*- coding: utf-8 -*-
# Baseline.py — CSV 기반 '부족 역량' 추천 (결측치 없는 버전)

import os
import pandas as pd
from baseline_recommender import score_programs_for_student

# [1] 경로 설정
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "..","Data")
OUT_DIR  = os.path.join(ROOT,".." "Output")
os.makedirs(OUT_DIR, exist_ok=True)

# CSV 파일 경로
PROGRAMS_CSV = os.path.join(DATA_DIR, "program_remove.csv")
HISTORY_CSV  = os.path.join(DATA_DIR, "student_remove.csv")

# [2] CSV 읽기 (결측치 없는 파일이라고 가정)
programs_df = pd.read_csv(PROGRAMS_CSV, encoding="utf-8-sig")
history_df  = pd.read_csv(HISTORY_CSV,  encoding="utf-8-sig")

# [3] 추천 실행 모드
STUDENT_ID = None      # 단일 학생 테스트: "0448919-9983430"
TOP_K = 15
USE_TARGET_GAP = False # True → 목표치 기반, False → 평균 대비 부족 가중치
TARGET_PER_COMP = 5.0  # 목표치 기반일 때 사용

# [4] 추천 실행
if STUDENT_ID:
    # 단일 학생 추천
    recs = score_programs_for_student(
        programs_df=programs_df,
        history_df=history_df,
        student_id=str(STUDENT_ID),
        top_k=TOP_K,
        use_target_gap=USE_TARGET_GAP,
        target_per_comp=TARGET_PER_COMP
    )
    out_csv  = os.path.join(OUT_DIR, f"추천결과_{STUDENT_ID}_top{TOP_K}.csv")
    out_xlsx = os.path.join(OUT_DIR, f"추천결과_{STUDENT_ID}_top{TOP_K}.xlsx")
    recs.to_csv(out_csv, index=False, encoding="utf-8-sig")
    recs.to_excel(out_xlsx, index=False)
    print(f"[저장] {out_csv}")
    print(f"[저장] {out_xlsx}")

else:
    # 전체 학생 추천
    student_id_col_candidates = ["학생ID", "학번", "이수자ID", "Student", "student"]
    student_id_col = None
    for c in history_df.columns:
        if any(key in str(c) for key in student_id_col_candidates):
            student_id_col = c
            break
    if student_id_col is None:
        raise ValueError("학생ID 컬럼을 찾지 못했습니다.")

    all_ids = history_df[student_id_col].astype(str).unique().tolist()

    results = []
    for sid in all_ids:
        try:
            recs_sid = score_programs_for_student(
                programs_df=programs_df,
                history_df=history_df,
                student_id=str(sid),
                top_k=TOP_K,
                use_target_gap=USE_TARGET_GAP,
                target_per_comp=TARGET_PER_COMP
            ).copy()
            recs_sid.insert(0, "학생ID", sid)
            recs_sid.insert(1, "rank", range(1, len(recs_sid)+1))
            results.append(recs_sid)
        except Exception as e:
            print(f"[경고] 학생 {sid} 처리 실패: {e}")

    if results:
        all_recs = pd.concat(results, ignore_index=True)
        out_csv  = os.path.join(OUT_DIR, f"추천결과_전체학생_top{TOP_K}.csv")
        out_xlsx = os.path.join(OUT_DIR, f"추천결과_전체학생_top{TOP_K}.xlsx")
        all_recs.to_csv(out_csv, index=False, encoding="utf-8-sig")
        all_recs.to_excel(out_xlsx, index=False)
        print(f"[저장] {out_csv}")
        print(f"[저장] {out_xlsx}")
    else:
        print("[정보] 추천 결과 없음. 입력 데이터 확인 필요.")
