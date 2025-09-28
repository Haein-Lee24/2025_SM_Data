# =================================================================
# baseline_recommender.py : 부족 역량 기반 추천 알고리즘
# =================================================================
# 역할:
# 1. 특정 학생의 현재 역량 프로필(점수) 계산
# 2. '평균 대비'와 '목표 대비' 두 방식으로 부족 역량 가중치 산출
# 3. 각 프로그램의 점수를 계산하고 Top-K 추천 목록 생성
# ---------------------------------------------------------------

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        for col in cols:
            if c in str(col):
                return col
    return None

def _detect_columns(df: pd.DataFrame,
                    program_id_col: Optional[str]=None,
                    student_id_col: Optional[str]=None,
                    completion_col: Optional[str]=None,
                    competency_cols: Optional[List[str]]=None
                   ) -> Tuple[str, Optional[str], Optional[str], List[str]]:
    cols = list(df.columns)
    if competency_cols is None:
        competency_cols = [c for c in cols if str(c).startswith('핵심')]
    if program_id_col is None:
        program_id_col = _find_col(cols, ['프로그램', '프로그', 'Program', 'program'])
    if student_id_col is None:
        student_id_col = _find_col(cols, ['학생ID','학번','이수자ID','Student','student'])
    if completion_col is None:
        completion_col = _find_col(cols, ['이수여부','이수', '수료여부'])
    return program_id_col, student_id_col, completion_col, competency_cols

def _need_weights_from_mean(student_levels: pd.Series) -> pd.Series:
    # Below-mean competencies get positive weight; above-mean -> smaller or 0 weight
    mu = student_levels.mean() if len(student_levels) else 0.0
    w = (mu - student_levels).clip(lower=0)
    # If all zeros (flat profile), fall back to equal weights
    if (w.sum() == 0) and len(student_levels) > 0:
        w[:] = 1.0
    return w

def _need_weights_from_targets(student_levels: pd.Series, target_per_comp: float) -> pd.Series:
    target = pd.Series([target_per_comp]*len(student_levels), index=student_levels.index)
    w = (target - student_levels).clip(lower=0)
    if (w.sum() == 0) and len(student_levels) > 0:
        w[:] = 1.0
    return w

def build_student_profile(history_df: pd.DataFrame,
                          student_id: str,
                          student_id_col: Optional[str]=None,
                          completion_col: Optional[str]=None,
                          competency_cols: Optional[List[str]]=None
                         ) -> pd.Series:
    """Return student's accumulated competency levels (sum over completed rows)."""
    _, student_id_col, completion_col, competency_cols = _detect_columns(history_df,
                                                                         None,
                                                                         student_id_col,
                                                                         completion_col,
                                                                         competency_cols)
    if student_id_col is None:
        raise ValueError("학생 ID 컬럼을 찾지 못했습니다. student_id_col 파라미터로 지정해주세요.")
    if competency_cols is None or len(competency_cols)==0:
        raise ValueError("핵심역량 컬럼을 찾지 못했습니다. competency_cols 파라미터로 지정해주세요.")

    df_s = history_df[history_df[student_id_col].astype(str)==str(student_id)].copy()
    if len(df_s)==0:
        # No history → zero vector
        return pd.Series([0.0]*len(competency_cols), index=competency_cols)

    if completion_col is not None:
        # Consider records whose completion text contains '이수'
        mask_completed = df_s[completion_col].astype(str).str.contains('이수', na=False)
        df_s = df_s[mask_completed] if mask_completed.any() else df_s

    profile = df_s[competency_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=0)
    return profile

def generate_recommendations_for_student(programs_df: pd.DataFrame,
                                         history_df: pd.DataFrame,
                                         student_id: str,
                                         top_k: int = 20,
                                         target_per_comp: float = 5.0,
                                         program_id_col: Optional[str] = None,
                                         student_id_col: Optional[str] = None,
                                         completion_col: Optional[str] = None,
                                         competency_cols: Optional[List[str]] = None
                                        ) -> pd.DataFrame:
    """
    [수정된 함수] 학생 한 명에 대해 '평균기반'과 '목표기반' 추천을 모두 생성하고,
    두 결과를 합쳐서 하나의 DataFrame으로 반환합니다.
    """
    program_id_col, _, _, comp_cols_prog = _detect_columns(programs_df, program_id_col, None, None, competency_cols)
    _, student_id_col, completion_col, comp_cols_hist = _detect_columns(history_df, None, student_id_col, completion_col, competency_cols)

    competency_cols = [c for c in comp_cols_prog if c in set(comp_cols_hist)]
    if not competency_cols:
        raise ValueError("프로그램과 이력 데이터의 '핵심' 컬럼 교집합이 없습니다. 컬럼명을 확인하세요.")

    # 1. 학생 현재 역량 프로필 생성
    levels = build_student_profile(history_df, student_id, student_id_col, completion_col, competency_cols)
    
    # 2. 두 가지 방식의 '부족 역량 가중치' 계산
    need_w_mean = _need_weights_from_mean(levels)
    need_w_target = _need_weights_from_targets(levels, target_per_comp)
    
    # 3. 이수한 프로그램 목록 제외 처리
    taken_prog_ids = set()
    if program_id_col and student_id_col:
        df_s = history_df[history_df[student_id_col].astype(str) == str(student_id)]
        if completion_col in df_s.columns:
            mask_completed = df_s[completion_col].astype(str).str.contains('이수', na=False)
            taken_prog_ids = set(df_s.loc[mask_completed, program_id_col].astype(str).tolist())

    cand = programs_df.copy()
    if program_id_col is None:
        raise ValueError("프로그램 식별 컬럼을 찾지 못했습니다. program_id_col 파라미터로 지정해주세요.")
    
    cand = cand[~cand[program_id_col].astype(str).isin(taken_prog_ids)]
    cand_comp = cand[competency_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    all_recs = []
    # 4. 각 방식별로 점수 계산 및 결과 생성
    for method, weights in [('평균기반', need_w_mean), ('목표기반', need_w_target)]:
        contrib = cand_comp.mul(weights, axis=1)
        score = contrib.sum(axis=1)

        out = cand[[program_id_col]].copy()
        out['추천방식'] = method
        out['총점'] = score
        
        # 투명성을 위해 각 역량별 기여 점수 추가
        for c in competency_cols:
            out[f'기여점수::{c}'] = contrib[c]
        
        # 점수 높은 순으로 정렬 후 Top-K 선택
        recs = out.sort_values('총점', ascending=False).head(top_k).reset_index(drop=True)
        all_recs.append(recs)

    # 5. 두 방식의 추천 결과를 하나로 합쳐서 반환
    return pd.concat(all_recs, ignore_index=True)


if __name__ == "__main__":
    # --- Minimal CLI example (edit paths) ---
    # programs_df = pd.read_csv("programs.csv")      # 카탈로그
    # history_df  = pd.read_csv("student_history.csv")  # 수강/이수 이력
    # recs = score_programs_for_student(programs_df, history_df, student_id="0448919-9983430", top_k=15)
    # print(recs.to_string(index=False))
    pass
