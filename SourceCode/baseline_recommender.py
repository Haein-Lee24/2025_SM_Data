# -*- coding: utf-8 -*-
"""
Baseline recommender: recommend programs to a student based on *lacking* (low) competencies.

How it works (simple & explainable):
1) Build each student's current competency profile by summing competencies from already *completed(이수)* programs.
2) Compute "need weights" for each competency — the lower the student's level on a competency, the higher its weight.
   - Option A (default): data-driven weights = (mean_of_student_levels - level). Below-mean comps get positive weight.
   - Option B: goal-driven gaps      = max(0, target_per_competency - level). Set `target_per_comp` if you prefer.
3) Score every candidate program = Σ (program_contribution_on_comp * need_weight_for_that_comp).
4) Exclude programs the student already completed. Return Top-K with transparent per-comp contribution & total score.

Expected inputs (CSV or DataFrame):
- programs_df: one row per program *offering* (카탈로그). Must include program identifier and competency gain columns.
- history_df : one row per program the student has (미)이수 기록. Must include the same competency columns + 학생ID + 이수여부.

Column auto-detection:
- Competency columns are detected by columns whose name starts with '핵심' (e.g., '핵심역량A', '핵심C', '핵심□'…).
- Program id is detected by the first column whose name contains '프로그램'.
- Student id column is detected by any of ['학생ID','학번','이수자ID'] (first match is used).
- Completion flag column is detected by any of ['이수여부','이수', '수료여부'] with values containing '이수'.

You can override all of these via function parameters.

Author: ChatGPT (baseline for quick start)
"""

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

def score_programs_for_student(programs_df: pd.DataFrame,
                               history_df: pd.DataFrame,
                               student_id: str,
                               top_k: int = 20,
                               use_target_gap: bool = False,
                               target_per_comp: float = 5.0,
                               program_id_col: Optional[str]=None,
                               student_id_col: Optional[str]=None,
                               completion_col: Optional[str]=None,
                               competency_cols: Optional[List[str]]=None
                              ) -> pd.DataFrame:
    """
    Return a scoring table with recommended programs for the student.
    Columns:
      - <program_id_col>, 'score', and per-competency contributions (need_weight * program_contrib)
    """
    program_id_col, _, _, comp_cols_prog = _detect_columns(programs_df, program_id_col, None, None, competency_cols)
    _, student_id_col, completion_col, comp_cols_hist = _detect_columns(history_df, None, student_id_col, completion_col, competency_cols)

    # Align competency column names (intersection)
    competency_cols = [c for c in comp_cols_prog if c in set(comp_cols_hist)]
    if len(competency_cols) == 0:
        raise ValueError("프로그램과 이력 데이터의 '핵심' 컬럼 교집합이 없습니다. 컬럼명을 확인하세요.")

    # Current student profile & need weights
    levels = build_student_profile(history_df, student_id, student_id_col, completion_col, competency_cols)
    if use_target_gap:
        need_w = _need_weights_from_targets(levels, target_per_comp)
    else:
        need_w = _need_weights_from_mean(levels)

    # Programs already completed -> exclude
    taken_prog_ids = set()
    if program_id_col and student_id_col:
        df_s = history_df[history_df[student_id_col].astype(str)==str(student_id)]
        if completion_col in df_s.columns:
            mask_completed = df_s[completion_col].astype(str).str.contains('이수', na=False)
            taken_prog_ids = set(df_s.loc[mask_completed, program_id_col].astype(str).tolist())

    # Score each program
    cand = programs_df.copy()
    if program_id_col is None:
        raise ValueError("프로그램 식별 컬럼을 찾지 못했습니다. program_id_col 파라미터로 지정해주세요.")

    # Clean competency numbers
    cand_comp = cand[competency_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Per-competency contributions = program_contrib * need_weight
    contrib = cand_comp.mul(need_w, axis=1)
    score = contrib.sum(axis=1)

    out = cand[[program_id_col]].copy()
    out['score'] = score
    # Add per-competency pieces for transparency
    for c in competency_cols:
        out[f'contrib::{c}'] = contrib[c]

    # Drop already completed programs
    if taken_prog_ids:
        out = out[~out[program_id_col].astype(str).isin(taken_prog_ids)]

    # Rank & return top_k
    out = out.sort_values('score', ascending=False).head(top_k).reset_index(drop=True)
    return out

if __name__ == "__main__":
    # --- Minimal CLI example (edit paths) ---
    # programs_df = pd.read_csv("programs.csv")      # 카탈로그
    # history_df  = pd.read_csv("student_history.csv")  # 수강/이수 이력
    # recs = score_programs_for_student(programs_df, history_df, student_id="0448919-9983430", top_k=15)
    # print(recs.to_string(index=False))
    pass
