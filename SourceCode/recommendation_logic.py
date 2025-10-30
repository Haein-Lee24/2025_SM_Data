# -*- coding: utf-8 -*-
"""
Reproduce personalized_recommendations_v3.csv and major_recommendations_v3.csv
from the unified source: (삭제완료) student_program_total.csv

- Input:
    /mnt/data/(삭제완료) student_program_total.csv

- Outputs:
    /mnt/data/personalized_recommendations_v3.csv
    /mnt/data/major_recommendations_v3.csv
"""

import os
import re
from typing import List, Set, Dict, Tuple
import numpy as np
import pandas as pd

# ---------- Helpers ----------

def read_csv_kr(path: str) -> pd.DataFrame:
    """Robust CSV reader for Korean encodings."""
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # last attempt without encoding arg
    return pd.read_csv(path)

def tokenize(text: str) -> List[str]:
    """Simple multilingual tokenizer with light synonym mapping."""
    if pd.isna(text):
        return []
    s = str(text)
    s = (
        s.replace(",", " ")
         .replace("/", " ")
         .replace("|", " ")
         .replace("·", " ")
         .replace("•", " ")
    )
    toks = re.split(r"[^\w가-힣+#]+", s)
    toks = [t.strip().lower() for t in toks if len(t.strip()) >= 2]
    mapping = {
        "ai": "ai", "인공지능": "ai",
        "머신러닝": "머신러닝", "ml": "머신러닝",
        "데이터": "데이터", "data": "데이터",
        "python": "python", "파이썬": "python",
        "취업": "취업", "진로": "진로", "멘토링": "멘토링",
        "공모전": "공모전", "경진대회": "공모전",
        "deeplearning": "딥러닝", "딥러닝": "딥러닝",
        "빅데이터": "데이터",
    }
    toks = [mapping.get(t, t) for t in toks]
    # unique while preserving order
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def jaccard_like(a: Set[str], b: Set[str]) -> float:
    if not b:
        return 0.0
    return len(a & b) / max(1, len(b))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def norm01_to100(v: np.ndarray) -> np.ndarray:
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if vmax - vmin == 0:
        return np.ones_like(v) * 50.0
    return (v - vmin) / (vmax - vmin) * 100.0

def mode_or_nan(s: pd.Series):
    s = s.dropna().astype(str)
    if len(s) == 0:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]

def expand_major_tokens(maj_name: str) -> Set[str]:
    toks = set(tokenize(maj_name))
    expanded = set(toks)
    m = maj_name.lower() if isinstance(maj_name, str) else ""
    # very lightweight expansion rules (can be replaced with a real dictionary)
    if any(x in m for x in ["ai", "인공지능", "data", "데이터", "컴퓨터", "software", "소프트", "정보", "it"]):
        expanded.update({"ai", "인공지능", "데이터", "머신러닝", "python", "ml"})
    if any(x in m for x in ["stat", "통계"]):
        expanded.update({"데이터", "통계", "분석"})
    if any(x in m for x in ["경영", "경제", "상경", "biz", "business"]):
        expanded.update({"경영", "기업", "회계", "재무", "마케팅"})
    if any(x in m for x in ["교육"]):
        expanded.update({"교육", "멘토링", "교원", "교직"})
    if any(x in m for x in ["약학", "생명", "화공", "bio", "chem"]):
        expanded.update({"실험", "연구", "논문", "분석"})
    return expanded

# ---------- I/O ----------

SRC = "/mnt/data/(삭제완료) student_program_total.csv"
OUT_PERS = "/mnt/data/personalized_recommendations_v3.csv"
OUT_MAJ = "/mnt/data/major_recommendations_v3.csv"

df = read_csv_kr(SRC)

# ---------- Column bindings (from the unified source schema) ----------

cid   = "student_id"
pid   = "program_id"
pname = "program_name"
ptopic= "program_topic"
ptag  = "program_tag"
ptype = "program_type1"
satis = "satisfaction_avg"
status= "completion_status"
major = "major"
grade = "grade"
comp_cols = [c for c in df.columns if re.fullmatch(r"핵심역량점수[1-8]", c)]

# ---------- Completed filter ----------

completed_mask = (
    df[status].astype(str).str.lower().isin(["completed", "complete", "y", "yes", "1", "true"])
    | df[status].astype(str).str.contains("이수", na=False)
)
df_c = df.loc[completed_mask].copy()

# ---------- Program-level table ----------

# Per-program competency vector (mean across rows)
prog_comp = df.groupby(pid)[comp_cols].mean().fillna(0.0)
# Program metadata (first non-null values)
prog_meta = df.groupby(pid).agg({pname: "first", ptopic: "first", ptag: "first", ptype: "first"}).reset_index()
# Program satisfaction from COMPLETED only; impute by type
prog_satis = df_c.groupby(pid)[satis].mean().rename("program_satisfaction").reset_index()

programs = (
    prog_meta
    .merge(prog_comp.reset_index(), on=pid, how="left")
    .merge(prog_satis, on=pid, how="left")
)

# Impute satisfaction by type average (fallback to overall avg)
type_avg = programs.groupby(ptype)["program_satisfaction"].mean().to_dict()
overall_avg = programs["program_satisfaction"].mean()
programs["program_satisfaction"] = programs.apply(
    lambda r: r["program_satisfaction"]
    if pd.notna(r["program_satisfaction"])
    else type_avg.get(r.get(ptype), overall_avg),
    axis=1,
)

# Token sets + comp_vec
programs["token_set"] = programs[[pname, ptopic, ptag]].apply(
    lambda r: set(tokenize(" ".join([str(x) for x in r if pd.notna(x)]))), axis=1
)
programs["comp_vec"] = programs[comp_cols].fillna(0.0).values.tolist()

# ---------- Student profiles ----------

# Interests: union of token sets from programs they COMPLETED
df_c_prog = df_c.merge(programs[[pid, "token_set"]], on=pid, how="left")
stu_interest = df_c_prog.groupby(cid)["token_set"].apply(
    lambda sets: set().union(*[s for s in sets if isinstance(s, set)]) if len(sets) > 0 else set()
)

# Student competency: mean of comp vectors of COMPLETED programs (merge with suffix to avoid collisions)
merged = df_c.merge(programs[[pid] + comp_cols], on=pid, how="left", suffixes=("", "_prog"))
prog_comp_cols = [c + "_prog" for c in comp_cols]
stu_comp = merged.groupby(cid)[prog_comp_cols].mean().fillna(0.0)
stu_comp["comp_vec"] = stu_comp[prog_comp_cols].values.tolist()

# Major / Grade: mode across records
stu_major = df.groupby(cid)[major].apply(mode_or_nan)
stu_grade = df.groupby(cid)[grade].apply(mode_or_nan)

# Assemble student table
students = pd.DataFrame({cid: df[cid].drop_duplicates().values}).set_index(cid)
students = (
    students.join(stu_major.rename(major), how="left")
            .join(stu_grade.rename(grade), how="left")
            .join(stu_interest.rename("interest_tokens"), how="left")
            .join(stu_comp[["comp_vec"]], how="left")
)
students["interest_tokens"] = students["interest_tokens"].apply(lambda x: x if isinstance(x, set) else set())
students["comp_vec"] = students["comp_vec"].apply(lambda x: x if isinstance(x, list) else [0.0] * len(comp_cols))
students = students.reset_index()

# ---------- Prebuild arrays ----------

P = programs.copy()
P_ids    = P[pid].tolist()
P_names  = P[pname].astype(str).tolist()
P_tags   = P[ptag].astype(str).tolist()
P_types  = P[ptype].astype(str).tolist()
P_satis  = P["program_satisfaction"].astype(float).values
P_comp   = np.vstack(P["comp_vec"].values)
P_tokens = P["token_set"].values

# ---------- Personalized recommendations (Top 10 per student) ----------

personalized_rows: List[Dict] = []

for _, r in students.iterrows():
    sid = r[cid]
    maj = str(r.get(major, ""))
    grd = str(r.get(grade, ""))
    s_tokens = set(tokenize(maj)) | set(tokenize(grd)) | (r.get("interest_tokens") or set())
    s_vec = np.array(r.get("comp_vec", [0.0] * P_comp.shape[1]), dtype=float)
    if s_vec.size != P_comp.shape[1]:
        s_vec = np.zeros(P_comp.shape[1], dtype=float)

    gap_raw  = np.maximum(0, P_comp - s_vec).sum(axis=1)
    pers_raw = np.array([jaccard_like(s_tokens, ts if isinstance(ts, set) else set()) for ts in P_tokens], dtype=float)
    satis_raw= P_satis.copy()

    gap_n   = norm01_to100(gap_raw)
    pers_n  = norm01_to100(pers_raw)
    satis_n = norm01_to100(satis_raw)

    final = gap_n * 0.5 + satis_n * 0.3 + pers_n * 0.2
    top_idx = np.argsort(-final)[:10]

    for rank, i in enumerate(top_idx, start=1):
        personalized_rows.append({
            "이수자ID": sid,
            "전공명": maj,
            "학년": grd,
            "프로그램ID": P_ids[i],
            "프로그램명": P_names[i],
            "프로그램 태그": P_tags[i],
            "프로그램유형1": P_types[i],
            "gap_score": round(float(gap_n[i]), 4),
            "satisfaction_score": round(float(satis_n[i]), 4),
            "personalization_score": round(float(pers_n[i]), 4),
            "final_score": round(float(final[i]), 4),
            "rank": rank,
        })

personalized_df = pd.DataFrame(personalized_rows)
personalized_df.to_csv(OUT_PERS, index=False, encoding="utf-8-sig")

# ---------- Major-based recommendations (Top 10 per major) ----------

# Major competency profile: mean of comps (completed) for that major
maj_comp = (
    df_c.merge(programs[[pid] + comp_cols], on=pid, how="left", suffixes=("", "_prog"))
        .groupby(major)[[c + "_prog" for c in comp_cols]].mean().fillna(0.0)
)
maj_comp["comp_vec"] = maj_comp[[c + "_prog" for c in comp_cols]].values.tolist()

# History share: for each (program, major), completed count share
cnt_pm = df_c.groupby([pid, major]).size().rename("complete_cnt").reset_index()
cnt_p  = df_c.groupby(pid).size().rename("complete_total").reset_index()
hist_df = cnt_pm.merge(cnt_p, on=pid, how="left")
hist_df["major_history_share"] = (hist_df["complete_cnt"] / hist_df["complete_total"]).fillna(0.0)

major_rows: List[Dict] = []
majors_list = students[major].dropna().unique().tolist()
prog_comp_mat = P_comp
prog_token_sets = P_tokens

for maj in majors_list:
    maj_tokens = expand_major_tokens(str(maj))

    if maj in maj_comp.index:
        m_vec = np.array(maj_comp.loc[maj, "comp_vec"], dtype=float)
    else:
        m_vec = np.zeros(len(comp_cols), dtype=float)

    tag_raw = np.array(
        [jaccard_like(maj_tokens, ts if isinstance(ts, set) else set()) for ts in prog_token_sets],
        dtype=float,
    )
    comp_raw = np.array(
        [cosine_sim(m_vec, prog_comp_mat[i, :]) for i in range(prog_comp_mat.shape[0])],
        dtype=float,
    )
    # history for this major → align to program_id order
    hist_map = hist_df[hist_df[major] == maj].set_index(pid)["major_history_share"].to_dict()
    hist_raw = np.array([hist_map.get(P_ids[i], 0.0) for i in range(len(P_ids))], dtype=float)

    tag_n  = norm01_to100(tag_raw)
    comp_n = norm01_to100(comp_raw)
    hist_n = norm01_to100(hist_raw)

    major_score = tag_n * 0.3 + comp_n * 0.4 + hist_n * 0.3
    top_idx = np.argsort(-major_score)[:10]

    # stash top programs for this major
    recs = [{
        "전공명": maj,
        "프로그램ID": P_ids[i],
        "프로그램명": P_names[i],
        "프로그램 태그": P_tags[i],
        "프로그램유형1": P_types[i],
        "major_tag_score": round(float(tag_n[i]), 4),
        "major_competency_score": round(float(comp_n[i]), 4),
        "major_history_score": round(float(hist_n[i]), 4),
        "major_score": round(float(major_score[i]), 4),
        "rank": rk + 1,
    } for rk, i in enumerate(top_idx)]

    # assign to all students with this major
    sids = students.loc[students[major] == maj, cid].tolist()
    for rec in recs:
        for sid in sids:
            row = rec.copy()
            row["이수자ID"] = sid
            major_rows.append(row)

major_df = pd.DataFrame(major_rows)
major_df.to_csv(OUT_MAJ, index=False, encoding="utf-8-sig")

print("Wrote:", OUT_PERS)
print("Wrote:", OUT_MAJ)
