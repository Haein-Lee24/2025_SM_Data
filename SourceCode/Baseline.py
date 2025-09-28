# Baseline.py (경로 문제 해결 최종 버전)

import os
import sys
import pandas as pd

# 'baseline_recommender.py' 파일이 같은 폴더에 있다고 가정합니다.
# 만약 다른 곳에 있다면 이 부분을 수정해야 할 수 있습니다.
try:
    from baseline_recommender import generate_recommendations_for_student
except ImportError:
    print("❌ 치명적 오류: 'baseline_recommender.py' 파일을 찾을 수 없습니다.")
    print("   해결책: 'Baseline.py'와 'baseline_recommender.py'가 같은 폴더에 있는지 확인해주세요.")
    sys.exit()

# --- [1] 기본 설정 ---
# [해결책] __file__을 사용해 현재 스크립트 파일의 절대 경로를 기준으로 폴더 경로를 설정합니다.
# 이렇게 하면 어떤 위치에서 실행해도 경로가 꼬이지 않습니다.
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Jupyter Notebook 등 __file__이 없는 환경을 위한 예외 처리
    ROOT = os.getcwd()

# 안전하게 상위 폴더로 이동하여 Data, Output 폴더 경로를 만듭니다.
DATA_DIR = os.path.join(ROOT, "..", "Data")
OUT_DIR = os.path.join(ROOT, "..", "Output")

# Output 폴더가 없으면 생성
if not os.path.exists(OUT_DIR):
    print(f"INFO: Output 폴더가 존재하지 않아 새로 생성합니다: {OUT_DIR}")
    os.makedirs(OUT_DIR)

# 최종 CSV 파일 경로
PROGRAMS_CSV = os.path.join(DATA_DIR, "program_remove.csv")
HISTORY_CSV = os.path.join(DATA_DIR, "student_remove.csv")

# 추천 관련 파라미터
TOP_K = 15
TARGET_PER_COMP = 5.0

# --- [2] 데이터 로딩 ---
print("--- [1/4] 데이터 로딩 시작 ---")
try:
    if not os.path.exists(PROGRAMS_CSV):
        raise FileNotFoundError(f"프로그램 파일을 찾을 수 없습니다: {PROGRAMS_CSV}")
    if not os.path.exists(HISTORY_CSV):
        raise FileNotFoundError(f"이수내역 파일을 찾을 수 없습니다: {HISTORY_CSV}")
        
    programs_df = pd.read_csv(PROGRAMS_CSV, encoding='utf-8-sig')
    history_df = pd.read_csv(HISTORY_CSV, encoding='utf-8-sig')
    print(f"✅ 프로그램 데이터 ({len(programs_df)} 행) 및 이수내역 데이터 ({len(history_df)} 행) 로딩 성공.")

except Exception as e:
    print(f"❌ 치명적 오류: 파일을 읽는 중 문제가 발생했습니다.")
    print(f"   오류 내용: {e}")
    sys.exit()

# --- [3] 전체 학생 대상 추천 실행 ---
print("\n--- [2/4] 전체 학생 추천 생성 시작 ---")

student_id_col_candidates = ["학생ID", "학번", "이수자ID", "Student", "student"]
student_id_col = next((c for c in history_df.columns if any(key in str(c).strip() for key in student_id_col_candidates)), None)

if student_id_col is None:
    print("❌ 치명적 오류: 이수내역 데이터(student_remove.csv)에서 학생 ID 컬럼을 찾을 수 없습니다.")
    sys.exit()
else:
    print(f"INFO: 학생 ID로 사용할 컬럼을 찾았습니다: '{student_id_col}'")

all_student_ids = history_df[student_id_col].astype(str).unique().tolist()
all_results = []
total_students = len(all_student_ids)

if total_students == 0:
    print("⚠️ 경고: 처리할 학생 데이터가 없습니다.")
else:
    print(f"INFO: 총 {total_students} 명의 학생에 대한 추천을 시작합니다.")
    for i, sid in enumerate(all_student_ids):
        try:
            recs_sid = generate_recommendations_for_student(
                programs_df=programs_df, history_df=history_df, student_id=str(sid),
                top_k=TOP_K, target_per_comp=TARGET_PER_COMP
            )
            
            if not recs_sid.empty:
                recs_sid.insert(0, "추천대상ID", sid)
                recs_sid['추천순위'] = recs_sid.groupby('추천방식').cumcount() + 1
                all_results.append(recs_sid)
                
            if (i + 1) % 10 == 0 or (i + 1) == total_students:
                 print(f"  - 진행률: {i+1}/{total_students} ({((i+1)/total_students)*100:.1f}%)")

        except Exception as e:
            print(f"  - ⚠️ 학생 {sid} 처리 중 개별 오류 발생: {e}")

# --- [4] 최종 결과 저장 ---
print("\n--- [3/4] 최종 결과 파일 저장 시작 ---")
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, f"추천결과_전체학생_통합_top{TOP_K}.csv")
    
    try:
        final_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 추천 결과 {len(final_df)} 건을 성공적으로 저장했습니다.")
        print(f"   💾 저장 위치: {out_csv}")
    except Exception as e:
        print(f"❌ 치명적 오류: 최종 CSV 파일을 저장하는 데 실패했습니다.")
        print(f"   오류 내용: {e}")
        
else:
    print("⚠️ 최종 결과 없음: 모든 학생에 대한 추천 결과가 생성되지 않았습니다.")

print("\n--- [4/4] 모든 프로세스 종료 ---")