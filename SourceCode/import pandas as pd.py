import pandas as pd

try:
    # 1. 두 개의 추천 목록 CSV 파일을 각각 불러옵니다.
    df_major = pd.read_csv('major_recommendations.csv')
    df_personalized = pd.read_csv('personalized_recommendations.csv')
    print("파일을 성공적으로 불러왔습니다.")

    # 2. 각 데이터에 추천 유형을 명시하는 '추천유형' 열을 추가합니다.
    #    (전공 추천 / 개인 맞춤 추천)
    df_major['추천유형'] = '전공 추천'
    df_personalized['추천유형'] = '개인 맞춤 추천'

    # 3. 두 데이터를 위아래로 합칩니다. (이수자별 20개 목록 생성)
    #    'rank' 열은 각 추천 유형 내에서의 순위를 의미하게 됩니다.
    df_combined = pd.concat([df_major, df_personalized], ignore_index=True)

    # 4. '이수자ID'를 기준으로 데이터를 정렬하여 보기 좋게 만듭니다.
    #    같은 이수자 내에서는 '추천유형', 'rank' 순으로 정렬합니다.
    df_final = df_combined.sort_values(by=['이수자ID', '추천유형', 'rank'], ascending=[True, True, True])

    # 5. 최종 결과를 'final_recommendations.csv' 파일로 저장합니다.
    output_filename = 'final_recommendations.csv'
    df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"'{output_filename}' 파일으로 최종 추천 목록 생성이 완료되었습니다.")
    print("\n----- 최종 데이터 정보 -----")
    # 이수자 ID 하나를 선택하여 결과가 잘 나왔는지 확인합니다.
    first_student_id = df_final['이수자ID'].iloc[0]
    print(f"\n'{first_student_id}' 학생의 추천 목록 예시:")
    print(df_final[df_final['이수자ID'] == first_student_id][['이수자ID', '프로그램명', '추천유형', 'rank']].to_string())


except FileNotFoundError as e:
    print(f"오류: {e}")
    print("스크립트와 동일한 폴더에 CSV 파일('major_recommendations.csv', 'personalized_recommendations.csv')이 있는지 확인해주세요.")
except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")