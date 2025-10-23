import csv
import os, shutil, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import glob
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from datetime import datetime
import matplotlib.font_manager as fm
from collections import defaultdict

#  폰트 설정 함수
def set_korean_font():
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False

#  파일 목록 필터링
def get_male_files(folder_path):
    all_files = os.listdir(folder_path)
    male_files = sorted([f for f in all_files if '_m_' in f and f.endswith('.csv')])
    return male_files

# 날짜별 통계 계산
def calculate_daily_stats(folder_path, male_files):
    daily_stats = {}

    for filename in male_files:
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath, header=None)

        date = filename.split('_')[0]
        value_sum = df.values.sum()
        value_count = df.size

        if date not in daily_stats:
            daily_stats[date] = {'total': value_sum, 'count': value_count}
        else:
            daily_stats[date]['total'] += value_sum
            daily_stats[date]['count'] += value_count

    return daily_stats
# 시간대별 통계 계산
def calculate_hourly_stats(folder_path, male_files):
    hourly_stats = defaultdict(lambda: {'total': 0, 'count': 0})

    for filename in male_files:
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath, header=None)

        hour = filename.split('_')[-1].replace('.csv', '')
        value_sum = df.values.sum()
        value_count = df.size

        hourly_stats[hour]['total'] += value_sum
        hourly_stats[hour]['count'] += value_count

    return hourly_stats

# 요일별 평균 인구수 분석
def calculate_weekday_stats(daily_stats):
    weekday_totals = defaultdict(lambda: {'total': 0, 'count': 0})
    for date_str, stats in daily_stats.items():
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        weekday = date_obj.strftime('%A')  # Monday, Tuesday, ...
        weekday_totals[weekday]['total'] += stats['total']
        weekday_totals[weekday]['count'] += stats['count']
    
    weekday_avg = {
        k: v['total'] / v['count'] for k, v in weekday_totals.items()
    }
    return weekday_avg

# 시간대별 분산 분석 (시간대별 변화의 정도)
def calculate_hourly_variance(folder_path, male_files):
    hourly_values = defaultdict(list)

    for filename in male_files:
        hour = filename.split('_')[-1].replace('.csv', '')
        df = pd.read_csv(os.path.join(folder_path, filename), header=None)
        hourly_values[hour].append(df.values.mean())  # or .sum()

    hourly_std = {
        hour: np.std(vals) for hour, vals in hourly_values.items()
    }
    return hourly_std

# 가장 붐비는 시간대 및 날짜 Top-N 추출
def find_peak_times(daily_stats, hourly_stats, top_n=3):
    top_days = sorted(
        daily_stats.items(), key=lambda x: x[1]['total'] / x[1]['count'], reverse=True
    )[:top_n]
    top_hours = sorted(
        hourly_stats.items(), key=lambda x: x[1]['total'] / x[1]['count'], reverse=True
    )[:top_n]

    return top_days, top_hours

#  날짜별 시각화 함수
def plot_daily_avg(daily_stats):
    dates = sorted(daily_stats.keys())
    averages = [daily_stats[d]['total'] / daily_stats[d]['count'] for d in dates]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, averages, marker='o', color='blue')
    plt.title('남성 2529 연령대 일별 평균 값')
    plt.xlabel('날짜')
    plt.ylabel('평균 값')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#  시간대별 시각화 함수
def plot_hourly_avg(hourly_stats):
    sorted_hours = sorted(hourly_stats.keys())
    averages = [hourly_stats[h]['total'] / hourly_stats[h]['count'] for h in sorted_hours]

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_hours, averages, marker='o', color='green')
    plt.title('시간대별 평균 값 (남성 2529 연령대)')
    plt.xlabel('시간대')
    plt.ylabel('평균 값')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 요일별 평균 인구수 시각화
def plot_weekday_stats(weekday_avg):
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    values = [weekday_avg.get(day, 0) for day in ordered_days]
    plt.figure(figsize=(10,5))
    plt.bar(ordered_days, values, color='skyblue')
    plt.title('요일별 평균 인구수')
    plt.ylabel('평균값')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()

# 시간대별 표준편차 분석 시각화
def plot_hourly_std(hourly_std):
    hours = sorted(hourly_std.keys())
    stds = [hourly_std[h] for h in hours]
    plt.figure(figsize=(10, 5))
    plt.plot(hours, stds, marker='o', color='orange')
    plt.title('시간대별 인구 변화 표준편차')
    plt.xlabel('시간대')
    plt.ylabel('표준편차')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_weekday_total_and_avg(daily_stats):
    weekday_totals = defaultdict(lambda: {'total': 0, 'count': 0})

    for date_str, stats in daily_stats.items():
        weekday = datetime.strptime(date_str, "%Y%m%d").strftime('%A')
        weekday_totals[weekday]['total'] += stats['total']
        weekday_totals[weekday]['count'] += stats['count']

    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    totals = [weekday_totals[d]['total'] for d in ordered_days]
    avgs = [weekday_totals[d]['total'] / weekday_totals[d]['count'] for d in ordered_days]

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.bar(ordered_days, totals, color='tomato')
    plt.title('요일별 총 인구수')
    plt.ylabel('총합')

    plt.subplot(1, 2, 2)
    plt.bar(ordered_days, avgs, color='skyblue')
    plt.title('요일별 평균 인구수')
    plt.ylabel('평균')

    plt.tight_layout()
    plt.show()

def print_weekday_summary(weekday_avg):
    print("요일별 평균 인구수 통계 요약")
    for day, avg in weekday_avg.items():
        print(f"{day}: {avg:.2f}")
    
    avg_values = list(weekday_avg.values())
    print("\n전체 평균:", np.mean(avg_values))
    print("표준편차:", np.std(avg_values))
    print("최댓값:", np.max(avg_values))
    print("최솟값:", np.min(avg_values))


#  메인 
if __name__ == "__main__":
    set_korean_font()
    folder_path = './miniData'

    male_files = get_male_files(folder_path)
    # print(f"남성 파일 개수: {len(male_files)}")

    daily_stats = calculate_daily_stats(folder_path, male_files)
    hourly_stats = calculate_hourly_stats(folder_path, male_files)

    # # 시각화
    # plot_daily_avg(daily_stats)
    # plot_hourly_avg(hourly_stats)

    # # 요일별 분석
    weekday_avg = calculate_weekday_stats(daily_stats)
    # plot_weekday_stats(weekday_avg)
    # plot_weekday_total_and_avg(daily_stats)

    # # 시간대별 표준편차 분석
    hourly_std = calculate_hourly_variance(folder_path, male_files)
    # plot_hourly_std(hourly_std)

    # # 피크 타임 출력
    # top_days, top_hours = find_peak_times(daily_stats, hourly_stats)

    # print("\n가장 붐볐던 날짜 Top-3:")
    # for date, stat in top_days:
    #     avg = stat['total'] / stat['count']
    #     print(f"{date} - 평균값: {avg:.2f}")

    # print("\n가장 붐볐던 시간대 Top-3:")
    # for hour, stat in top_hours:
    #     avg = stat['total'] / stat['count']
    #     print(f"{hour}시 - 평균값: {avg:.2f}")

weekday_avg = calculate_weekday_stats(daily_stats)
print_weekday_summary(weekday_avg)
plot_weekday_stats(weekday_avg)