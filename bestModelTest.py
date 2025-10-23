import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def set_korean_font():
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False

# ----------- 데이터 로딩 및 전처리 함수들 -----------
def load_male_files(folder_path):
    return [f for f in os.listdir(folder_path) if '_m_' in f and f.endswith('.csv')]

def compute_min_max(folder_path, file_list):
    all_values = []
    for file in file_list:
        arr = np.loadtxt(os.path.join(folder_path, file), delimiter=',')
        all_values.append(arr.flatten())
    all_values = np.concatenate(all_values)
    return all_values.min(), all_values.max()

def load_and_normalize_data(folder_path, file_list, data_min, data_max):
    data_list = []
    meta_list = []
    for file in file_list:
        parts = file.replace('.csv', '').split('_')
        date = parts[0]
        gender = parts[1]
        age = parts[2]
        hour = parts[-1]
        arr = np.loadtxt(os.path.join(folder_path, file), delimiter=',')
        arr_norm = (arr - data_min) / (data_max - data_min)
        data_list.append(arr_norm)
        meta_list.append({'date': date, 'hour': hour, 'age': age, 'gender': gender})
    df = pd.DataFrame(meta_list)
    df['data'] = data_list
    df = df.sort_values(['date', 'hour', 'age'])
    return df

# def build_sequences(df, window_size=3):
#     X_list = []
#     y_list = []
#     meta_y_list = []
#     for i in range(len(df) - window_size):
#         if (df.iloc[i]['date'] == df.iloc[i+window_size]['date']) and (df.iloc[i]['age'] == df.iloc[i+window_size]['age']):
#             hours = [int(df.iloc[i+j]['hour']) for j in range(window_size+1)]
#             if hours == list(range(hours[0], hours[0]+window_size+1)):
#                 X = [df.iloc[i+j]['data'] for j in range(window_size)]
#                 y = df.iloc[i+window_size]['data']
#                 X_list.append(np.stack(X))
#                 y_list.append(y)
#                 meta_y_list.append(df.iloc[i+window_size][['date', 'hour', 'age', 'gender']].to_dict())
#     return np.stack(X_list), np.stack(y_list), pd.DataFrame(meta_y_list)

# 시간 연속성 체크 빌드 시퀀스 함수
def build_sequences_cross_day(df, window_size=3):
    X_list = []
    y_list = []
    meta_y_list = []
    for i in range(len(df) - window_size):
        # 나이, 성별이 동일한지 확인
        if (df.iloc[i]['age'] == df.iloc[i+window_size]['age']) and (df.iloc[i]['gender'] == df.iloc[i+window_size]['gender']):
            # 시간 연속성 체크 (날짜 넘어가도 허용)
            hours = [int(df.iloc[i+j]['hour']) for j in range(window_size+1)]
            dates = [df.iloc[i+j]['date'] for j in range(window_size+1)]
            # 날짜+시간을 datetime으로 변환
            datetimes = pd.to_datetime([f"{d} {h:02d}" for d, h in zip(dates, hours)], format="%Y%m%d %H")
            # 연속성 체크
            if all((datetimes[j+1] - datetimes[j]).total_seconds() == 3600 for j in range(window_size)):
                X = [df.iloc[i+j]['data'] for j in range(window_size)]
                y = df.iloc[i+window_size]['data']
                X_list.append(np.stack(X))
                y_list.append(y)
                meta_y_list.append(df.iloc[i+window_size][['date', 'hour', 'age', 'gender']].to_dict())
    return np.stack(X_list), np.stack(y_list), pd.DataFrame(meta_y_list)

def split_data(X, y, meta_df):
    test_idx = meta_df['date'] == '20220228'
    X_test = X[test_idx]
    y_test = y[test_idx]
    X_test = np.transpose(X_test, (0, 2, 3, 1))
    y_test = y_test.reshape(-1, 72, 49, 1)
    meta_test = meta_df[test_idx].reset_index(drop=True)
    return X_test, y_test, meta_test

def denormalize(arr, data_min, data_max):
    return arr * (data_max - data_min) + data_min

# ----------- 예측 및 시각화 함수 -----------
# def predict_specific_time(model, X_test, y_test, meta_test, hour, data_min, data_max):
#     # hour는 문자열이어야 함 (예: '05', '12')
#     idx = (meta_test['hour'] == hour)
#     if idx.sum() == 0:
#         print(f"{hour}시에 해당하는 데이터가 없습니다.")
#         return
#     X_target = X_test[idx.values]
#     y_target = y_test[idx.values]
#     y_pred = model.predict(X_target)
#     y_pred_denorm = denormalize(y_pred, data_min, data_max)
#     y_target_denorm = denormalize(y_target, data_min, data_max)

#     plt.subplot(1, 2, 1)
#     plt.imshow(y_target_denorm[0, :, :, 0], cmap='viridis')
#     plt.title(f'Actual 2022-02-28 {hour}시')

#     plt.subplot(1, 2, 2)
#     plt.imshow(y_pred_denorm[0, :, :, 0], cmap='viridis')
#     plt.title(f'Predicted 2022-02-28 {hour}시')

#     plt.show()

def predict_specific_time(model, X_test, y_test, meta_test, hour, data_min, data_max):
    idx = (meta_test['hour'] == hour)
    if idx.sum() == 0:
        print(f"{hour}시에 해당하는 데이터가 없습니다.")
        return
    X_target = X_test[idx.values]
    y_target = y_test[idx.values]
    y_pred = model.predict(X_target)
    y_pred_denorm = denormalize(y_pred, data_min, data_max)
    y_target_denorm = denormalize(y_target, data_min, data_max)
    error = np.abs(y_target_denorm - y_pred_denorm)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(y_target_denorm[0, :, :, 0], cmap='viridis')
    plt.title(f'실제 2022-02-28 {hour}시')

    plt.subplot(1, 3, 2)
    plt.imshow(y_pred_denorm[0, :, :, 0], cmap='viridis')
    plt.title(f'예측 2022-02-28 {hour}시')

    plt.subplot(1, 3, 3)
    plt.imshow(error[0, :, :, 0], cmap='hot')
    plt.title(f'오차(절댓값) 2022-02-28 {hour}시')

    plt.tight_layout()
    plt.show()

# ----------- 메인 실행부 -----------
if __name__ == '__main__':
    folder_path = 'miniData'
    window_size = 3
    target_hour = '23'  # 예측하고 싶은 시간 (문자열, 예: '00', '05', '12', '23' 등)

    set_korean_font()

    # 데이터 준비
    file_list = load_male_files(folder_path)
    data_min, data_max = compute_min_max(folder_path, file_list)
    df = load_and_normalize_data(folder_path, file_list, data_min, data_max)
    # X, y, meta_df = build_sequences(df, window_size)
    X, y, meta_df = build_sequences_cross_day(df, window_size)
    X_test, y_test, meta_test = split_data(X, y, meta_df)

    # 모델 구조 정의 (학습 때와 동일하게!)
    input_shape = (72, 49, window_size)
    model = keras.models.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu", padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu", padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(256, 3, activation="relu", padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(256, 3, activation="relu", padding='same'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(72*49, activation='linear'),
        layers.Reshape((72, 49, 1))
    ])
    model.compile(optimizer='adam', loss='mse')

    # 베스트 모델 가중치 로드
    model.load_weights('best_model.keras')

    # 원하는 시간 예측 및 시각화
    predict_specific_time(model, X_test, y_test, meta_test, target_hour, data_min, data_max)

    # 정확도 계산 및 출력
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"target_hour: {target_hour}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    # print(meta_test['hour'].unique())

    #  # 원하는 시간 예측 및 시각화 (수정된 함수 사용)
    # predict_specific_time(model, X_test, y_test, meta_test, target_hour, data_min, data_max)