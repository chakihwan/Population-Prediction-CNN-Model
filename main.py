import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# 남성 인구 CSV 파일 목록 불러오기
def load_male_files(folder_path):
    return [f for f in os.listdir(folder_path) if '_m_' in f and f.endswith('.csv')]

# 전체 데이터에서 최소값과 최대값 계산 (정규화를 위해)
def compute_min_max(folder_path, file_list):
    all_values = []
    for file in file_list:
        arr = np.loadtxt(os.path.join(folder_path, file), delimiter=',')
        all_values.append(arr.flatten())
    all_values = np.concatenate(all_values)
    return all_values.min(), all_values.max()

# 파일을 읽고 정규화한 뒤 DataFrame으로 구성
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

# 시계열 학습용 데이터를 입력(X)과 타겟(y) 쌍으로 구성
def build_sequences(df, window_size=3):
    X_list = []
    y_list = []
    meta_y_list = []

    for i in range(len(df) - window_size):
        if (df.iloc[i]['date'] == df.iloc[i+window_size]['date']) and (df.iloc[i]['age'] == df.iloc[i+window_size]['age']):
            hours = [int(df.iloc[i+j]['hour']) for j in range(window_size+1)]
            if hours == list(range(hours[0], hours[0]+window_size+1)):
                X = [df.iloc[i+j]['data'] for j in range(window_size)]
                y = df.iloc[i+window_size]['data']
                X_list.append(np.stack(X))
                y_list.append(y)
                meta_y_list.append(df.iloc[i+window_size][['date', 'hour', 'age', 'gender']].to_dict())

    return np.stack(X_list), np.stack(y_list), pd.DataFrame(meta_y_list)

# 날짜 기준으로 학습/검증/테스트 데이터 분할 및 리쉐이프
def split_data(X, y, meta_df):
    train_idx = meta_df['date'] <= '20220224'
    val_idx = (meta_df['date'] >= '20220225') & (meta_df['date'] <= '20220227')
    test_idx = meta_df['date'] == '20220228'

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    X_train = np.transpose(X_train, (0, 2, 3, 1))
    X_val = np.transpose(X_val, (0, 2, 3, 1))
    X_test = np.transpose(X_test, (0, 2, 3, 1))

    y_train = y_train.reshape(-1, 72, 49, 1)
    y_val = y_val.reshape(-1, 72, 49, 1)
    y_test = y_test.reshape(-1, 72, 49, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test

# CNN 모델 정의
def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, activation="relu", padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, activation="relu", padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(72*49, activation='linear')(x)
    outputs = layers.Reshape((72, 49, 1))(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# 모델 학습
def train_model(model,
                X_train, y_train, X_val, y_val, 
                epochs=200, 
                batch_size=32):
    checkpoint = ModelCheckpoint('best_model.keras', 
                                 monitor='val_loss', 
                                 save_best_only=True, 
                                 mode='min', 
                                 verbose=1)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[checkpoint])
    return history

# 학습 곡선 시각화
def plot_training_history(history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.show()

# 테스트셋 평가 및 예측 결과 시각화
def evaluate_and_visualize(model, X_test, y_test, data_min, data_max):
    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)

    y_pred = model.predict(X_test)

    # 역정규화
    y_pred_denorm = denormalize(y_pred, data_min, data_max)
    y_test_denorm = denormalize(y_test, data_min, data_max)

    plt.subplot(1, 2, 1)
    plt.imshow(y_test_denorm[0, :, :, 0], cmap='viridis')
    plt.title('Actual (denorm)')

    plt.subplot(1, 2, 2)
    plt.imshow(y_pred_denorm[0, :, :, 0], cmap='viridis')
    plt.title('Predicted (denorm)')

    plt.show()

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

def denormalize(arr, data_min, data_max):
    return arr * (data_max - data_min) + data_min

# 전체 실행
if __name__ == '__main__':

    folder_path = 'miniData'

    file_list = load_male_files(folder_path)
    data_min, data_max = compute_min_max(folder_path, file_list)

    df = load_and_normalize_data(folder_path, file_list, data_min, data_max)

    X, y, meta_df = build_sequences(df)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, meta_df)

    input_shape = (72, 49, 3)
    model = build_model(input_shape)
    model.summary()

    history = train_model(model, X_train, y_train, X_val, y_val)
    plot_training_history(history)
    evaluate_and_visualize(model, X_test, y_test, data_min, data_max)
    