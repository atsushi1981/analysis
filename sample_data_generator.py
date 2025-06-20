#!/usr/bin/env python3
"""
サンプルデータ生成スクリプト
テスト用の異常を含むCSVファイルを生成
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_normal_signal(length, frequency=1.0, noise_level=0.1):
    """正常な信号の生成"""
    t = np.linspace(0, length/100, length)
    signal = np.sin(2 * np.pi * frequency * t)
    signal += np.random.normal(0, noise_level, length)
    return signal


def add_anomalies(signal, anomaly_rate=0.05):
    """信号に異常を追加"""
    n_anomalies = int(len(signal) * anomaly_rate)
    anomaly_indices = np.random.choice(len(signal), n_anomalies, replace=False)
    
    signal_with_anomalies = signal.copy()
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['spike', 'dip', 'shift'])
        
        if anomaly_type == 'spike':
            # スパイク異常
            signal_with_anomalies[idx] += np.random.uniform(2, 5) * np.std(signal)
        elif anomaly_type == 'dip':
            # ディップ異常
            signal_with_anomalies[idx] -= np.random.uniform(2, 5) * np.std(signal)
        else:
            # レベルシフト異常
            shift_length = min(50, len(signal) - idx)
            signal_with_anomalies[idx:idx+shift_length] += np.random.uniform(1, 3) * np.std(signal)
    
    return signal_with_anomalies, anomaly_indices


def generate_multivariate_data(n_samples=7000, n_features=3):
    """多変量データの生成"""
    data = {}
    
    # 基本となる時間軸
    time = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')
    
    # 相関のある複数の信号を生成
    base_signal = generate_normal_signal(n_samples, frequency=0.5)
    
    for i in range(n_features):
        if i == 0:
            signal = base_signal + generate_normal_signal(n_samples, frequency=2.0, noise_level=0.05)
        else:
            # 基本信号に相関を持たせる
            correlation = 0.7 - i * 0.2
            signal = correlation * base_signal + (1 - correlation) * generate_normal_signal(
                n_samples, frequency=1.0 + i * 0.5, noise_level=0.1
            )
        
        # 異常を追加
        signal_with_anomalies, _ = add_anomalies(signal, anomaly_rate=0.03)
        data[f'column{i+1}'] = signal_with_anomalies
    
    # DataFrameの作成
    df = pd.DataFrame(data, index=time)
    df.index.name = 'timestamp'
    
    return df


def generate_sample_files(output_dir='input_data', n_files=5):
    """複数のサンプルファイルを生成"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"サンプルデータを生成中... (出力先: {output_dir})")
    
    for i in range(n_files):
        # パラメータをランダムに変更
        n_samples = np.random.randint(6000, 8000)
        n_features = np.random.randint(1, 5)
        
        # データ生成
        df = generate_multivariate_data(n_samples, n_features)
        
        # ファイル名
        filename = f'sample_data_{i+1:03d}.csv'
        filepath = os.path.join(output_dir, filename)
        
        # CSV保存
        df.to_csv(filepath)
        
        print(f"  ✅ {filename} - {n_samples}行 × {n_features}列")
    
    print(f"\n合計 {n_files} ファイルを生成しました。")
    print("\n使用方法:")
    print("1. python main.py で異常検知を実行")
    print("2. streamlit run viewer/run_viewer.py でビューアーを起動")


if __name__ == '__main__':
    # デフォルトで5つのサンプルファイルを生成
    generate_sample_files()