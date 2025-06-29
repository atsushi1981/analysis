"""
時系列波形データの異常検知システム - メイン処理
マハラノビス距離、SVM+マハラノビス距離、SVR+マハラノビス距離による異常検知
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 機械学習・統計ライブラリ
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import OneClassSVM, SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnomalyDetectionConfig:
    """異常検知システムの設定クラス - パラメータを修正しやすい形で管理"""
    
    # データ処理パラメータ
    DATA_PARAMS = {
        'missing_value_threshold': 0.1,  # 欠損値が10%を超える列は削除
        'outlier_threshold': 3.0,        # 外れ値検出のZ-score閾値
        'normalize_method': 'standard',   # 'standard', 'minmax', 'robust'
    }
    
    # 特徴量抽出パラメータ
    FEATURE_PARAMS = {
        'window_size': 100,               # 移動窓サイズ
        'overlap': 0.5,                   # 窓の重複率
        'fft_n_components': 50,           # FFTで使用する主要成分数
        'autocorr_lags': 20,              # 自己相関で使用するラグ数
    }
    
    # マハラノビス距離パラメータ
    MAHALANOBIS_PARAMS = {
        'threshold_method': 'percentile', # 'percentile', 'chi2', 'mad'
        'threshold_percentile': 95,       # パーセンタイル閾値
        'chi2_confidence': 0.95,          # カイ二乗分布の信頼度
        'regularization': 1e-6,           # 共分散行列の正則化項
    }
    
    # SVM + マハラノビス距離パラメータ
    SVM_MAHALANOBIS_PARAMS = {
        'svm_params': {
            'kernel': 'rbf',              # 'linear', 'poly', 'rbf', 'sigmoid'
            'gamma': 'scale',             # 'scale', 'auto', 数値
            'nu': 0.05,                   # 異常データの割合
        },
        'grid_search': {
            'nu': [0.01, 0.05, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        },
        'mahalanobis_weight': 0.5,        # マハラノビス距離の重み
    }
    
    # SVR + マハラノビス距離パラメータ
    SVR_MAHALANOBIS_PARAMS = {
        'svr_params': {
            'kernel': 'rbf',              # 'linear', 'poly', 'rbf', 'sigmoid'
            'C': 1.0,                     # 正則化パラメータ
            'epsilon': 0.1,               # イプシロン管
            'gamma': 'scale',             # 'scale', 'auto', 数値
        },
        'grid_search': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        },
        'prediction_steps': 5,            # 予測ステップ数
        'mahalanobis_weight': 0.3,        # マハラノビス距離の重み
    }
    
    # 出力パラメータ
    OUTPUT_PARAMS = {
        'save_results': True,
        'results_dir': './results',
        'save_format': ['csv', 'json'],   # 'csv', 'json', 'pickle'
        'plot_results': True,
        'verbose': True,
    }


class FeatureExtractor:
    """時系列データの特徴量抽出クラス"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.feature_names = []
    
    def extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """統計的特徴量を抽出"""
        features = []
        names = []
        
        # 基本統計量
        features.extend([
            np.mean(data), np.std(data), np.var(data),
            np.min(data), np.max(data), np.median(data),
            stats.skew(data), stats.kurtosis(data)
        ])
        names.extend(['mean', 'std', 'var', 'min', 'max', 'median', 'skewness', 'kurtosis'])
        
        # パーセンタイル
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features.append(np.percentile(data, p))
            names.append(f'percentile_{p}')
        
        # 範囲統計
        features.extend([
            np.max(data) - np.min(data),  # 範囲
            np.percentile(data, 75) - np.percentile(data, 25),  # IQR
        ])
        names.extend(['range', 'iqr'])
        
        return np.array(features), names
    
    def extract_time_series_features(self, data: np.ndarray) -> np.ndarray:
        """時系列特徴量を抽出"""
        features = []
        names = []
        
        # 差分統計
        diff1 = np.diff(data)
        diff2 = np.diff(data, n=2)
        
        features.extend([
            np.mean(diff1), np.std(diff1),
            np.mean(diff2), np.std(diff2)
        ])
        names.extend(['diff1_mean', 'diff1_std', 'diff2_mean', 'diff2_std'])
        
        # 自己相関
        lags = self.config.FEATURE_PARAMS['autocorr_lags']
        autocorr_values = []
        for lag in range(1, lags + 1):
            if len(data) > lag:
                autocorr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                autocorr_values.append(autocorr if not np.isnan(autocorr) else 0)
            else:
                autocorr_values.append(0)
        
        features.extend(autocorr_values)
        names.extend([f'autocorr_lag_{i+1}' for i in range(lags)])
        
        # トレンド
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        features.extend([slope, r_value, p_value])
        names.extend(['trend_slope', 'trend_r_value', 'trend_p_value'])
        
        return np.array(features), names
    
    def extract_frequency_features(self, data: np.ndarray) -> np.ndarray:
        """周波数領域特徴量を抽出"""
        features = []
        names = []
        
        # FFT
        fft_values = np.abs(fft(data))
        freqs = fftfreq(len(data))
        
        # 主要周波数成分
        n_components = min(self.config.FEATURE_PARAMS['fft_n_components'], len(fft_values) // 2)
        dominant_freqs = np.argsort(fft_values[:len(fft_values)//2])[-n_components:]
        
        features.extend([
            np.mean(fft_values), np.std(fft_values),
            np.max(fft_values), np.sum(fft_values**2)
        ])
        names.extend(['fft_mean', 'fft_std', 'fft_max', 'spectral_energy'])
        
        # スペクトル重心
        spectral_centroid = np.sum(freqs[:len(freqs)//2] * fft_values[:len(fft_values)//2]) / np.sum(fft_values[:len(fft_values)//2])
        features.append(spectral_centroid)
        names.append('spectral_centroid')
        
        return np.array(features), names
    
    def extract_change_detection_features(self, data: np.ndarray) -> np.ndarray:
        """微細変化検出のための特徴量を抽出"""
        features = []
        names = []
        
        # 勾配統計
        gradients = np.gradient(data)
        features.extend([
            np.mean(np.abs(gradients)), np.std(gradients),
            np.max(np.abs(gradients)), np.sum(gradients**2)
        ])
        names.extend(['gradient_mean_abs', 'gradient_std', 'gradient_max_abs', 'gradient_energy'])
        
        # ピーク検出
        peaks, _ = find_peaks(data)
        valleys, _ = find_peaks(-data)
        
        features.extend([
            len(peaks), len(valleys), len(peaks) + len(valleys)
        ])
        names.extend(['n_peaks', 'n_valleys', 'n_extrema'])
        
        # 移動窓での変動
        window_size = min(100, len(data) // 10)
        if window_size > 1:
            windowed_std = []
            for i in range(len(data) - window_size + 1):
                windowed_std.append(np.std(data[i:i + window_size]))
            
            features.extend([
                np.mean(windowed_std), np.std(windowed_std), np.max(windowed_std)
            ])
        else:
            features.extend([0, 0, 0])
        
        names.extend(['windowed_std_mean', 'windowed_std_std', 'windowed_std_max'])
        
        return np.array(features), names
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """全ての特徴量を抽出"""
        all_features = []
        all_names = []
        
        # 各特徴量カテゴリを抽出
        stat_features, stat_names = self.extract_statistical_features(data)
        ts_features, ts_names = self.extract_time_series_features(data)
        freq_features, freq_names = self.extract_frequency_features(data)
        change_features, change_names = self.extract_change_detection_features(data)
        
        # 結合
        all_features = np.concatenate([stat_features, ts_features, freq_features, change_features])
        all_names = stat_names + ts_names + freq_names + change_names
        
        self.feature_names = all_names
        return all_features


class MahalanobisDetector:
    """マハラノビス距離による異常検知"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.mean = None
        self.cov_inv = None
        self.threshold = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray):
        """学習"""
        X_scaled = self.scaler.fit_transform(X)
        
        # 平均と共分散行列の計算
        self.mean = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled, rowvar=False)
        
        # 正則化
        reg = self.config.MAHALANOBIS_PARAMS['regularization']
        cov += reg * np.eye(cov.shape[0])
        
        # 逆行列計算
        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            logger.warning("共分散行列が特異行列です。擬似逆行列を使用します。")
            self.cov_inv = np.linalg.pinv(cov)
        
        # 閾値設定
        distances = self._calculate_distances(X_scaled)
        self._set_threshold(distances, X_scaled.shape[1])
    
    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        """マハラノビス距離を計算"""
        diff = X - self.mean
        distances = np.sqrt(np.sum((diff @ self.cov_inv) * diff, axis=1))
        return distances
    
    def _set_threshold(self, distances: np.ndarray, n_features: int):
        """閾値を設定"""
        method = self.config.MAHALANOBIS_PARAMS['threshold_method']
        
        if method == 'percentile':
            percentile = self.config.MAHALANOBIS_PARAMS['threshold_percentile']
            self.threshold = np.percentile(distances, percentile)
        elif method == 'chi2':
            confidence = self.config.MAHALANOBIS_PARAMS['chi2_confidence']
            self.threshold = np.sqrt(stats.chi2.ppf(confidence, n_features))
        elif method == 'mad':
            median = np.median(distances)
            mad = np.median(np.abs(distances - median))
            self.threshold = median + 3 * mad
        else:
            self.threshold = np.percentile(distances, 95)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """予測"""
        X_scaled = self.scaler.transform(X)
        distances = self._calculate_distances(X_scaled)
        anomalies = distances > self.threshold
        return anomalies, distances


class SVMMahalanobisDetector:
    """SVM + マハラノビス距離による異常検知"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.svm = None
        self.mahalanobis = MahalanobisDetector(config)
        self.best_params = None
    
    def fit(self, X: np.ndarray):
        """学習"""
        # グリッドサーチでSVMパラメータを最適化
        svm_params = self.config.SVM_MAHALANOBIS_PARAMS['svm_params']
        grid_params = self.config.SVM_MAHALANOBIS_PARAMS['grid_search']
        
        if len(grid_params) > 0:
            self.svm = OneClassSVM(kernel=svm_params['kernel'])
            grid_search = GridSearchCV(
                self.svm, grid_params, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=0
            )
            
            # ダミーラベル（すべて正常として扱う）
            y_dummy = np.ones(X.shape[0])
            grid_search.fit(X, y_dummy)
            
            self.best_params = grid_search.best_params_
            self.svm = OneClassSVM(**{**svm_params, **self.best_params})
        else:
            self.svm = OneClassSVM(**svm_params)
        
        # SVMの学習
        self.svm.fit(X)
        
        # マハラノビス距離の学習
        self.mahalanobis.fit(X)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """予測"""
        # SVM予測
        svm_pred = self.svm.predict(X)
        svm_scores = self.svm.decision_function(X)
        
        # マハラノビス距離
        _, mahalanobis_distances = self.mahalanobis.predict(X)
        
        # 統合スコア
        weight = self.config.SVM_MAHALANOBIS_PARAMS['mahalanobis_weight']
        combined_scores = (1 - weight) * (-svm_scores) + weight * mahalanobis_distances
        
        # 異常判定（SVMの判定を基準に）
        anomalies = svm_pred == -1
        
        return anomalies, combined_scores


class SVRMahalanobisDetector:
    """SVR + マハラノビス距離による異常検知"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.svr = None
        self.mahalanobis = MahalanobisDetector(config)
        self.best_params = None
        self.threshold = None
    
    def _create_sequences(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """時系列予測用のシーケンスを作成"""
        steps = self.config.SVR_MAHALANOBIS_PARAMS['prediction_steps']
        X_seq, y_seq = [], []
        
        for i in range(len(X) - steps):
            X_seq.append(X[i:i + steps].flatten())
            y_seq.append(X[i + steps])
        
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X: np.ndarray):
        """学習"""
        # シーケンス作成
        X_seq, y_seq = self._create_sequences(X)
        
        if len(X_seq) == 0:
            logger.error("シーケンス作成に失敗しました。データが不足している可能性があります。")
            return
        
        # グリッドサーチでSVRパラメータを最適化
        svr_params = self.config.SVR_MAHALANOBIS_PARAMS['svr_params']
        grid_params = self.config.SVR_MAHALANOBIS_PARAMS['grid_search']
        
        if len(grid_params) > 0:
            self.svr = SVR(kernel=svr_params['kernel'])
            grid_search = GridSearchCV(
                self.svr, grid_params, cv=3, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=0
            )
            
            # 最初の特徴量のみで学習（簡単化）
            grid_search.fit(X_seq, y_seq[:, 0])
            
            self.best_params = grid_search.best_params_
            self.svr = SVR(**{**svr_params, **self.best_params})
        else:
            self.svr = SVR(**svr_params)
        
        # SVRの学習
        self.svr.fit(X_seq, y_seq[:, 0])
        
        # 予測誤差を計算
        y_pred = self.svr.predict(X_seq)
        prediction_errors = np.abs(y_seq[:, 0] - y_pred)
        
        # マハラノビス距離の学習
        self.mahalanobis.fit(X)
        
        # 閾値設定
        self.threshold = np.percentile(prediction_errors, 95)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """予測"""
        X_seq, y_seq = self._create_sequences(X)
        
        if len(X_seq) == 0:
            # データが不足している場合はマハラノビス距離のみ使用
            return self.mahalanobis.predict(X)
        
        # SVR予測
        y_pred = self.svr.predict(X_seq)
        prediction_errors = np.abs(y_seq[:, 0] - y_pred)
        
        # マハラノビス距離
        _, mahalanobis_distances = self.mahalanobis.predict(X)
        
        # 予測誤差を元の長さに合わせる
        steps = self.config.SVR_MAHALANOBIS_PARAMS['prediction_steps']
        full_prediction_errors = np.zeros(len(X))
        full_prediction_errors[steps:] = prediction_errors
        
        # 統合スコア
        weight = self.config.SVR_MAHALANOBIS_PARAMS['mahalanobis_weight']
        combined_scores = (1 - weight) * full_prediction_errors + weight * mahalanobis_distances
        
        # 異常判定
        anomalies = full_prediction_errors > self.threshold
        
        return anomalies, combined_scores


class AnomalyDetectionSystem:
    """異常検知システムのメインクラス"""
    
    def __init__(self, config: AnomalyDetectionConfig = None):
        self.config = config or AnomalyDetectionConfig()
        self.feature_extractor = FeatureExtractor(self.config)
        self.detectors = {
            'mahalanobis': MahalanobisDetector(self.config),
            'svm_mahalanobis': SVMMahalanobisDetector(self.config),
            'svr_mahalanobis': SVRMahalanobisDetector(self.config)
        }
        self.results = {}
        
        # 結果保存ディレクトリ作成
        if self.config.OUTPUT_PARAMS['save_results']:
            Path(self.config.OUTPUT_PARAMS['results_dir']).mkdir(parents=True, exist_ok=True)
    
    def load_data(self, data_path: str, target_columns: List[str] = None) -> pd.DataFrame:
        """データ読み込み"""
        if os.path.isdir(data_path):
            # ディレクトリの場合、CSVファイルを全て読み込み
            csv_files = glob.glob(os.path.join(data_path, "*.csv"))
            dfs = []
            
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    df['file_name'] = os.path.basename(file)
                    dfs.append(df)
                    logger.info(f"ファイル読み込み完了: {file}")
                except Exception as e:
                    logger.error(f"ファイル読み込みエラー: {file}, {e}")
            
            if not dfs:
                raise ValueError("読み込み可能なCSVファイルが見つかりません。")
            
            data = pd.concat(dfs, ignore_index=True)
        else:
            # 単一ファイルの場合
            data = pd.read_csv(data_path)
        
        # 対象カラムの選択
        if target_columns:
            available_columns = [col for col in target_columns if col in data.columns]
            if not available_columns:
                raise ValueError(f"指定されたカラムが見つかりません: {target_columns}")
            data = data[available_columns + (['file_name'] if 'file_name' in data.columns else [])]
        
        logger.info(f"データ読み込み完了: {data.shape}")
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """データ前処理"""
        # 数値列のみ選択
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        data_processed = data[numeric_columns].copy()
        
        # 欠損値処理
        missing_threshold = self.config.DATA_PARAMS['missing_value_threshold']
        missing_ratio = data_processed.isnull().sum() / len(data_processed)
        columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
        
        if len(columns_to_drop) > 0:
            logger.info(f"欠損値が多い列を削除: {columns_to_drop.tolist()}")
            data_processed = data_processed.drop(columns=columns_to_drop)
        
        # 残りの欠損値を前方補完
        data_processed = data_processed.fillna(method='ffill').fillna(method='bfill')
        
        # 外れ値処理（Z-score）
        z_threshold = self.config.DATA_PARAMS['outlier_threshold']
        for col in data_processed.columns:
            z_scores = np.abs(stats.zscore(data_processed[col]))
            outliers = z_scores > z_threshold
            if outliers.sum() > 0:
                # 外れ値をクリッピング
                percentile_99 = np.percentile(data_processed[col], 99)
                percentile_1 = np.percentile(data_processed[col], 1)
                data_processed.loc[data_processed[col] > percentile_99, col] = percentile_99
                data_processed.loc[data_processed[col] < percentile_1, col] = percentile_1
        
        logger.info(f"前処理完了: {data_processed.shape}")
        return data_processed
    
    def extract_features_from_data(self, data: pd.DataFrame) -> np.ndarray:
        """データから特徴量を抽出"""
        features_list = []
        
        for col in data.columns:
            if data[col].dtype in [np.float64, np.int64]:
                features = self.feature_extractor.extract_features(data[col].values)
                features_list.append(features)
        
        if not features_list:
            raise ValueError("特徴量抽出に失敗しました。")
        
        # 列方向に結合
        features_matrix = np.column_stack(features_list)
        
        # NaNや無限大値の処理
        features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=1e10, neginf=-1e10)
        
        logger.info(f"特徴量抽出完了: {features_matrix.shape}")
        return features_matrix
    
    def train_models(self, features: np.ndarray):
        """モデル学習"""
        for name, detector in self.detectors.items():
            try:
                logger.info(f"{name} モデルの学習開始")
                detector.fit(features)
                logger.info(f"{name} モデルの学習完了")
            except Exception as e:
                logger.error(f"{name} モデルの学習エラー: {e}")
    
    def detect_anomalies(self, features: np.ndarray) -> Dict:
        """異常検知実行"""
        results = {}
        
        for name, detector in self.detectors.items():
            try:
                logger.info(f"{name} による異常検知開始")
                anomalies, scores = detector.predict(features)
                results[name] = {
                    'anomalies': anomalies,
                    'scores': scores,
                    'n_anomalies': np.sum(anomalies),
                    'anomaly_rate': np.mean(anomalies)
                }
                logger.info(f"{name} 異常検知完了: {np.sum(anomalies)}個の異常を検出")
            except Exception as e:
                logger.error(f"{name} 異常検知エラー: {e}")
                results[name] = {
                    'anomalies': np.zeros(len(features), dtype=bool),
                    'scores': np.zeros(len(features)),
                    'n_anomalies': 0,
                    'anomaly_rate': 0.0
                }
        
        return results
    
    def save_results(self, results: Dict, original_data: pd.DataFrame, features: np.ndarray):
        """結果保存"""
        if not self.config.OUTPUT_PARAMS['save_results']:
            return
        
        results_dir = Path(self.config.OUTPUT_PARAMS['results_dir'])
        
        # CSV形式で保存
        if 'csv' in self.config.OUTPUT_PARAMS['save_format']:
            results_df = original_data.copy()
            
            for method, result in results.items():
                results_df[f'{method}_anomaly'] = result['anomalies']
                results_df[f'{method}_score'] = result['scores']
            
            csv_path = results_dir / 'anomaly_detection_results.csv'
            results_df.to_csv(csv_path, index=False)
            logger.info(f"結果をCSVで保存: {csv_path}")
        
        # JSON形式で保存
        if 'json' in self.config.OUTPUT_PARAMS['save_format']:
            json_results = {
                'summary': {method: {
                    'n_anomalies': int(result['n_anomalies']),
                    'anomaly_rate': float(result['anomaly_rate'])
                } for method, result in results.items()},
                'feature_names': self.feature_extractor.feature_names,
                'config': {
                    'data_params': self.config.DATA_PARAMS,
                    'feature_params': self.config.FEATURE_PARAMS,
                    'mahalanobis_params': self.config.MAHALANOBIS_PARAMS,
                }
            }
            
            json_path = results_dir / 'anomaly_detection_results.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            logger.info(f"結果をJSONで保存: {json_path}")
    
    def generate_report(self, results: Dict, original_data: pd.DataFrame) -> str:
        """分析レポート生成"""
        report = []
        report.append("=" * 60)
        report.append("異常検知分析レポート")
        report.append("=" * 60)
        report.append(f"データサイズ: {original_data.shape}")
        report.append(f"分析日時: {pd.Timestamp.now()}")
        report.append("")
        
        report.append("【検知結果サマリー】")
        for method, result in results.items():
            report.append(f"{method}:")
            report.append(f"  - 異常数: {result['n_anomalies']}")
            report.append(f"  - 異常率: {result['anomaly_rate']:.4f}")
            report.append(f"  - 平均スコア: {np.mean(result['scores']):.4f}")
            report.append(f"  - 最大スコア: {np.max(result['scores']):.4f}")
            report.append("")
        
        report.append("【パラメータ設定】")
        report.append(f"データ処理: {self.config.DATA_PARAMS}")
        report.append(f"特徴量抽出: {self.config.FEATURE_PARAMS}")
        report.append(f"マハラノビス距離: {self.config.MAHALANOBIS_PARAMS}")
        report.append("")
        
        return "\n".join(report)
    
    def plot_results(self, results: Dict, original_data: pd.DataFrame):
        """結果可視化"""
        if not self.config.OUTPUT_PARAMS['plot_results']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 異常スコア分布
        ax1 = axes[0, 0]
        for method, result in results.items():
            ax1.hist(result['scores'], alpha=0.7, label=method, bins=50)
        ax1.set_xlabel('異常スコア')
        ax1.set_ylabel('頻度')
        ax1.set_title('異常スコア分布')
        ax1.legend()
        
        # 異常率比較
        ax2 = axes[0, 1]
        methods = list(results.keys())
        anomaly_rates = [results[method]['anomaly_rate'] for method in methods]
        ax2.bar(methods, anomaly_rates)
        ax2.set_ylabel('異常率')
        ax2.set_title('手法別異常率')
        ax2.tick_params(axis='x', rotation=45)
        
        # 時系列プロット（最初の数値列）
        ax3 = axes[1, 0]
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            first_col = numeric_cols[0]
            ax3.plot(original_data[first_col].values, alpha=0.7, label='Original')
            
            # 異常点をハイライト
            for method, result in results.items():
                anomaly_indices = np.where(result['anomalies'])[0]
                if len(anomaly_indices) > 0:
                    ax3.scatter(anomaly_indices, 
                              original_data[first_col].iloc[anomaly_indices],
                              alpha=0.8, s=20, label=f'{method} anomalies')
            
            ax3.set_xlabel('時間インデックス')
            ax3.set_ylabel(first_col)
            ax3.set_title('時系列データと異常点')
            ax3.legend()
        
        # 相関ヒートマップ
        ax4 = axes[1, 1]
        score_df = pd.DataFrame({method: results[method]['scores'] for method in results.keys()})
        correlation_matrix = score_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4)
        ax4.set_title('手法間スコア相関')
        
        plt.tight_layout()
        
        if self.config.OUTPUT_PARAMS['save_results']:
            plot_path = Path(self.config.OUTPUT_PARAMS['results_dir']) / 'analysis_plots.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"プロットを保存: {plot_path}")
        
        plt.show()
    
    def run_analysis(self, data_path: str, target_columns: List[str] = None) -> Dict:
        """完全な分析パイプラインを実行"""
        try:
            # データ読み込み
            logger.info("分析開始")
            original_data = self.load_data(data_path, target_columns)
            
            # 前処理
            processed_data = self.preprocess_data(original_data)
            
            # 特徴量抽出
            features = self.extract_features_from_data(processed_data)
            
            # モデル学習
            self.train_models(features)
            
            # 異常検知
            results = self.detect_anomalies(features)
            
            # 結果保存
            self.save_results(results, original_data, features)
            
            # レポート生成
            report = self.generate_report(results, original_data)
            if self.config.OUTPUT_PARAMS['verbose']:
                print(report)
            
            # 可視化
            self.plot_results(results, original_data)
            
            # 結果を保存
            self.results = results
            
            logger.info("分析完了")
            return results
            
        except Exception as e:
            logger.error(f"分析エラー: {e}")
            raise


def main():
    """メイン実行関数"""
    # 設定
    config = AnomalyDetectionConfig()
    
    # システム初期化
    system = AnomalyDetectionSystem(config)
    
    # 分析実行例
    try:
        # データパスを指定（ディレクトリまたは単一ファイル）
        data_path = "./data"  # 実際のパスに変更してください
        
        # 対象カラムを指定（Noneの場合は全数値列を使用）
        target_columns = None  # 例: ['sensor_1', 'sensor_2', 'temperature']
        
        # 分析実行
        results = system.run_analysis(data_path, target_columns)
        
        # 結果の詳細表示
        print("\n=== 詳細結果 ===")
        for method, result in results.items():
            print(f"\n{method.upper()}:")
            print(f"  異常検知数: {result['n_anomalies']}")
            print(f"  異常率: {result['anomaly_rate']:.4f}")
            print(f"  スコア統計:")
            print(f"    平均: {np.mean(result['scores']):.4f}")
            print(f"    標準偏差: {np.std(result['scores']):.4f}")
            print(f"    最小値: {np.min(result['scores']):.4f}")
            print(f"    最大値: {np.max(result['scores']):.4f}")
            
            # 異常インデックスの例
            anomaly_indices = np.where(result['anomalies'])[0]
            if len(anomaly_indices) > 0:
                print(f"    異常インデックス例（最初の10個）: {anomaly_indices[:10].tolist()}")
    
    except Exception as e:
        logger.error(f"メイン実行エラー: {e}")
        print("実行前に以下を確認してください：")
        print("1. データディレクトリ/ファイルのパスが正しいか")
        print("2. CSVファイルが存在するか")
        print("3. CSVファイルに数値データが含まれているか")


# 実行例とテストケース
def run_test_example():
    """テスト用の実行例"""
    import tempfile
    import shutil
    
    # テンポラリディレクトリ作成
    temp_dir = tempfile.mkdtemp()
    
    try:
        # テストデータ生成
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        # 正常データ
        normal_data = np.random.normal(0, 1, (n_samples, n_features))
        
        # 異常データを挿入
        anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
        normal_data[anomaly_indices] += np.random.normal(5, 2, (50, n_features))
        
        # DataFrameに変換
        test_df = pd.DataFrame(normal_data, columns=[f'feature_{i}' for i in range(n_features)])
        test_df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        
        # CSVファイルとして保存
        test_csv_path = os.path.join(temp_dir, 'test_data.csv')
        test_df.to_csv(test_csv_path, index=False)
        
        # 設定
        config = AnomalyDetectionConfig()
        config.OUTPUT_PARAMS['results_dir'] = temp_dir
        
        # システム実行
        system = AnomalyDetectionSystem(config)
        results = system.run_analysis(test_csv_path, target_columns=['feature_0', 'feature_1'])
        
        print("テスト実行完了！")
        print(f"結果は {temp_dir} に保存されました")
        
        return results
        
    except Exception as e:
        print(f"テスト実行エラー: {e}")
        return None
    
    finally:
        # クリーンアップ
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("時系列波形データ異常検知システム")
    print("1. 通常実行: main()")
    print("2. テスト実行: run_test_example()")
    print()
    
    # テスト実行
    run_test_example()
    
    # 通常実行（コメントアウト解除して使用）
    # main()