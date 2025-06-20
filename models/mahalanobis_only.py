"""
マハラノビス距離のみによる異常検知モデル
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


class MahalanobisOnly:
    """マハラノビス距離のみによる異常検知"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.threshold_percentile = config['thresholds']['mahalanobis_percentile']
        self.z_threshold = config['thresholds']['z_score']
        
    def detect_anomalies(self, data):
        """異常検知の実行"""
        self.logger.info("マハラノビス距離による異常検知を開始")
        
        # データの準備
        X = self._prepare_data(data)
        
        # マハラノビス距離の計算
        mahalanobis_distances = self._calculate_mahalanobis(X)
        
        # 異常判定
        anomaly_flags = self._detect_anomalies_by_threshold(mahalanobis_distances)
        
        self.logger.info(
            f"異常検知完了: {sum(anomaly_flags)}/{len(anomaly_flags)} "
            f"({sum(anomaly_flags)/len(anomaly_flags)*100:.2f}%)"
        )
        
        return mahalanobis_distances, anomaly_flags
    
    def _prepare_data(self, data):
        """データの準備"""
        if len(data.columns) > 1:
            # 複数カラムの場合はそのまま使用
            X = data.values
        else:
            # 単一カラムの場合は移動窓で特徴量を生成
            values = data.iloc[:, 0].values
            window_size = 20
            
            X = []
            for i in range(len(values) - window_size + 1):
                window = values[i:i + window_size]
                # 各種統計量を特徴量として使用
                features = [
                    np.mean(window),
                    np.std(window),
                    np.median(window),
                    np.percentile(window, 25),
                    np.percentile(window, 75),
                    np.max(window) - np.min(window),  # レンジ
                    self._calculate_slope(window),      # 傾き
                    self._calculate_rms(window)         # RMS
                ]
                X.append(features)
            
            X = np.array(X)
            
            # パディング（元データと同じ長さにする）
            if len(X) < len(values):
                padding_size = len(values) - len(X)
                # 最初の値を繰り返してパディング
                padding = np.repeat(X[0:1], padding_size, axis=0)
                X = np.vstack([padding, X])
        
        return X
    
    def _calculate_slope(self, window):
        """窓内データの傾きを計算"""
        x = np.arange(len(window))
        coeffs = np.polyfit(x, window, 1)
        return coeffs[0]
    
    def _calculate_rms(self, window):
        """RMS（二乗平均平方根）を計算"""
        return np.sqrt(np.mean(window ** 2))
    
    def _calculate_mahalanobis(self, X):
        """マハラノビス距離の計算"""
        # データの標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 平均と共分散行列の計算
        mean = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled.T)
        
        # 共分散行列の次元確認
        if cov.ndim == 0:
            # スカラーの場合（1次元）
            cov = np.array([[cov + 1e-6]])
        elif cov.ndim == 1:
            # 1次元配列の場合
            cov = np.diag(cov + 1e-6)
        else:
            # 正則化（対角成分に小さな値を加える）
            cov += np.eye(cov.shape[0]) * 1e-6
        
        # 逆行列の計算
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.logger.warning("共分散行列が特異です。擬似逆行列を使用します。")
            inv_cov = np.linalg.pinv(cov)
        
        # マハラノビス距離の計算
        distances = []
        for x in X_scaled:
            try:
                # 中心からの距離を計算
                diff = x - mean
                dist = np.sqrt(diff.dot(inv_cov).dot(diff))
                distances.append(dist)
            except Exception as e:
                self.logger.warning(f"マハラノビス距離計算エラー: {e}")
                # エラーの場合はユークリッド距離で代替
                dist = np.linalg.norm(x - mean)
                distances.append(dist)
        
        return np.array(distances)
    
    def _detect_anomalies_by_threshold(self, distances):
        """閾値による異常判定"""
        # 複数の閾値手法を組み合わせる
        
        # 1. パーセンタイルベースの閾値
        percentile_threshold = np.percentile(distances, self.threshold_percentile)
        
        # 2. Zスコアベースの閾値
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        z_score_threshold = mean_dist + self.z_threshold * std_dist
        
        # 3. カイ二乗分布による理論的閾値
        # マハラノビス距離の二乗は自由度pのカイ二乗分布に従う
        # ここでは特徴量の次元数を自由度とする
        df = 8  # 特徴量の数（prepare_dataで定義）
        chi2_threshold = np.sqrt(chi2.ppf(self.threshold_percentile / 100, df=df))
        
        # 最も適切な閾値を選択（中央値を使用）
        thresholds = [percentile_threshold, z_score_threshold, chi2_threshold]
        final_threshold = np.median(thresholds)
        
        # 異常判定
        anomaly_flags = distances > final_threshold
        
        self.logger.info(
            f"閾値: {final_threshold:.3f} "
            f"(percentile={percentile_threshold:.3f}, "
            f"z-score={z_score_threshold:.3f}, "
            f"chi2={chi2_threshold:.3f})"
        )
        
        return anomaly_flags