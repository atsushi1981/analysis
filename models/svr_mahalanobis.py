"""
SVR + マハラノビス距離による異常検知モデル
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


class SVRMahalanobis:
    """SVR + マハラノビス距離による異常検知"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.svr_params = config['model_params']['svr']
        self.threshold_percentile = config['thresholds']['mahalanobis_percentile']
        self.z_threshold = config['thresholds']['z_score']
        
    def detect_anomalies(self, data):
        """異常検知の実行"""
        self.logger.info("SVR + マハラノビス距離による異常検知を開始")
        
        # データの準備
        X, y = self._prepare_data(data)
        
        # SVRによる予測
        predictions, residuals = self._svr_predict(X, y)
        
        # マハラノビス距離の計算
        mahalanobis_distances = self._calculate_mahalanobis(residuals)
        
        # 異常判定
        anomaly_flags = self._detect_anomalies_by_threshold(mahalanobis_distances)
        
        self.logger.info(
            f"異常検知完了: {sum(anomaly_flags)}/{len(anomaly_flags)} "
            f"({sum(anomaly_flags)/len(anomaly_flags)*100:.2f}%)"
        )
        
        return mahalanobis_distances, anomaly_flags
    
    def _prepare_data(self, data):
        """データの準備（時系列データから特徴量とターゲットを作成）"""
        # 複数カラムの場合は最初のカラムをターゲットとする
        if len(data.columns) > 1:
            # 他のカラムを特徴量として使用
            X = data.iloc[:, 1:].values
            y = data.iloc[:, 0].values
        else:
            # 単一カラムの場合はラグ特徴量を作成
            values = data.iloc[:, 0].values
            lag = 5  # ラグ数
            
            X = []
            y = []
            
            for i in range(lag, len(values)):
                X.append(values[i-lag:i])
                y.append(values[i])
            
            X = np.array(X)
            y = np.array(y)
        
        return X, y
    
    def _svr_predict(self, X, y):
        """SVRによる予測と残差計算"""
        # データの標準化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # SVRモデルの構築と学習
        svr = SVR(
            kernel=self.svr_params['kernel'],
            C=self.svr_params['C'],
            epsilon=self.svr_params['epsilon'],
            gamma=self.svr_params['gamma']
        )
        
        svr.fit(X_scaled, y_scaled)
        
        # 予測
        predictions_scaled = svr.predict(X_scaled)
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        
        # 残差の計算
        residuals = y - predictions
        
        return predictions, residuals
    
    def _calculate_mahalanobis(self, residuals):
        """マハラノビス距離の計算"""
        # 残差を2次元配列に変換
        residuals_2d = residuals.reshape(-1, 1)
        
        # 平均と共分散行列の計算
        mean = np.mean(residuals_2d, axis=0)
        cov = np.cov(residuals_2d.T)
        
        # 共分散行列が特異な場合の対処
        if cov.ndim == 0 or cov == 0:
            cov = np.array([[np.var(residuals) + 1e-6]])
        elif cov.ndim == 1:
            cov = cov.reshape(1, 1)
        
        # 逆行列の計算
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # 特異行列の場合は擬似逆行列を使用
            inv_cov = np.linalg.pinv(cov)
        
        # マハラノビス距離の計算
        distances = []
        for residual in residuals_2d:
            try:
                dist = mahalanobis(residual, mean, inv_cov)
                distances.append(dist)
            except:
                # エラーの場合はユークリッド距離で代替
                dist = np.linalg.norm(residual - mean)
                distances.append(dist)
        
        return np.array(distances)
    
    def _detect_anomalies_by_threshold(self, distances):
        """閾値による異常判定"""
        # パーセンタイルベースの閾値
        threshold = np.percentile(distances, self.threshold_percentile)
        
        # カイ二乗分布による理論的閾値も考慮
        # 自由度1のカイ二乗分布の95%点
        chi2_threshold = chi2.ppf(self.threshold_percentile / 100, df=1)
        
        # より厳しい閾値を採用
        final_threshold = max(threshold, chi2_threshold)
        
        # 異常判定
        anomaly_flags = distances > final_threshold
        
        self.logger.info(
            f"閾値: {final_threshold:.3f} "
            f"(percentile={threshold:.3f}, chi2={chi2_threshold:.3f})"
        )
        
        return anomaly_flags