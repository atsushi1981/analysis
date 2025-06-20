"""
One-Class SVM + マハラノビス距離による異常検知モデル
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


class SVMMahalanobis:
    """One-Class SVM + マハラノビス距離による異常検知"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.svm_params = config['model_params']['svm']
        self.threshold_percentile = config['thresholds']['mahalanobis_percentile']
        self.z_threshold = config['thresholds']['z_score']
        
    def detect_anomalies(self, data):
        """異常検知の実行"""
        self.logger.info("One-Class SVM + マハラノビス距離による異常検知を開始")
        
        # データの準備
        X = self._prepare_data(data)
        
        # One-Class SVMによる異常スコア計算
        svm_scores = self._svm_detect(X)
        
        # マハラノビス距離の計算
        mahalanobis_distances = self._calculate_mahalanobis(X)
        
        # 統合スコアの計算
        combined_scores = self._combine_scores(svm_scores, mahalanobis_distances)
        
        # 異常判定
        anomaly_flags = self._detect_anomalies_by_threshold(combined_scores)
        
        self.logger.info(
            f"異常検知完了: {sum(anomaly_flags)}/{len(anomaly_flags)} "
            f"({sum(anomaly_flags)/len(anomaly_flags)*100:.2f}%)"
        )
        
        return combined_scores, anomaly_flags
    
    def _prepare_data(self, data):
        """データの準備（特徴量行列の作成）"""
        if len(data.columns) > 1:
            # 複数カラムの場合はそのまま使用
            X = data.values
        else:
            # 単一カラムの場合は滑走窓で特徴量を作成
            values = data.iloc[:, 0].values
            window_size = 10
            stride = 1
            
            X = []
            for i in range(0, len(values) - window_size + 1, stride):
                window = values[i:i + window_size]
                # 統計的特徴量の抽出
                features = [
                    np.mean(window),
                    np.std(window),
                    np.min(window),
                    np.max(window),
                    np.percentile(window, 25),
                    np.percentile(window, 75)
                ]
                X.append(features)
            
            X = np.array(X)
        
        return X
    
    def _svm_detect(self, X):
        """One-Class SVMによる異常検知"""
        # データの標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # One-Class SVMモデルの構築と学習
        svm = OneClassSVM(
            kernel=self.svm_params['kernel'],
            gamma=self.svm_params['gamma'],
            nu=self.svm_params['nu']
        )
        
        svm.fit(X_scaled)
        
        # 決定関数値（異常スコア）の計算
        # 負の値が異常を示すので、符号を反転して正の異常スコアに変換
        scores = -svm.decision_function(X_scaled)
        
        # スコアの正規化（0-1の範囲に）
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        return scores_normalized
    
    def _calculate_mahalanobis(self, X):
        """マハラノビス距離の計算"""
        # 平均と共分散行列の計算
        mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        
        # 共分散行列の正則化（特異性を避ける）
        cov += np.eye(cov.shape[0]) * 1e-6
        
        # 逆行列の計算
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # 特異行列の場合は擬似逆行列を使用
            inv_cov = np.linalg.pinv(cov)
        
        # マハラノビス距離の計算
        distances = []
        for x in X:
            try:
                dist = mahalanobis(x, mean, inv_cov)
                distances.append(dist)
            except:
                # エラーの場合はユークリッド距離で代替
                dist = np.linalg.norm(x - mean)
                distances.append(dist)
        
        distances = np.array(distances)
        
        # 距離の正規化（0-1の範囲に）
        distances_normalized = (distances - distances.min()) / (distances.max() - distances.min() + 1e-6)
        
        return distances_normalized
    
    def _combine_scores(self, svm_scores, mahalanobis_distances):
        """SVMスコアとマハラノビス距離の統合"""
        # 重み付き平均（設定可能にしても良い）
        svm_weight = 0.5
        mahalanobis_weight = 0.5
        
        combined = svm_weight * svm_scores + mahalanobis_weight * mahalanobis_distances
        
        self.logger.info(
            f"スコア統合: SVM重み={svm_weight}, "
            f"マハラノビス重み={mahalanobis_weight}"
        )
        
        return combined
    
    def _detect_anomalies_by_threshold(self, scores):
        """閾値による異常判定"""
        # パーセンタイルベースの閾値
        threshold = np.percentile(scores, self.threshold_percentile)
        
        # 異常判定
        anomaly_flags = scores > threshold
        
        self.logger.info(f"閾値: {threshold:.3f}")
        
        return anomaly_flags
        