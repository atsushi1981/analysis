#!/usr/bin/env python3
"""
異常検知メインスクリプト
設定に基づいて選択された手法で全CSVファイルを処理
"""

import os
import sys
import yaml
import glob
import pandas as pd
from datetime import datetime

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from utils.data_loader import DataLoader
from utils.plotter import Plotter
from models.svr_mahalanobis import SVRMahalanobis
from models.svm_mahalanobis import SVMMahalanobis
from models.mahalanobis_only import MahalanobisOnly


class AnomalyDetectionEngine:
    """異常検知エンジンクラス"""
    
    def __init__(self, config_path='config.yaml'):
        """初期化"""
        self.config = self._load_config(config_path)
        self.logger = setup_logger('AnomalyDetectionEngine', self.config)
        self._create_directories()
        self.data_loader = DataLoader(self.config, self.logger)
        self.plotter = Plotter(self.config, self.logger)
        self.model = self._initialize_model()
        
    def _load_config(self, config_path):
        """設定ファイルの読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_directories(self):
        """必要なディレクトリの作成"""
        dirs = [
            self.config['paths']['results_dir'],
            self.config['paths']['csv_output_dir'],
            self.config['paths']['graph_output_dir'],
            self.config['paths']['log_dir']
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _initialize_model(self):
        """選択されたモデルの初期化"""
        mode = self.config['mode']
        
        if mode == 'svr_mahalanobis':
            self.logger.info("SVR + マハラノビス距離モードで初期化")
            return SVRMahalanobis(self.config, self.logger)
        elif mode == 'svm_mahalanobis':
            self.logger.info("SVM + マハラノビス距離モードで初期化")
            return SVMMahalanobis(self.config, self.logger)
        elif mode == 'mahalanobis_only':
            self.logger.info("マハラノビス距離のみモードで初期化")
            return MahalanobisOnly(self.config, self.logger)
        else:
            raise ValueError(f"不明なモード: {mode}")
    
    def process_all_files(self):
        """全CSVファイルの処理"""
        input_pattern = os.path.join(self.config['paths']['input_dir'], '*.csv')
        csv_files = glob.glob(input_pattern)
        
        if not csv_files:
            self.logger.warning(f"CSVファイルが見つかりません: {input_pattern}")
            return
        
        self.logger.info(f"{len(csv_files)}個のファイルを処理開始")
        
        results_summary = []
        
        for i, csv_file in enumerate(csv_files, 1):
            self.logger.info(f"処理中 ({i}/{len(csv_files)}): {os.path.basename(csv_file)}")
            
            try:
                # データ読み込み
                data = self.data_loader.load_csv(csv_file)
                
                if data is None:
                    continue
                
                # 対象カラムの抽出
                target_data = self._extract_target_columns(data, csv_file)
                
                if target_data is None:
                    continue
                
                # 異常検知実行
                anomaly_scores, anomaly_flags = self.model.detect_anomalies(target_data)
                
                # 結果の保存
                result_data = self._save_results(
                    data, target_data, anomaly_scores, 
                    anomaly_flags, csv_file
                )
                
                # サマリー情報の記録
                results_summary.append({
                    'file': os.path.basename(csv_file),
                    'total_points': len(data),
                    'anomaly_points': sum(anomaly_flags),
                    'anomaly_rate': sum(anomaly_flags) / len(data) * 100
                })
                
            except Exception as e:
                self.logger.error(f"ファイル処理エラー {csv_file}: {str(e)}")
                continue
        
        # 処理結果サマリーの保存
        self._save_summary(results_summary)
        self.logger.info("全ファイルの処理が完了しました")
    
    def _extract_target_columns(self, data, csv_file):
        """対象カラムの抽出"""
        target_columns = self.config['target_columns']
        available_columns = data.columns.tolist()
        
        # 存在するカラムのみ抽出
        valid_columns = [col for col in target_columns if col in available_columns]
        
        if not valid_columns:
            self.logger.warning(
                f"対象カラムが見つかりません {csv_file}: "
                f"指定={target_columns}, 利用可能={available_columns}"
            )
            return None
        
        if len(valid_columns) < len(target_columns):
            missing = set(target_columns) - set(valid_columns)
            self.logger.warning(f"一部カラムが欠落: {missing}")
        
        return data[valid_columns]
    
    def _save_results(self, original_data, target_data, anomaly_scores, 
                      anomaly_flags, csv_file):
        """結果の保存"""
        basename = os.path.splitext(os.path.basename(csv_file))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 結果データの作成
        result_data = original_data.copy()
        result_data['anomaly_score'] = anomaly_scores
        result_data['is_anomaly'] = anomaly_flags
        
        # CSV保存
        csv_output = os.path.join(
            self.config['paths']['csv_output_dir'],
            f"{basename}_result_{timestamp}.csv"
        )
        result_data.to_csv(csv_output, index=False)
        self.logger.info(f"CSV保存: {csv_output}")
        
        # グラフ保存
        for col in target_data.columns:
            graph_output = os.path.join(
                self.config['paths']['graph_output_dir'],
                f"{basename}_{col}_{timestamp}.png"
            )
            
            self.plotter.plot_anomaly_results(
                target_data[col].values,
                anomaly_scores,
                anomaly_flags,
                title=f"{basename} - {col}",
                save_path=graph_output
            )
        
        return result_data
    
    def _save_summary(self, results_summary):
        """処理結果サマリーの保存"""
        if not results_summary:
            return
        
        summary_df = pd.DataFrame(results_summary)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join(
            self.config['paths']['results_dir'],
            f"summary_{timestamp}.csv"
        )
        
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"サマリー保存: {summary_path}")
        
        # 統計情報のログ出力
        total_files = len(results_summary)
        total_points = sum(r['total_points'] for r in results_summary)
        total_anomalies = sum(r['anomaly_points'] for r in results_summary)
        
        self.logger.info(
            f"処理完了: {total_files}ファイル, "
            f"{total_points}データポイント, "
            f"{total_anomalies}異常検知 "
            f"(平均異常率: {total_anomalies/total_points*100:.2f}%)"
        )


def main():
    """メイン処理"""
    engine = AnomalyDetectionEngine()
    engine.process_all_files()


if __name__ == '__main__':
    main()