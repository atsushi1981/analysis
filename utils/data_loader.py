"""
データ読み込みユーティリティ
"""

import pandas as pd
import numpy as np


class DataLoader:
    """データローダークラス"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def load_csv(self, filepath):
        """CSVファイルの読み込み"""
        try:
            # CSVの読み込み
            data = pd.read_csv(filepath)
            
            self.logger.info(
                f"CSV読み込み成功: {filepath} "
                f"(形状: {data.shape})"
            )
            
            # データ検証
            if data.empty:
                self.logger.warning(f"空のデータファイル: {filepath}")
                return None
            
            # 数値データの確認
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                self.logger.warning(f"数値カラムが存在しません: {filepath}")
                return None
            
            # 欠損値の処理
            if data.isnull().any().any():
                self.logger.warning(f"欠損値を検出: {filepath}")
                # 欠損値を前方補完
                data = data.fillna(method='ffill').fillna(method='bfill')
            
            return data
            
        except Exception as e:
            self.logger.error(f"CSV読み込みエラー {filepath}: {str(e)}")
            return None
    
    def validate_columns(self, data, required_columns):
        """必要なカラムの存在確認"""
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            self.logger.warning(
                f"必要なカラムが不足: {missing_columns}"
            )
            return False
        
        return True