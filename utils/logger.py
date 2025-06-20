"""
ロギングユーティリティ
"""

import os
import logging
from datetime import datetime


def setup_logger(name, config):
    """ロガーのセットアップ"""
    # ログディレクトリの作成
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # ロガーの作成
    logger = logging.getLogger(name)
    logger.setLevel(config['logging']['level'])
    
    # 既存のハンドラーをクリア
    logger.handlers = []
    
    # ファイルハンドラーの設定
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'anomaly_detection_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(config['logging']['level'])
    
    # コンソールハンドラーの設定
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config['logging']['level'])
    
    # フォーマッターの設定
    formatter = logging.Formatter(config['logging']['format'])
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # ハンドラーの追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger