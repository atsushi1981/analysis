# 時系列波形データ異常検知システム

マハラノビス距離、SVM+マハラノビス距離、SVR+マハラノビス距離を用いた時系列波形データの異常検知システムです。

## 📋 システム概要

- **目的**: 時系列波形データの微細な変化と異常を検知
- **データ**: 1ショット=1ファイル（約7,500行）のCSV形式時系列データ群
- **手法**: 3つの異常検知アルゴリズム
  - マハラノビス距離
  - SVM + マハラノビス距離  
  - SVR + マハラノビス距離
- **構成**: 
  - `main.py`: 分析処理
  - `visualization.py`: Streamlit動的可視化
  - `create_csv.py`: テストデータ生成

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリクローン
git clone <your-repository>
cd anomaly-detection-system

# 仮想環境作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存ライブラリインストール
pip install -r requirements.txt
```

### 2. テストデータ生成

```bash
# 基本実行（15ファイル、各7500行、8カラム、異常率15%）
python create_csv.py

# カスタム設定での実行
python create_csv.py --num_files 20 --rows 5000 --columns 6 --anomaly_ratio 0.2 --output_dir ./my_data --visualize

# デモ実行（小規模データセット）
python create_csv.py  # 引数なしでデモ実行
```

### 3. 異常検知分析実行

```bash
# main.py内のパスを設定してから実行
python main.py
```

### 4. 結果可視化

```bash
# Streamlitアプリ起動
streamlit run visualization.py
```

ブラウザで `http://localhost:8501` にアクセスして結果を確認できます。

## 📁 ファイル構成

```
anomaly-detection-system/
├── main.py                    # メイン分析スクリプト
├── visualization.py           # Streamlit可視化アプリ
├── create_csv.py              # テストデータ生成スクリプト
├── requirements.txt           # 必要ライブラリリスト
├── README.md                  # このファイル
├── data/                      # データディレクトリ（自動生成）
│   ├── data_normal_001.csv    # 正常データ
│   ├── data_anomaly_002.csv   # 異常データ
│   └── ...
└── results/                   # 分析結果ディレクトリ（自動生成）
    ├── anomaly_detection_results.csv
    ├── anomaly_detection_results.json
    └── analysis_plots.png
```

## ⚙️ 設定・カスタマイズ

### main.py パラメータ設定

`main.py`内の`AnomalyDetectionConfig`クラスでパラメータを簡単に変更できます：

```python
class AnomalyDetectionConfig:
    # データ処理パラメータ
    DATA_PARAMS = {
        'missing_value_threshold': 0.1,  # 欠損値閾値
        'outlier_threshold': 3.0,        # 外れ値検出Z-score閾値
        'normalize_method': 'standard',   # 正規化方法
    }
    
    # マハラノビス距離パラメータ
    MAHALANOBIS_PARAMS = {
        'threshold_method': 'percentile', # 閾値設定方法
        'threshold_percentile': 95,       # パーセンタイル閾値
        'regularization': 1e-6,           # 正則化項
    }
    
    # SVM + マハラノビス距離パラメータ
    SVM_MAHALANOBIS_PARAMS = {
        'svm_params': {
            'kernel': 'rbf',              # カーネル種類
            'nu': 0.05,                   # 異常データ割合
        },
        'mahalanobis_weight': 0.5,        # マハラノビス距離の重み
    }
    
    # ... 他のパラメータ
```

### データパス設定

`main.py`の`main()`関数内でデータパスを設定：

```python
def main():
    # データパスを指定（ディレクトリまたは単一ファイル）
    data_path = "./data"  # 実際のパスに変更
    
    # 対象カラムを指定（Noneの場合は全数値列を使用）
    target_columns = None  # 例: ['sensor_1', 'sensor_2', 'temperature']
    
    # 分析実行
    results = system.run_analysis(data_path, target_columns)
```

## 📊 出力結果

### 分析結果ファイル

- **CSV形式** (`anomaly_detection_results.csv`): 
  - 元データ + 各手法の異常判定結果 + 異常スコア
- **JSON形式** (`anomaly_detection_results.json`):
  - 分析サマリー、設定パラメータ、特徴量名
- **画像ファイル** (`analysis_plots.png`):
  - 異常スコア分布、時系列プロット、相関分析

### Streamlit可視化機能

1. **📊 概要ダッシュボード**: 異常検知結果のサマリー
2. **📈 時系列データ**: データと異常点の時系列表示
3. **🎯 異常スコア分析**: スコアの詳細分析
4. **🔍 特徴量分析**: PCA、相関分析
5. **📋 統計レポート**: 詳細統計とエクスポート機能

## 🔧 トラブルシューティング

### よくある問題

1. **ファイルが読み込めない**
   - パスが正しいか確認
   - CSVファイルの形式を確認（ヘッダー行があるか）

2. **メモリ不足エラー**
   - データサイズを小さくする
   - ファイル数を減らす

3. **計算に時間がかかる**
   - `main.py`のグリッドサーチパラメータを減らす
   - 並列処理オプションを有効化

4. **Streamlitアプリが起動しない**
   - `streamlit --version`でインストール確認
   - ポート番号を変更: `streamlit run visualization.py --server.port 8502`

### エラー対処方法

```bash
# 依存関係の再インストール
pip install --upgrade -r requirements.txt

# キャッシュクリア
pip cache purge

# 仮想環境の再作成
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📈 使用例

### 例1: 基本的な使用方法

```bash
# 1. テストデータ生成
python create_csv.py --num_files 10 --visualize

# 2. 分析実行
python main.py

# 3. 結果確認
streamlit run visualization.py
```

### 例2: カスタム設定での分析

```python
# main.py内でパラメータ調整
config = AnomalyDetectionConfig()
config.MAHALANOBIS_PARAMS['threshold_percentile'] = 90
config.SVM_MAHALANOBIS_PARAMS['svm_params']['nu'] = 0.1

system = AnomalyDetectionSystem(config)
results = system.run_analysis("./custom_data", ['sensor_1', 'sensor_2'])
```

### 例3: 大規模データセット

```bash
# 大規模データ生成
python create_csv.py --num_files 50 --rows 10000 --columns 12

# 並列処理での分析（main.py内で設定）
python main.py
```

## 🛠️ 開発・拡張

### 新しい異常検知手法の追加

1. `main.py`に新しい検知器クラスを作成
2. `AnomalyDetectionSystem.detectors`に追加
3. `visualization.py`で新手法に対応

### カスタム特徴量の追加

`FeatureExtractor`クラスの`extract_features`メソッドを修正して新しい特徴量を追加できます。

## 📚 参考資料

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 コントリビューション

プルリクエストやイシューの報告を歓迎します！

---

**作成者**: AI Assistant  
**更新日**: 2024年12月
