# 異常検知システム

完全オフライン環境で動作する、複数の異常検知手法を切り替えて実行できる分析アプリケーションです。

## 特徴

- **3つの異常検知手法**
  1. SVR + マハラノビス距離
  2. One-Class SVM + マハラノビス距離
  3. マハラノビス距離のみ

- **完全ローカル動作**
  - インターネット接続不要
  - すべての処理がローカルで完結

- **インタラクティブなビューアー**
  - Streamlitベースの使いやすいUI
  - リアルタイムでパラメータ調整
  - Plotlyによる拡大・ホバー可能なグラフ

## システム要件

- Python 3.8以上
- 必要なライブラリは `requirements.txt` に記載

## インストール

### 1. リポジトリのクローンまたはダウンロード

```bash
git clone <repository_url>
cd anomaly-detection-system
```

### 2. セットアップスクリプトの実行

Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

Windows (手動セットアップ):
```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt

# ディレクトリの作成
mkdir input_data
mkdir results\csv
mkdir results\graphs
mkdir logs
```

## ディレクトリ構成

```
project_root/
├── config.yaml          # 設定ファイル
├── main.py             # メイン実行スクリプト
├── requirements.txt    # 依存ライブラリ
├── setup.sh           # セットアップスクリプト
├── README.md          # このファイル
├── models/            # 異常検知モデル
│   ├── svr_mahalanobis.py
│   ├── svm_mahalanobis.py
│   └── mahalanobis_only.py
├── utils/             # ユーティリティ
│   ├── data_loader.py
│   ├── plotter.py
│   └── logger.py
├── viewer/            # ビューアーアプリ
│   └── run_viewer.py
├── input_data/        # 入力CSVファイル格納
├── results/           # 結果出力
│   ├── csv/          # 結果CSV
│   └── graphs/       # グラフ画像
└── logs/             # ログファイル
```

## 使用方法

### 1. 設定ファイルの編集

`config.yaml` を編集して、使用する異常検知手法やパラメータを設定します。

```yaml
# 実行モード
mode: svr_mahalanobis  # または svm_mahalanobis, mahalanobis_only

# 対象カラム
target_columns:
  - column1
  - column2
  - column3

# 閾値設定
thresholds:
  z_score: 3.0
  mahalanobis_percentile: 95
```

### 2. データの準備

分析対象のCSVファイルを `input_data/` フォルダに配置します。
- 1ショット = 1CSVファイル
- 約7000行 × 数列の形式

### 3. 異常検知の実行

```bash
# 仮想環境の有効化
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows

# 異常検知実行
python main.py
```

### 4. ビューアーの起動

```bash
streamlit run viewer/run_viewer.py
```

ブラウザが自動的に開き、`http://localhost:8501` でビューアーが起動します。

## ビューアーの機能

- **手法選択**: ドロップダウンで異常検知手法を切り替え
- **ファイル選択**: 入力データまたは結果データを選択
- **カラム選択**: 表示するカラムを複数選択可能
- **閾値調整**: スライダーで閾値をリアルタイム調整
- **グラフ表示**: 
  - 元データと異常点の可視化
  - 異常スコアの表示
  - Plotlyによるインタラクティブな操作
- **データ表示**: 数値データの確認とダウンロード
- **統計情報**: 基本統計量と相関行列の表示

## 出力ファイル

### 結果CSV (`results/csv/`)
- `{元ファイル名}_result_{タイムスタンプ}.csv`
- 元データに異常スコアと異常フラグを追加

### グラフ画像 (`results/graphs/`)
- `{元ファイル名}_{カラム名}_{タイムスタンプ}.png`
- 各カラムの異常検知結果グラフ

### ログファイル (`logs/`)
- `anomaly_detection_{日付}.log`
- 処理の詳細ログ

## カスタマイズ

### モデルパラメータの調整

`config.yaml` でモデルパラメータを調整できます：

```yaml
model_params:
  svr:
    kernel: rbf
    C: 1.0
    epsilon: 0.1
  svm:
    kernel: rbf
    C: 1.0
    nu: 0.1
```

### 新しい異常検知手法の追加

1. `models/` に新しいモデルクラスを作成
2. `detect_anomalies(data)` メソッドを実装
3. `main.py` の `_initialize_model()` に追加

## トラブルシューティング

### CSVファイルが読み込めない
- ファイルのエンコーディングがUTF-8であることを確認
- カラム名に日本語が含まれる場合は英数字に変更

### メモリエラーが発生する
- 大きなCSVファイルの場合、バッチ処理を検討
- `config.yaml` でウィンドウサイズを調整

### ビューアーが起動しない
- ポート8501が使用されていないか確認
- `streamlit run viewer/run_viewer.py --server.port 8502` で別ポートを指定

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。