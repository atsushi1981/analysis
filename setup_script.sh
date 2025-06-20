#!/bin/bash

# 異常検知システムのセットアップスクリプト

echo "=== 異常検知システム セットアップ ==="

# Python環境の確認
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3が見つかりません。Python 3.8以上をインストールしてください。"
    exit 1
fi

echo "✅ Python3が見つかりました: $(python3 --version)"

# 仮想環境の作成
echo "📦 仮想環境を作成中..."
python3 -m venv venv

# 仮想環境の有効化
echo "🔄 仮想環境を有効化中..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# 依存関係のインストール
echo "📥 依存関係をインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

# ディレクトリ構造の作成
echo "📁 ディレクトリ構造を作成中..."
mkdir -p input_data
mkdir -p results/csv
mkdir -p results/graphs
mkdir -p logs
mkdir -p models
mkdir -p utils
mkdir -p viewer

# 完了メッセージ
echo ""
echo "✅ セットアップが完了しました！"
echo ""
echo "使用方法:"
echo "1. CSVファイルを input_data/ フォルダに配置"
echo "2. config.yaml で設定を調整"
echo "3. 分析実行: python main.py"
echo "4. ビューアー起動: streamlit run viewer/run_viewer.py"
echo ""
echo "仮想環境の有効化:"
echo "  Linux/Mac: source venv/bin/activate"
echo "  Windows: venv\\Scripts\\activate"