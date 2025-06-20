#!/usr/bin/env python3
"""
異常検知結果ビューアー（Streamlit版）
ローカルブラウザで結果を確認・分析できるWebアプリケーション
"""

import os
import sys
import glob
import yaml
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# プロジェクトパスの追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.plotter import Plotter
from models.svr_mahalanobis import SVRMahalanobis
from models.svm_mahalanobis import SVMMahalanobis
from models.mahalanobis_only import MahalanobisOnly


# ページ設定
st.set_page_config(
    page_title="異常検知ビューアー",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_config():
    """設定ファイルの読み込み"""
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@st.cache_data
def get_csv_files(directory):
    """CSVファイルのリスト取得"""
    pattern = os.path.join(directory, '*.csv')
    files = glob.glob(pattern)
    return sorted(files)


@st.cache_data
def load_data(filepath):
    """データの読み込み"""
    return pd.read_csv(filepath)


def create_anomaly_plot(data, anomaly_scores, anomaly_flags, column_name, threshold):
    """異常検知結果のプロット作成"""
    x = np.arange(len(data))
    
    fig = go.Figure()
    
    # 元データ
    fig.add_trace(go.Scatter(
        x=x,
        y=data,
        mode='lines',
        name='元データ',
        line=dict(color='blue', width=1)
    ))
    
    # 異常点
    anomaly_indices = np.where(anomaly_flags)[0]
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_indices,
            y=data[anomaly_indices],
            mode='markers',
            name='異常点',
            marker=dict(color='red', size=8, symbol='circle')
        ))
    
    # 異常スコア（第2軸）
    fig.add_trace(go.Scatter(
        x=x,
        y=anomaly_scores,
        mode='lines',
        name='異常スコア',
        line=dict(color='green', width=1),
        yaxis='y2'
    ))
    
    # 閾値ライン
    fig.add_hline(
        y=threshold,
        line=dict(color='red', dash='dash'),
        annotation_text=f"閾値: {threshold:.2f}",
        yref='y2'
    )
    
    # レイアウト設定
    fig.update_layout(
        title=f'{column_name} - 異常検知結果',
        xaxis=dict(title='インデックス'),
        yaxis=dict(title='値', side='left'),
        yaxis2=dict(title='異常スコア', side='right', overlaying='y'),
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99),
        height=600
    )
    
    return fig


def main():
    """メイン処理"""
    st.title("🔍 異常検知システム ビューアー")
    st.markdown("---")
    
    # 設定読み込み
    config = load_config()
    
    # サイドバー
    with st.sidebar:
        st.header("設定")
        
        # モード選択
        mode = st.selectbox(
            "異常検知手法",
            options=['svr_mahalanobis', 'svm_mahalanobis', 'mahalanobis_only'],
            index=['svr_mahalanobis', 'svm_mahalanobis', 'mahalanobis_only'].index(config['mode'])
        )
        
        # ファイル選択
        st.subheader("データ選択")
        
        # 入力データか結果データかを選択
        data_source = st.radio(
            "データソース",
            options=['入力データ', '結果データ']
        )
        
        if data_source == '入力データ':
            csv_dir = config['paths']['input_dir']
        else:
            csv_dir = config['paths']['csv_output_dir']
        
        csv_files = get_csv_files(csv_dir)
        
        if not csv_files:
            st.warning(f"CSVファイルが見つかりません: {csv_dir}")
            return
        
        selected_file = st.selectbox(
            "ファイル選択",
            options=csv_files,
            format_func=lambda x: os.path.basename(x)
        )
        
        # データ読み込み
        if selected_file:
            data = load_data(selected_file)
            
            # カラム選択
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # 結果データの場合は異常スコアカラムを除外
            if data_source == '結果データ' and 'anomaly_score' in numeric_columns:
                numeric_columns.remove('anomaly_score')
            if data_source == '結果データ' and 'is_anomaly' in numeric_columns:
                numeric_columns.remove('is_anomaly')
            
            selected_columns = st.multiselect(
                "表示カラム",
                options=numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))]
            )
        
        # 閾値設定
        st.subheader("閾値設定")
        
        percentile_threshold = st.slider(
            "パーセンタイル閾値",
            min_value=80,
            max_value=99,
            value=config['thresholds']['mahalanobis_percentile'],
            step=1
        )
        
        z_score_threshold = st.slider(
            "Zスコア閾値",
            min_value=1.0,
            max_value=5.0,
            value=config['thresholds']['z_score'],
            step=0.1
        )
        
        # 実行ボタン
        st.markdown("---")
        run_analysis = st.button("🚀 異常検知実行", type="primary", use_container_width=True)
    
    # メインコンテンツ
    if selected_file and selected_columns:
        # データ情報表示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("データ点数", f"{len(data):,}")
        with col2:
            st.metric("選択カラム数", len(selected_columns))
        with col3:
            st.metric("ファイルサイズ", f"{os.path.getsize(selected_file) / 1024:.1f} KB")
        
        # タブ作成
        tabs = st.tabs(["📊 グラフ表示", "📋 データ表示", "📈 統計情報"])
        
        with tabs[0]:
            if run_analysis or (data_source == '結果データ' and 'anomaly_score' in data.columns):
                # 異常検知実行または結果表示
                if data_source == '入力データ' and run_analysis:
                    # 新規実行
                    with st.spinner("異常検知を実行中..."):
                        # モデル初期化
                        if mode == 'svr_mahalanobis':
                            model = SVRMahalanobis(config, None)
                        elif mode == 'svm_mahalanobis':
                            model = SVMMahalanobis(config, None)
                        else:
                            model = MahalanobisOnly(config, None)
                        
                        # 異常検知実行
                        target_data = data[selected_columns]
                        anomaly_scores, anomaly_flags = model.detect_anomalies(target_data)
                        
                        # 結果表示
                        st.success(
                            f"異常検知完了: {sum(anomaly_flags)}/{len(anomaly_flags)} 点 "
                            f"({sum(anomaly_flags)/len(anomaly_flags)*100:.2f}%)"
                        )
                else:
                    # 既存結果の読み込み
                    anomaly_scores = data['anomaly_score'].values
                    anomaly_flags = data['is_anomaly'].values
                
                # カラムごとにグラフ表示
                for col in selected_columns:
                    threshold = np.percentile(anomaly_scores, percentile_threshold)
                    fig = create_anomaly_plot(
                        data[col].values,
                        anomaly_scores,
                        anomaly_flags,
                        col,
                        threshold
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # 単純なデータプロット
                for col in selected_columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(data)),
                        y=data[col],
                        mode='lines',
                        name=col
                    ))
                    fig.update_layout(
                        title=f'{col} - 時系列データ',
                        xaxis_title='インデックス',
                        yaxis_title='値',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            # データ表示
            st.subheader("データプレビュー")
            
            # 表示行数選択
            n_rows = st.slider("表示行数", 10, 100, 50)
            
            # データ表示
            display_cols = selected_columns.copy()
            if data_source == '結果データ':
                if 'anomaly_score' in data.columns:
                    display_cols.append('anomaly_score')
                if 'is_anomaly' in data.columns:
                    display_cols.append('is_anomaly')
            
            st.dataframe(
                data[display_cols].head(n_rows),
                use_container_width=True
            )
            
            # データダウンロード
            csv = data[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 データをダウンロード",
                data=csv,
                file_name=f"anomaly_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tabs[2]:
            # 統計情報表示
            st.subheader("基本統計量")
            st.dataframe(
                data[selected_columns].describe(),
                use_container_width=True
            )
            
            # 相関行列
            if len(selected_columns) > 1:
                st.subheader("相関行列")
                corr = data[selected_columns].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="カラム間の相関係数",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("サイドバーからファイルとカラムを選択してください。")


if __name__ == '__main__':
    main()