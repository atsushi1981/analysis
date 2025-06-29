"""
時系列波形データ異常検知システム - Streamlit可視化アプリ
main.pyの分析結果を動的に可視化する
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import glob
from pathlib import Path
import io
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# スタイリング
st.set_page_config(
    page_title="異常検知分析ダッシュボード",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# キャッシュ機能を活用
@st.cache_data
def load_csv_file(file_path: str) -> pd.DataFrame:
    """CSVファイルを読み込み（キャッシュ付き）"""
    return pd.read_csv(file_path)

@st.cache_data
def load_json_results(file_path: str) -> dict:
    """JSON結果ファイルを読み込み（キャッシュ付き）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class AnomalyVisualizationApp:
    """異常検知結果可視化アプリのメインクラス"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """セッション状態の初期化"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'selected_methods' not in st.session_state:
            st.session_state.selected_methods = []
        if 'selected_columns' not in st.session_state:
            st.session_state.selected_columns = []
    
    def render_sidebar(self):
        """サイドバーのUI要素をレンダリング"""
        st.sidebar.title("🔍 異常検知ダッシュボード")
        st.sidebar.markdown("---")
        
        # データ読み込みセクション
        st.sidebar.header("📁 データ読み込み")
        
        # ファイルアップロード
        uploaded_file = st.sidebar.file_uploader(
            "結果CSVファイルをアップロード",
            type=['csv'],
            help="main.pyで生成された anomaly_detection_results.csv をアップロードしてください"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.sidebar.success(f"データ読み込み完了！ ({data.shape[0]}行, {data.shape[1]}列)")
            except Exception as e:
                st.sidebar.error(f"ファイル読み込みエラー: {e}")
        
        # ローカルファイル選択
        st.sidebar.subheader("ローカルファイル選択")
        results_dir = st.sidebar.text_input(
            "結果ディレクトリパス",
            value="./results",
            help="main.pyで出力された結果ディレクトリのパスを入力"
        )
        
        if st.sidebar.button("ローカルファイル読み込み"):
            self.load_local_results(results_dir)
        
        # データが読み込まれている場合の設定
        if st.session_state.data is not None:
            st.sidebar.markdown("---")
            self.render_analysis_settings()
    
    def load_local_results(self, results_dir: str):
        """ローカルの結果ファイルを読み込み"""
        try:
            csv_path = os.path.join(results_dir, "anomaly_detection_results.csv")
            json_path = os.path.join(results_dir, "anomaly_detection_results.json")
            
            if os.path.exists(csv_path):
                data = load_csv_file(csv_path)
                st.session_state.data = data
                st.sidebar.success(f"CSV読み込み完了！ ({data.shape[0]}行, {data.shape[1]}列)")
            
            if os.path.exists(json_path):
                results = load_json_results(json_path)
                st.session_state.results = results
                st.sidebar.success("JSON結果読み込み完了！")
            
            if not os.path.exists(csv_path):
                st.sidebar.error(f"CSVファイルが見つかりません: {csv_path}")
                
        except Exception as e:
            st.sidebar.error(f"ファイル読み込みエラー: {e}")
    
    def render_analysis_settings(self):
        """分析設定のUI要素をレンダリング"""
        st.sidebar.header("⚙️ 分析設定")
        
        data = st.session_state.data
        
        # 異常検知手法の選択
        anomaly_columns = [col for col in data.columns if '_anomaly' in col]
        method_names = [col.replace('_anomaly', '') for col in anomaly_columns]
        
        selected_methods = st.sidebar.multiselect(
            "異常検知手法選択",
            options=method_names,
            default=method_names,
            help="表示したい異常検知手法を選択してください"
        )
        st.session_state.selected_methods = selected_methods
        
        # データカラムの選択
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        # 異常検知関連の列を除外
        data_columns = [col for col in numeric_columns if not any(suffix in col for suffix in ['_anomaly', '_score'])]
        
        selected_columns = st.sidebar.multiselect(
            "表示データカラム選択",
            options=data_columns,
            default=data_columns[:5] if len(data_columns) > 5 else data_columns,
            help="可視化したいデータカラムを選択してください"
        )
        st.session_state.selected_columns = selected_columns
        
        # 表示設定
        st.sidebar.subheader("表示設定")
        
        show_anomaly_only = st.sidebar.checkbox(
            "異常点のみ表示",
            value=False,
            help="チェックすると異常と判定された点のみを表示します"
        )
        
        plot_height = st.sidebar.slider(
            "グラフの高さ",
            min_value=300,
            max_value=800,
            value=400,
            step=50,
            help="グラフの表示高さを調整"
        )
        
        return show_anomaly_only, plot_height
    
    def render_main_content(self):
        """メインコンテンツエリアをレンダリング"""
        st.title("🔍 時系列波形データ異常検知 - 分析ダッシュボード")
        
        if st.session_state.data is None:
            st.info("👈 サイドバーからデータファイルを読み込んでください")
            self.render_sample_data_info()
            return
        
        # タブ構成
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 概要ダッシュボード",
            "📈 時系列データ",
            "🎯 異常スコア分析",
            "🔍 特徴量分析",
            "📋 統計レポート"
        ])
        
        with tab1:
            self.render_overview_dashboard()
        
        with tab2:
            self.render_timeseries_analysis()
        
        with tab3:
            self.render_anomaly_score_analysis()
        
        with tab4:
            self.render_feature_analysis()
        
        with tab5:
            self.render_statistical_report()
    
    def render_sample_data_info(self):
        """サンプルデータの情報を表示"""
        st.markdown("## 🚀 クイックスタート")
        
        st.markdown("""
        ### 使用方法：
        1. **main.py** を実行して異常検知分析を行う
        2. 生成された **anomaly_detection_results.csv** をアップロード
        3. サイドバーから表示設定を調整
        4. 各タブで結果を確認
        """)
        
        st.markdown("### サンプルデータ構造：")
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'sensor_1': np.random.normal(0, 1, 10),
            'sensor_2': np.random.normal(0, 1, 10),
            'mahalanobis_anomaly': [False] * 8 + [True, False],
            'mahalanobis_score': np.random.uniform(0, 5, 10),
            'svm_mahalanobis_anomaly': [False] * 9 + [True],
            'svm_mahalanobis_score': np.random.uniform(0, 3, 10),
        })
        st.dataframe(sample_data)
    
    def render_overview_dashboard(self):
        """概要ダッシュボードをレンダリング"""
        st.header("📊 分析結果概要")
        
        data = st.session_state.data
        selected_methods = st.session_state.selected_methods
        
        if not selected_methods:
            st.warning("サイドバーから異常検知手法を選択してください")
            return
        
        # メトリクス表示
        cols = st.columns(len(selected_methods) + 1)
        
        with cols[0]:
            st.metric("総データ数", f"{len(data):,}")
        
        for i, method in enumerate(selected_methods):
            anomaly_col = f"{method}_anomaly"
            if anomaly_col in data.columns:
                n_anomalies = data[anomaly_col].sum()
                anomaly_rate = (n_anomalies / len(data)) * 100
                
                with cols[i + 1]:
                    st.metric(
                        f"{method.replace('_', ' ').title()}",
                        f"{n_anomalies:,}個",
                        f"{anomaly_rate:.2f}%"
                    )
        
        # 異常率比較チャート
        st.subheader("手法別異常検知率")
        
        anomaly_rates = []
        method_labels = []
        
        for method in selected_methods:
            anomaly_col = f"{method}_anomaly"
            if anomaly_col in data.columns:
                rate = (data[anomaly_col].sum() / len(data)) * 100
                anomaly_rates.append(rate)
                method_labels.append(method.replace('_', ' ').title())
        
        if anomaly_rates:
            fig = px.bar(
                x=method_labels,
                y=anomaly_rates,
                title="異常検知率比較",
                labels={'x': '手法', 'y': '異常率 (%)'},
                color=anomaly_rates,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # スコア分布ヒストグラム
        st.subheader("異常スコア分布")
        
        fig = make_subplots(
            rows=1, cols=len(selected_methods),
            subplot_titles=[m.replace('_', ' ').title() for m in selected_methods]
        )
        
        for i, method in enumerate(selected_methods):
            score_col = f"{method}_score"
            if score_col in data.columns:
                fig.add_trace(
                    go.Histogram(
                        x=data[score_col],
                        name=method,
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(height=400, title_text="手法別異常スコア分布")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_timeseries_analysis(self):
        """時系列分析をレンダリング"""
        st.header("📈 時系列データ分析")
        
        data = st.session_state.data
        selected_methods = st.session_state.selected_methods
        selected_columns = st.session_state.selected_columns
        
        if not selected_columns:
            st.warning("サイドバーから表示データカラムを選択してください")
            return
        
        # 設定の取得
        try:
            show_anomaly_only, plot_height = self.render_analysis_settings()
        except:
            show_anomaly_only, plot_height = False, 400
        
        # 時系列プロット
        for col in selected_columns:
            if col in data.columns:
                st.subheader(f"📊 {col}")
                
                fig = go.Figure()
                
                # 元データをプロット
                if not show_anomaly_only:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data[col],
                        mode='lines',
                        name='データ',
                        line=dict(color='blue', width=1),
                        opacity=0.7
                    ))
                
                # 異常点をハイライト
                colors = ['red', 'orange', 'purple', 'green', 'pink']
                for i, method in enumerate(selected_methods):
                    anomaly_col = f"{method}_anomaly"
                    if anomaly_col in data.columns:
                        anomaly_mask = data[anomaly_col]
                        anomaly_indices = data.index[anomaly_mask]
                        anomaly_values = data.loc[anomaly_mask, col]
                        
                        if len(anomaly_values) > 0:
                            fig.add_trace(go.Scatter(
                                x=anomaly_indices,
                                y=anomaly_values,
                                mode='markers',
                                name=f'{method} 異常点',
                                marker=dict(
                                    color=colors[i % len(colors)],
                                    size=8,
                                    symbol='diamond'
                                )
                            ))
                
                fig.update_layout(
                    title=f'{col} - 時系列データと異常点',
                    xaxis_title='インデックス',
                    yaxis_title=col,
                    height=plot_height,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 統計情報
                with st.expander(f"{col} の統計情報"):
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("平均", f"{data[col].mean():.4f}")
                    with stats_cols[1]:
                        st.metric("標準偏差", f"{data[col].std():.4f}")
                    with stats_cols[2]:
                        st.metric("最小値", f"{data[col].min():.4f}")
                    with stats_cols[3]:
                        st.metric("最大値", f"{data[col].max():.4f}")
    
    def render_anomaly_score_analysis(self):
        """異常スコア分析をレンダリング"""
        st.header("🎯 異常スコア詳細分析")
        
        data = st.session_state.data
        selected_methods = st.session_state.selected_methods
        
        if not selected_methods:
            st.warning("サイドバーから異常検知手法を選択してください")
            return
        
        # スコア時系列プロット
        st.subheader("異常スコアの時系列変化")
        
        fig = go.Figure()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, method in enumerate(selected_methods):
            score_col = f"{method}_score"
            if score_col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[score_col],
                    mode='lines',
                    name=f'{method} スコア',
                    line=dict(color=colors[i % len(colors)])
                ))
        
        fig.update_layout(
            title='異常スコア時系列',
            xaxis_title='インデックス',
            yaxis_title='異常スコア',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # スコア相関分析
        st.subheader("手法間スコア相関")
        
        score_data = {}
        for method in selected_methods:
            score_col = f"{method}_score"
            if score_col in data.columns:
                score_data[method] = data[score_col]
        
        if len(score_data) > 1:
            score_df = pd.DataFrame(score_data)
            correlation_matrix = score_df.corr()
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="手法間スコア相関マトリックス",
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # スコア統計テーブル
        st.subheader("異常スコア統計")
        
        stats_data = []
        for method in selected_methods:
            score_col = f"{method}_score"
            anomaly_col = f"{method}_anomaly"
            
            if score_col in data.columns and anomaly_col in data.columns:
                scores = data[score_col]
                anomalies = data[anomaly_col]
                
                stats_data.append({
                    '手法': method.replace('_', ' ').title(),
                    '平均スコア': f"{scores.mean():.4f}",
                    '標準偏差': f"{scores.std():.4f}",
                    '最小値': f"{scores.min():.4f}",
                    '最大値': f"{scores.max():.4f}",
                    '異常数': f"{anomalies.sum():,}",
                    '異常率': f"{(anomalies.sum() / len(data) * 100):.2f}%"
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
    
    def render_feature_analysis(self):
        """特徴量分析をレンダリング"""
        st.header("🔍 特徴量分析")
        
        data = st.session_state.data
        selected_columns = st.session_state.selected_columns
        
        if not selected_columns:
            st.warning("サイドバーから表示データカラムを選択してください")
            return
        
        # 相関分析
        st.subheader("データ相関分析")
        
        if len(selected_columns) > 1:
            correlation_data = data[selected_columns].corr()
            
            fig = px.imshow(
                correlation_data,
                text_auto=True,
                aspect="auto",
                title="データカラム間相関マトリックス",
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # 主成分分析（PCA）
        st.subheader("主成分分析 (PCA)")
        
        if len(selected_columns) >= 2:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # データの標準化
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data[selected_columns].fillna(0))
            
            # PCA実行
            pca = PCA(n_components=min(2, len(selected_columns)))
            pca_data = pca.fit_transform(scaled_data)
            
            # PCA結果をプロット
            fig = go.Figure()
            
            # 正常点
            normal_mask = True
            for method in st.session_state.selected_methods:
                anomaly_col = f"{method}_anomaly"
                if anomaly_col in data.columns:
                    normal_mask = normal_mask & (~data[anomaly_col])
            
            fig.add_trace(go.Scatter(
                x=pca_data[normal_mask, 0],
                y=pca_data[normal_mask, 1],
                mode='markers',
                name='正常データ',
                marker=dict(color='blue', size=4, opacity=0.6)
            ))
            
            # 異常点
            colors = ['red', 'orange', 'purple', 'green', 'pink']
            for i, method in enumerate(st.session_state.selected_methods):
                anomaly_col = f"{method}_anomaly"
                if anomaly_col in data.columns:
                    anomaly_mask = data[anomaly_col]
                    if anomaly_mask.sum() > 0:
                        fig.add_trace(go.Scatter(
                            x=pca_data[anomaly_mask, 0],
                            y=pca_data[anomaly_mask, 1],
                            mode='markers',
                            name=f'{method} 異常点',
                            marker=dict(
                                color=colors[i % len(colors)],
                                size=8,
                                symbol='diamond'
                            )
                        ))
            
            fig.update_layout(
                title='PCA散布図 (異常点ハイライト)',
                xaxis_title=f'第1主成分 (寄与率: {pca.explained_variance_ratio_[0]:.2%})',
                yaxis_title=f'第2主成分 (寄与率: {pca.explained_variance_ratio_[1]:.2%})',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 寄与率情報
            st.info(f"累積寄与率: {pca.explained_variance_ratio_.sum():.2%}")
        
        # データ分布分析
        st.subheader("データ分布分析")
        
        # 分布プロット
        col1, col2 = st.columns(2)
        
        with col1:
            if selected_columns:
                selected_col = st.selectbox(
                    "分布を表示するカラムを選択",
                    options=selected_columns,
                    key="dist_column"
                )
                
                if selected_col in data.columns:
                    fig = px.histogram(
                        data,
                        x=selected_col,
                        title=f'{selected_col} の分布',
                        nbins=50
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if selected_columns:
                # ボックスプロット
                fig = go.Figure()
                
                for col in selected_columns[:5]:  # 最大5カラムまで
                    fig.add_trace(go.Box(
                        y=data[col],
                        name=col,
                        boxpoints='outliers'
                    ))
                
                fig.update_layout(
                    title='データ分布ボックスプロット',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_statistical_report(self):
        """統計レポートをレンダリング"""
        st.header("📋 統計レポート")
        
        data = st.session_state.data
        selected_methods = st.session_state.selected_methods
        selected_columns = st.session_state.selected_columns
        
        # 基本統計情報
        st.subheader("基本統計情報")
        
        if selected_columns:
            basic_stats = data[selected_columns].describe()
            st.dataframe(basic_stats, use_container_width=True)
        
        # 異常検知結果サマリー
        st.subheader("異常検知結果サマリー")
        
        summary_data = []
        for method in selected_methods:
            anomaly_col = f"{method}_anomaly"
            score_col = f"{method}_score"
            
            if anomaly_col in data.columns and score_col in data.columns:
                anomalies = data[anomaly_col]
                scores = data[score_col]
                
                summary_data.append({
                    '検知手法': method.replace('_', ' ').title(),
                    '総データ数': len(data),
                    '異常データ数': anomalies.sum(),
                    '異常率': f"{(anomalies.sum() / len(data) * 100):.2f}%",
                    '平均異常スコア': f"{scores.mean():.4f}",
                    '異常スコア標準偏差': f"{scores.std():.4f}",
                    '最大異常スコア': f"{scores.max():.4f}",
                    '最小異常スコア': f"{scores.min():.4f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        
        # JSON結果表示
        if st.session_state.results is not None:
            st.subheader("詳細設定情報")
            
            with st.expander("設定パラメータ表示"):
                st.json(st.session_state.results)
        
        # データエクスポート
        st.subheader("📥 データエクスポート")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("CSVダウンロード"):
                csv_data = data.to_csv(index=False)
                st.download_button(
                    label="CSV形式でダウンロード",
                    data=csv_data,
                    file_name="anomaly_detection_results.csv",
                    mime="text/csv"
                )
        
        with col2:
            if selected_methods and st.button("サマリーレポート生成"):
                report = self.generate_text_report(data, selected_methods, selected_columns)
                st.download_button(
                    label="レポートダウンロード",
                    data=report,
                    file_name="anomaly_detection_report.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("フィルタ済みデータ"):
                if selected_methods:
                    # 異常データのみを抽出
                    anomaly_mask = False
                    for method in selected_methods:
                        anomaly_col = f"{method}_anomaly"
                        if anomaly_col in data.columns:
                            anomaly_mask = anomaly_mask | data[anomaly_col]
                    
                    anomaly_data = data[anomaly_mask]
                    csv_data = anomaly_data.to_csv(index=False)
                    st.download_button(
                        label="異常データのみダウンロード",
                        data=csv_data,
                        file_name="anomaly_only_data.csv",
                        mime="text/csv"
                    )
        
        # データ詳細テーブル
        st.subheader("📊 データ詳細テーブル")
        
        # フィルタ設定
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            show_anomalies_only = st.checkbox("異常データのみ表示", value=False)
        
        with filter_col2:
            max_rows = st.number_input(
                "最大表示行数",
                min_value=10,
                max_value=10000,
                value=100,
                step=10
            )
        
        # データフィルタリング
        display_data = data.copy()
        
        if show_anomalies_only and selected_methods:
            anomaly_mask = False
            for method in selected_methods:
                anomaly_col = f"{method}_anomaly"
                if anomaly_col in data.columns:
                    anomaly_mask = anomaly_mask | data[anomaly_col]
            display_data = display_data[anomaly_mask]
        
        # 表示行数制限
        display_data = display_data.head(max_rows)
        
        st.dataframe(display_data, use_container_width=True, height=400)
        
        # データ情報
        st.info(f"表示中のデータ: {len(display_data)}行 / 全体: {len(data)}行")
    
    def generate_text_report(self, data: pd.DataFrame, selected_methods: List[str], selected_columns: List[str]) -> str:
        """テキストレポートを生成"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("異常検知分析レポート")
        report_lines.append("=" * 60)
        report_lines.append(f"生成日時: {pd.Timestamp.now()}")
        report_lines.append(f"データサイズ: {data.shape[0]}行 × {data.shape[1]}列")
        report_lines.append("")
        
        # 異常検知結果サマリー
        report_lines.append("【異常検知結果サマリー】")
        for method in selected_methods:
            anomaly_col = f"{method}_anomaly"
            score_col = f"{method}_score"
            
            if anomaly_col in data.columns and score_col in data.columns:
                anomalies = data[anomaly_col]
                scores = data[score_col]
                
                report_lines.append(f"\n{method.replace('_', ' ').title()}:")
                report_lines.append(f"  異常データ数: {anomalies.sum():,}")
                report_lines.append(f"  異常率: {(anomalies.sum() / len(data) * 100):.2f}%")
                report_lines.append(f"  平均スコア: {scores.mean():.4f}")
                report_lines.append(f"  標準偏差: {scores.std():.4f}")
                report_lines.append(f"  最大スコア: {scores.max():.4f}")
                report_lines.append(f"  最小スコア: {scores.min():.4f}")
        
        # データ統計
        if selected_columns:
            report_lines.append("\n【データ統計】")
            for col in selected_columns:
                if col in data.columns:
                    col_data = data[col]
                    report_lines.append(f"\n{col}:")
                    report_lines.append(f"  平均: {col_data.mean():.4f}")
                    report_lines.append(f"  標準偏差: {col_data.std():.4f}")
                    report_lines.append(f"  最小値: {col_data.min():.4f}")
                    report_lines.append(f"  最大値: {col_data.max():.4f}")
                    report_lines.append(f"  欠損値: {col_data.isnull().sum()}")
        
        report_lines.append("\n" + "=" * 60)
        report_lines.append("レポート終了")
        
        return "\n".join(report_lines)
    
    def run(self):
        """アプリケーション実行"""
        # サイドバー
        self.render_sidebar()
        
        # メインコンテンツ
        self.render_main_content()


def main():
    """メイン実行関数"""
    # カスタムCSS
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric > div > div > div > div {
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # アプリケーション実行
    app = AnomalyVisualizationApp()
    app.run()


if __name__ == "__main__":
    main()