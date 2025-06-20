"""
プロット生成ユーティリティ
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import numpy as np


class Plotter:
    """プロッタークラス"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # matplotlib設定
        plt.rcParams['figure.figsize'] = config['plot_settings']['figure_size']
        plt.rcParams['figure.dpi'] = config['plot_settings']['dpi']
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_anomaly_results(self, data, anomaly_scores, anomaly_flags, 
                           title='Anomaly Detection Results', save_path=None):
        """異常検知結果のプロット（matplotlib）"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config['plot_settings']['figure_size'])
            
            # 元データのプロット
            x = np.arange(len(data))
            ax1.plot(x, data, 'b-', linewidth=0.8, label='Original Data')
            
            # 異常点のハイライト
            anomaly_indices = np.where(anomaly_flags)[0]
            if len(anomaly_indices) > 0:
                ax1.scatter(anomaly_indices, data[anomaly_indices], 
                          color='red', s=30, alpha=0.7, label='Anomaly')
            
            ax1.set_title(f'{title} - Original Data with Anomalies')
            ax1.set_xlabel('Index')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(self.config['plot_settings']['show_grid'], alpha=0.3)
            
            # 異常スコアのプロット
            ax2.plot(x, anomaly_scores, 'g-', linewidth=0.8, label='Anomaly Score')
            
            # 閾値ライン
            threshold = np.percentile(anomaly_scores, self.config['thresholds']['mahalanobis_percentile'])
            ax2.axhline(y=threshold, color='r', linestyle='--', 
                       linewidth=1, label=f'Threshold (95%ile={threshold:.2f})')
            
            # 異常領域のハイライト
            if len(anomaly_indices) > 0:
                for idx in anomaly_indices:
                    ax2.axvspan(idx-0.5, idx+0.5, alpha=0.3, color='red')
            
            ax2.set_title('Anomaly Scores')
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Score')
            ax2.legend()
            ax2.grid(self.config['plot_settings']['show_grid'], alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config['plot_settings']['dpi'], 
                          bbox_inches='tight')
                self.logger.info(f"グラフ保存: {save_path}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"プロットエラー: {str(e)}")
    
    def create_interactive_plot(self, data, anomaly_scores, anomaly_flags, 
                               title='Anomaly Detection Results'):
        """インタラクティブプロットの作成（Plotly）"""
        try:
            # インデックス作成
            x = np.arange(len(data))
            
            # サブプロットの作成
            fig = go.Figure()
            
            # 元データ
            fig.add_trace(go.Scatter(
                x=x,
                y=data,
                mode='lines',
                name='Original Data',
                line=dict(color='blue', width=1),
                yaxis='y'
            ))
            
            # 異常点
            anomaly_indices = np.where(anomaly_flags)[0]
            if len(anomaly_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly_indices,
                    y=data[anomaly_indices],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=8, symbol='circle'),
                    yaxis='y'
                ))
            
            # 異常スコア（第2軸）
            fig.add_trace(go.Scatter(
                x=x,
                y=anomaly_scores,
                mode='lines',
                name='Anomaly Score',
                line=dict(color='green', width=1),
                yaxis='y2'
            ))
            
            # 閾値ライン
            threshold = np.percentile(anomaly_scores, self.config['thresholds']['mahalanobis_percentile'])
            fig.add_hline(
                y=threshold,
                line=dict(color='red', dash='dash'),
                annotation_text=f"Threshold: {threshold:.2f}",
                yref='y2'
            )
            
            # レイアウト設定
            fig.update_layout(
                title=title,
                xaxis=dict(title='Index'),
                yaxis=dict(title='Original Value', side='left'),
                yaxis2=dict(title='Anomaly Score', side='right', overlaying='y'),
                hovermode='x unified',
                legend=dict(x=0.01, y=0.99),
                height=600
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"インタラクティブプロット作成エラー: {str(e)}")
            return None