"""
時系列波形データCSVファイル生成スクリプト
異常検知システムのテスト用に複数の時系列データを生成
"""

import numpy as np
import pandas as pd
import os
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeSeriesDataGenerator:
    """時系列波形データ生成クラス"""
    
    def __init__(self, 
                 num_files: int = 15,
                 rows_per_file: int = 7500,
                 num_columns: int = 8,
                 anomaly_ratio: float = 0.15,
                 output_dir: str = "./data",
                 random_seed: int = 42):
        """
        Parameters:
        -----------
        num_files : int
            生成するファイル数
        rows_per_file : int
            1ファイルあたりの行数
        num_columns : int
            データカラム数（タイムスタンプ除く）
        anomaly_ratio : float
            異常データの割合
        output_dir : str
            出力ディレクトリ
        random_seed : int
            乱数シード
        """
        self.num_files = num_files
        self.rows_per_file = rows_per_file
        self.num_columns = num_columns
        self.anomaly_ratio = anomaly_ratio
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        
        # 乱数シード設定
        np.random.seed(random_seed)
        
        # パラメータ設定
        self.setup_parameters()
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成ログ
        self.generation_log = []
    
    def setup_parameters(self):
        """データ生成パラメータの設定"""
        # 波形パラメータ
        self.wave_params = {
            'frequencies': [0.1, 0.2, 0.5, 1.0, 2.0],  # 基本周波数
            'amplitudes': [1.0, 1.5, 2.0, 0.5, 0.8],   # 振幅
            'phases': [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2],  # 位相
            'noise_levels': [0.1, 0.2, 0.05, 0.15, 0.3],  # ノイズレベル
        }
        
        # 異常パターン設定
        self.anomaly_params = {
            'spike_intensity': [3.0, 5.0, 7.0],        # スパイク強度
            'drift_rates': [0.001, 0.002, 0.005],      # ドリフト率
            'frequency_shift': [0.1, 0.2, 0.3],        # 周波数変化
            'amplitude_change': [1.5, 2.0, 3.0],       # 振幅変化
        }
        
        # センサーカラム名
        self.column_names = [
            'sensor_1', 'sensor_2', 'temperature', 'pressure', 
            'vibration', 'voltage', 'current', 'speed',
            'flow_rate', 'humidity', 'acceleration', 'force'
        ]
    
    def generate_normal_waveform(self, length: int, column_index: int) -> np.ndarray:
        """正常な波形データを生成"""
        t = np.linspace(0, length / 100, length)  # 時間軸（100Hzサンプリング想定）
        
        # パラメータ選択
        freq = self.wave_params['frequencies'][column_index % len(self.wave_params['frequencies'])]
        amp = self.wave_params['amplitudes'][column_index % len(self.wave_params['amplitudes'])]
        phase = self.wave_params['phases'][column_index % len(self.wave_params['phases'])]
        noise_level = self.wave_params['noise_levels'][column_index % len(self.wave_params['noise_levels'])]
        
        # 基本波形（正弦波 + 余弦波の組み合わせ）
        signal = amp * np.sin(2 * np.pi * freq * t + phase)
        signal += 0.5 * amp * np.cos(2 * np.pi * freq * 2 * t + phase/2)
        
        # トレンド成分
        trend = 0.001 * t * np.sin(0.01 * t)
        
        # ノイズ
        noise = np.random.normal(0, noise_level, length)
        
        # 季節性成分（長周期）
        seasonal = 0.2 * amp * np.sin(2 * np.pi * 0.01 * t)
        
        return signal + trend + noise + seasonal
    
    def inject_spike_anomaly(self, data: np.ndarray, intensity: float = 5.0, num_spikes: int = 5) -> Tuple[np.ndarray, List[int]]:
        """スパイク異常を注入"""
        anomaly_data = data.copy()
        spike_indices = []
        
        for _ in range(num_spikes):
            spike_idx = np.random.randint(100, len(data) - 100)  # 端を避ける
            spike_value = intensity * np.std(data) * np.random.choice([-1, 1])
            anomaly_data[spike_idx] += spike_value
            spike_indices.append(spike_idx)
        
        return anomaly_data, spike_indices
    
    def inject_drift_anomaly(self, data: np.ndarray, drift_rate: float = 0.002, start_point: float = 0.3) -> Tuple[np.ndarray, Tuple[int, int]]:
        """ドリフト異常を注入"""
        anomaly_data = data.copy()
        start_idx = int(len(data) * start_point)
        
        # ドリフト関数
        drift_length = len(data) - start_idx
        drift = np.linspace(0, drift_rate * drift_length, drift_length)
        anomaly_data[start_idx:] += drift * np.std(data)
        
        return anomaly_data, (start_idx, len(data) - 1)
    
    def inject_frequency_anomaly(self, data: np.ndarray, frequency_shift: float = 0.2, 
                                start_point: float = 0.4, duration: float = 0.3) -> Tuple[np.ndarray, Tuple[int, int]]:
        """周波数変化異常を注入"""
        anomaly_data = data.copy()
        start_idx = int(len(data) * start_point)
        end_idx = int(start_idx + len(data) * duration)
        
        # 異常セクションの長さ
        anomaly_length = end_idx - start_idx
        t = np.linspace(0, anomaly_length / 100, anomaly_length)
        
        # 新しい周波数成分
        original_amp = np.std(data)
        freq_anomaly = original_amp * np.sin(2 * np.pi * frequency_shift * t)
        
        anomaly_data[start_idx:end_idx] += freq_anomaly
        
        return anomaly_data, (start_idx, end_idx)
    
    def inject_amplitude_anomaly(self, data: np.ndarray, amplitude_change: float = 2.0, 
                                start_point: float = 0.6, duration: float = 0.2) -> Tuple[np.ndarray, Tuple[int, int]]:
        """振幅変化異常を注入"""
        anomaly_data = data.copy()
        start_idx = int(len(data) * start_point)
        end_idx = int(start_idx + len(data) * duration)
        
        # 振幅変化
        anomaly_data[start_idx:end_idx] *= amplitude_change
        
        return anomaly_data, (start_idx, end_idx)
    
    def generate_single_file_data(self, file_index: int, is_anomaly_file: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """単一ファイルのデータを生成"""
        try:
            print(f"  ファイル {file_index + 1} を生成中... ({'異常' if is_anomaly_file else '正常'})")
            
            # タイムスタンプ生成
            start_time = datetime(2024, 1, 1) + timedelta(days=file_index)
            timestamps = pd.date_range(start_time, periods=self.rows_per_file, freq='6S')  # 6秒間隔
            
            # データ初期化
            data_dict = {'timestamp': timestamps}
            anomaly_info = {
                'file_index': file_index,
                'is_anomaly': is_anomaly_file,
                'anomaly_details': []
            }
            
            # 各カラムのデータ生成
            for col_idx in range(self.num_columns):
                col_name = self.column_names[col_idx % len(self.column_names)]
                if col_idx >= len(self.column_names):
                    col_name = f"{col_name}_{col_idx}"
                
                # 正常波形生成
                normal_data = self.generate_normal_waveform(self.rows_per_file, col_idx)
                
                if is_anomaly_file:
                    # 異常注入（ランダムに選択）
                    anomaly_type = np.random.choice(['spike', 'drift', 'frequency', 'amplitude'])
                    
                    if anomaly_type == 'spike':
                        intensity = np.random.choice(self.anomaly_params['spike_intensity'])
                        num_spikes = np.random.randint(3, 8)
                        anomaly_data, spike_indices = self.inject_spike_anomaly(normal_data, intensity, num_spikes)
                        anomaly_info['anomaly_details'].append({
                            'column': col_name,
                            'type': 'spike',
                            'indices': spike_indices,
                            'intensity': intensity
                        })
                    
                    elif anomaly_type == 'drift':
                        drift_rate = np.random.choice(self.anomaly_params['drift_rates'])
                        start_point = np.random.uniform(0.2, 0.5)
                        anomaly_data, (start_idx, end_idx) = self.inject_drift_anomaly(normal_data, drift_rate, start_point)
                        anomaly_info['anomaly_details'].append({
                            'column': col_name,
                            'type': 'drift',
                            'start_index': start_idx,
                            'end_index': end_idx,
                            'drift_rate': drift_rate
                        })
                    
                    elif anomaly_type == 'frequency':
                        freq_shift = np.random.choice(self.anomaly_params['frequency_shift'])
                        start_point = np.random.uniform(0.3, 0.6)
                        duration = np.random.uniform(0.2, 0.4)
                        anomaly_data, (start_idx, end_idx) = self.inject_frequency_anomaly(normal_data, freq_shift, start_point, duration)
                        anomaly_info['anomaly_details'].append({
                            'column': col_name,
                            'type': 'frequency',
                            'start_index': start_idx,
                            'end_index': end_idx,
                            'frequency_shift': freq_shift
                        })
                    
                    else:  # amplitude
                        amp_change = np.random.choice(self.anomaly_params['amplitude_change'])
                        start_point = np.random.uniform(0.5, 0.8)
                        duration = np.random.uniform(0.1, 0.3)
                        anomaly_data, (start_idx, end_idx) = self.inject_amplitude_anomaly(normal_data, amp_change, start_point, duration)
                        anomaly_info['anomaly_details'].append({
                            'column': col_name,
                            'type': 'amplitude',
                            'start_index': start_idx,
                            'end_index': end_idx,
                            'amplitude_change': amp_change
                        })
                    
                    data_dict[col_name] = anomaly_data
                else:
                    data_dict[col_name] = normal_data
                
                # 欠損値をランダムに挿入（現実的なデータを模擬）
                missing_ratio = np.random.uniform(0.001, 0.01)  # 0.1%～1%の欠損
                missing_indices = np.random.choice(
                    self.rows_per_file, 
                    size=int(self.rows_per_file * missing_ratio), 
                    replace=False
                )
                if len(missing_indices) > 0:
                    data_dict[col_name] = np.array(data_dict[col_name])
                    data_dict[col_name][missing_indices] = np.nan
            
            # DataFrame作成
            df = pd.DataFrame(data_dict)
            
            return df, anomaly_info
            
        except Exception as e:
            logger.error(f"ファイル{file_index}のデータ生成でエラー: {e}")
            raise
    
    def generate_all_files(self) -> Dict:
        """全ファイルを生成"""
        logger.info(f"データ生成開始: {self.num_files}ファイル")
        print(f"出力ディレクトリ: {self.output_dir}")
        
        # 異常ファイルのインデックスを決定
        num_anomaly_files = int(self.num_files * self.anomaly_ratio)
        anomaly_file_indices = np.random.choice(
            self.num_files, 
            size=num_anomaly_files, 
            replace=False
        )
        
        generation_summary = {
            'total_files': self.num_files,
            'normal_files': self.num_files - num_anomaly_files,
            'anomaly_files': num_anomaly_files,
            'anomaly_file_indices': anomaly_file_indices.tolist(),
            'file_details': []
        }
        
        # ファイル生成ループ
        for file_idx in range(self.num_files):
            is_anomaly = file_idx in anomaly_file_indices
            
            try:
                # データ生成
                df, anomaly_info = self.generate_single_file_data(file_idx, is_anomaly)
                
                # ファイル名決定
                if is_anomaly:
                    filename = f"data_anomaly_{file_idx:03d}.csv"
                else:
                    filename = f"data_normal_{file_idx:03d}.csv"
                
                # ファイル保存
                file_path = self.output_dir / filename
                df.to_csv(file_path, index=False)
                
                # ログ記録
                logger.info(f"ファイル生成完了: {filename} ({'異常' if is_anomaly else '正常'})")
                
                # サマリー情報追加
                generation_summary['file_details'].append({
                    'filename': filename,
                    'file_index': file_idx,
                    'is_anomaly': is_anomaly,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'anomaly_info': anomaly_info
                })
                
                self.generation_log.append({
                    'filename': filename,
                    'status': 'success',
                    'anomaly_type': 'anomaly' if is_anomaly else 'normal'
                })
                
            except Exception as e:
                logger.error(f"ファイル{file_idx}の生成でエラー: {e}")
                self.generation_log.append({
                    'filename': f"error_{file_idx:03d}.csv",
                    'status': 'error',
                    'error': str(e)
                })
        
        # サマリー保存
        try:
            summary_path = self.output_dir / "generation_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(generation_summary, f, indent=2, ensure_ascii=False, default=str)
            
            # 統計情報保存
            self.save_statistics(generation_summary)
            
        except Exception as e:
            logger.error(f"サマリー保存エラー: {e}")
        
        logger.info(f"全ファイル生成完了: {self.output_dir}")
        return generation_summary
    
    def save_statistics(self, generation_summary: Dict):
        """統計情報をファイルに保存"""
        try:
            stats_lines = []
            stats_lines.append("=" * 60)
            stats_lines.append("時系列波形データ生成統計")
            stats_lines.append("=" * 60)
            stats_lines.append(f"生成日時: {datetime.now()}")
            stats_lines.append(f"総ファイル数: {generation_summary['total_files']}")
            stats_lines.append(f"正常ファイル数: {generation_summary['normal_files']}")
            stats_lines.append(f"異常ファイル数: {generation_summary['anomaly_files']}")
            stats_lines.append(f"異常率: {(generation_summary['anomaly_files'] / generation_summary['total_files'] * 100):.1f}%")
            stats_lines.append(f"1ファイルあたりの行数: {self.rows_per_file}")
            stats_lines.append(f"データカラム数: {self.num_columns}")
            stats_lines.append("")
            
            stats_lines.append("【生成パラメータ】")
            stats_lines.append(f"乱数シード: {self.random_seed}")
            stats_lines.append(f"出力ディレクトリ: {self.output_dir}")
            stats_lines.append("")
            
            stats_lines.append("【異常パターン詳細】")
            anomaly_types = {}
            for file_detail in generation_summary['file_details']:
                if file_detail['is_anomaly']:
                    for anomaly_detail in file_detail['anomaly_info']['anomaly_details']:
                        anomaly_type = anomaly_detail['type']
                        anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
            
            for anomaly_type, count in anomaly_types.items():
                stats_lines.append(f"{anomaly_type}: {count}回")
            
            stats_lines.append("")
            stats_lines.append("【ファイル一覧】")
            for file_detail in generation_summary['file_details']:
                anomaly_status = "異常" if file_detail['is_anomaly'] else "正常"
                stats_lines.append(f"{file_detail['filename']}: {anomaly_status}")
            
            stats_lines.append("")
            stats_lines.append("=" * 60)
            
            # 統計ファイル保存
            stats_path = self.output_dir / "generation_statistics.txt"
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(stats_lines))
            
            logger.info(f"統計情報保存完了: {stats_path}")
            
        except Exception as e:
            logger.error(f"統計情報保存エラー: {e}")
    
    def visualize_sample_data(self, num_samples: int = 3):
        """サンプルデータの可視化"""
        try:
            csv_files = list(self.output_dir.glob("*.csv"))
            
            if len(csv_files) == 0:
                logger.warning("可視化対象のCSVファイルが見つかりません")
                return
            
            # サンプルファイルを選択
            sample_files = np.random.choice(csv_files, size=min(num_samples, len(csv_files)), replace=False)
            
            # matplotlibバックエンド設定（GUI環境でない場合）
            try:
                plt.figure()
                plt.close()
            except:
                import matplotlib
                matplotlib.use('Agg')
            
            fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i, file_path in enumerate(sample_files):
                try:
                    df = pd.read_csv(file_path)
                    
                    # 最初の2つのデータカラムをプロット
                    data_columns = [col for col in df.columns if col != 'timestamp']
                    
                    # 左側：最初のカラム
                    if len(data_columns) > 0:
                        axes[i, 0].plot(df[data_columns[0]], alpha=0.8)
                        axes[i, 0].set_title(f"{file_path.name} - {data_columns[0]}")
                        axes[i, 0].set_xlabel("時間インデックス")
                        axes[i, 0].set_ylabel(data_columns[0])
                        axes[i, 0].grid(True, alpha=0.3)
                    
                    # 右側：2番目のカラム
                    if len(data_columns) > 1:
                        axes[i, 1].plot(df[data_columns[1]], alpha=0.8, color='orange')
                        axes[i, 1].set_title(f"{file_path.name} - {data_columns[1]}")
                        axes[i, 1].set_xlabel("時間インデックス")
                        axes[i, 1].set_ylabel(data_columns[1])
                        axes[i, 1].grid(True, alpha=0.3)
                
                except Exception as e:
                    logger.error(f"可視化エラー: {file_path.name}, {e}")
            
            plt.tight_layout()
            
            # 保存
            plot_path = self.output_dir / "sample_data_visualization.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # 表示（可能な環境でのみ）
            try:
                plt.show()
            except:
                logger.info("GUI環境が利用できないため、プロットの表示をスキップしました")
            
            logger.info(f"可視化保存完了: {plot_path}")
            
        except Exception as e:
            logger.error(f"可視化処理エラー: {e}")


def parse_arguments():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description="時系列波形データCSVファイル生成")
    
    parser.add_argument(
        '--num_files', type=int, default=15,
        help='生成するファイル数 (デフォルト: 15)'
    )
    parser.add_argument(
        '--rows', type=int, default=7500,
        help='1ファイルあたりの行数 (デフォルト: 7500)'
    )
    parser.add_argument(
        '--columns', type=int, default=8,
        help='データカラム数 (デフォルト: 8)'
    )
    parser.add_argument(
        '--anomaly_ratio', type=float, default=0.15,
        help='異常データの割合 (デフォルト: 0.15)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./data',
        help='出力ディレクトリ (デフォルト: ./data)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='乱数シード (デフォルト: 42)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='サンプルデータの可視化を行う'
    )
    
    return parser.parse_args()


def main():
    """メイン実行関数"""
    args = parse_arguments()
    
    logger.info("時系列波形データ生成開始")
    logger.info(f"設定: ファイル数={args.num_files}, 行数={args.rows}, カラム数={args.columns}")
    logger.info(f"異常率={args.anomaly_ratio}, 出力先={args.output_dir}")
    
    # データ生成器初期化
    generator = TimeSeriesDataGenerator(
        num_files=args.num_files,
        rows_per_file=args.rows,
        num_columns=args.columns,
        anomaly_ratio=args.anomaly_ratio,
        output_dir=args.output_dir,
        random_seed=args.seed
    )
    
    try:
        # データ生成実行
        generation_summary = generator.generate_all_files()
        
        # 結果表示
        print("\n" + "=" * 60)
        print("データ生成完了!")
        print("=" * 60)
        print(f"生成ファイル数: {generation_summary['total_files']}")
        print(f"正常ファイル: {generation_summary['normal_files']}")
        print(f"異常ファイル: {generation_summary['anomaly_files']}")
        print(f"出力ディレクトリ: {args.output_dir}")
        print("=" * 60)
        
        # 可視化
        if args.visualize:
            logger.info("サンプルデータ可視化実行")
            generator.visualize_sample_data(num_samples=3)
        
        # 使用例表示
        print(f"\n次のステップ:")
        print(f"1. main.pyでの分析実行:")
        print(f"   python main.py")
        print(f"2. Streamlitでの可視化:")
        print(f"   streamlit run visualization.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"データ生成エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


def demo_generation():
    """デモ用の簡単な生成例"""
    print("=" * 50)
    print("デモ: 小規模データセット生成")
    print("=" * 50)
    
    try:
        generator = TimeSeriesDataGenerator(
            num_files=5,
            rows_per_file=1000,
            num_columns=4,
            anomaly_ratio=0.2,
            output_dir="./demo_data",
            random_seed=123
        )
        
        print("データ生成を開始します...")
        summary = generator.generate_all_files()
        
        print("\nデータ生成完了!")
        print(f"生成ファイル数: {summary['total_files']}")
        print(f"正常ファイル数: {summary['normal_files']}")
        print(f"異常ファイル数: {summary['anomaly_files']}")
        print(f"出力ディレクトリ: ./demo_data")
        
        # ファイル一覧表示
        print("\n生成されたファイル:")
        for file_detail in summary['file_details']:
            status = "異常" if file_detail['is_anomaly'] else "正常"
            print(f"  {file_detail['filename']} ({status})")
        
        # 可視化実行
        print("\nサンプルデータの可視化を実行...")
        generator.visualize_sample_data(num_samples=2)
        
        return summary
        
    except Exception as e:
        print(f"デモ実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys
    
    print("時系列波形データ生成スクリプト")
    print("使用方法:")
    print("1. 通常実行: python create_csv.py [オプション]")
    print("2. デモ実行: 引数なしで実行")
    print()
    
    try:
        # コマンドライン引数がある場合は通常実行
        if len(sys.argv) > 1:
            print("コマンドライン引数が検出されました。通常実行モードで開始...")
            exit_code = main()
            sys.exit(exit_code)
        else:
            # 引数がない場合はデモ実行
            print("引数が指定されていません。デモ実行モードで開始...")
            result = demo_generation()
            if result is not None:
                print("\n✅ デモ実行完了!")
            else:
                print("\n❌ デモ実行に失敗しました")
            
            print("\n実際の使用例:")
            print("python create_csv.py --num_files 15 --rows 7500 --columns 8 --anomaly_ratio 0.15 --output_dir ./data/ --visualize")
    
    except KeyboardInterrupt:
        print("\n実行が中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"\n予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)