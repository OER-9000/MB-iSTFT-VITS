import pandas as pd
import argparse
import sys

def analyze_benchmark_results(filepath):
    """
    ベンチマーク結果のCSVファイルを読み込み、レイテンシの統計情報を計算する。
    """
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {filepath}", file=sys.stderr)
        sys.exit(1)

    # 'execution_time_s'列を数値に変換し、変換できない値はNaNにする
    # これにより、フッターの文字列などを除外できる
    df['execution_time_s'] = pd.to_numeric(df['execution_time_s'], errors='coerce')

    # 'execution_time_s'がNaNの行（フッターなど）を削除
    df.dropna(subset=['execution_time_s'], inplace=True)

    if df.empty:
        print("有効なデータが見つかりませんでした。CSVファイルが空か、フォーマットが正しくない可能性があります。")
        return

    # 'method'列でグループ化
    grouped = df.groupby('method')

    # 平均と標準偏差を計算
    # pandasのstd()はデフォルトでサンプル標準偏差(ddof=1)を計算する
    latency_stats = grouped['execution_time_s'].agg(['mean', 'std'])

    # 結果を表示
    print("--- Benchmark Latency Analysis (execution_time_s) ---")
    print(f"Source file: {filepath}")
    print("\nLatency (seconds):")
    print(latency_stats)
    print("\n'mean'は平均値、'std'はサンプル標準偏差を示します。")
    print("----------------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="ベンチマーク結果CSVを分析するスクリプト")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="benchmark_results.csv",
        help="分析対象のCSVファイルのパス"
    )
    args = parser.parse_args()
    
    analyze_benchmark_results(args.csv_path)

if __name__ == "__main__":
    main()