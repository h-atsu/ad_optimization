from typing import Literal

import numpy as np
import polars as pl

from src.consts import ROOT
from src.dataclass import OptimizeInputData
from src.estimate.reach_estimator import ProgramUniqueReachEstimator
from src.optimize.greedy_optimizer import GreedyOptimizer
from src.optimize.scip_mip_model import ScipMipModel


def load_data():
    """データを読み込む"""
    # データファイルのパス
    data_dir = ROOT / "data"
    company_reach_path = data_dir / "company_reach.parquet"
    individual_reach_path = data_dir / "individual_reach.parquet"

    # データの読み込み
    df_company_reach = pl.read_parquet(company_reach_path)
    df_individual_reach = pl.read_parquet(individual_reach_path)

    return df_company_reach, df_individual_reach


def prepare_reach_estimator(
    df_individual_reach: pl.DataFrame,
) -> ProgramUniqueReachEstimator:
    """リーチ推定モデルを準備する"""
    # 学習用データを抽出
    df_individual_reach_train = df_individual_reach.filter(pl.col.is_train == 1)

    # 個人ごとの各番組のCM接触確率を計算して横持ちで保存
    df_reach_prob_train = (
        # 個人ごとの各番組のCM接触確率を計算
        df_individual_reach_train.group_by("i", "j", maintain_order=True)
        .agg(pi=pl.col.r.sum() / pl.col.k.n_unique())
        # 横持ちに変換
        .pivot(on="j", index="i", values="pi")
    )

    # numpyのndarrayに変換
    PI_train = (
        df_reach_prob_train
        # 個人のインデックスが列として入っているので削除
        .drop("i")
        # n行m列の行列に変換
        .to_numpy()
    )

    # モデルの作成
    estimator = ProgramUniqueReachEstimator(PI_train)
    estimator.fit()

    return estimator


def create_optimizer(
    input_data: OptimizeInputData, solver_type: Literal["greedy", "scip_mip"] = "greedy"
):
    """
    指定されたソルバータイプに応じて最適化器を作成する

    Args:
        input_data: 最適化問題の入力データ
        solver_type: ソルバーの種類 ("greedy" または "scip_mip")

    Returns:
        最適化器のインスタンス
    """
    if solver_type == "greedy":
        return GreedyOptimizer(input_data)
    elif solver_type == "scip_mip":
        return ScipMipModel(input_data)
    else:
        raise ValueError(f"未対応のソルバータイプ: {solver_type}")


def main(solver_type: Literal["greedy", "scip_mip"] = "greedy"):
    """
    最適化の実行例

    Args:
        solver_type: 使用するソルバーの種類
    """
    print("=== 広告最適化システム実行例 ===")
    print(f"使用ソルバー: {solver_type}")

    # データの読み込み
    print("1. データを読み込み中...")
    df_company_reach, df_individual_reach = load_data()
    print(f"   企業データ: {df_company_reach.shape[0]} 行")
    print(f"   個人データ: {df_individual_reach.shape[0]} 行")

    # リーチ推定モデルの準備
    print("2. リーチ推定モデルを準備中...")
    estimator = prepare_reach_estimator(df_individual_reach)
    print(f"   番組数: {estimator.program_count_}")

    # 最適化パラメータの設定
    program_count = estimator.program_count_
    budget = 5000.0  # 予算
    costs = np.ones(program_count) * 500  # 各番組のコスト

    # 個人と番組のインデックスリストを作成
    individual_count = estimator.individual_reach_probs_.shape[0]
    list_individual_index = list(range(1, individual_count + 1))  # 1-indexed
    list_program_index = list(range(1, program_count + 1))  # 1-indexed

    # 入力データの作成
    input_data = OptimizeInputData(
        budget=budget,
        costs=costs,
        program_count=program_count,
        individual_count=individual_count,
        list_indivisual_index=list_individual_index,
        list_program_index=list_program_index,
        individual_reach_probs=estimator.individual_reach_probs_,
    )

    # 最適化器の作成と実行
    print("3. 最適化を実行中...")
    optimizer = create_optimizer(input_data, solver_type)
    output_data = optimizer.solve()

    # 結果の表示
    print("4. 結果:")
    print(f"   目的関数値（ユニークリーチ）: {output_data.objective_value:.3f}")
    print(f"   計算時間: {output_data.computation_time:.3f} 秒")
    print(f"   選択された番組数: {len(output_data.solution.selected_programs)}")
    print(f"   選択された番組: {output_data.solution.selected_programs}")
    print(f"   総コスト: {output_data.solution.total_cost:.1f}")
    print(f"   使用手法: {output_data.method}")

    return output_data


if __name__ == "__main__":
    # 乱数を固定
    np.random.seed(42)

    # 単一ソルバーの実行例（デフォルトは貪欲法）
    print("=== 単一ソルバー実行 ===")
    output_data = main("scip_mip")  # "greedy" または "scip_mip" を指定

    # 複数ソルバーの比較（オプション）
    # compare_solvers()

    print("\n=== 実行完了 ===")
