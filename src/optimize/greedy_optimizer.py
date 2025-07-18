import time
from typing import List, Optional, Tuple

import numpy as np

from src.dataclass import OptimizeInputData, OptimizeOutputData, SolutionData
from src.estimate.reach_estimator import ProgramUniqueReachEstimator
from src.optimize.model_base import SolverBase


class GreedyOptimizer(SolverBase):
    """貪欲法による番組選択最適化クラス"""

    def __init__(self, optimize_input_data: OptimizeInputData):
        """
        Args:
            optimize_input_data: 最適化問題の入力データ
        """
        self.input_data = optimize_input_data
        self.solution: Optional[SolutionData] = None
        self.output_data: Optional[OptimizeOutputData] = None

        # estimatorを作成
        if optimize_input_data.individual_reach_probs is not None:
            self.estimator = ProgramUniqueReachEstimator(
                optimize_input_data.individual_reach_probs
            )
        else:
            raise ValueError("individual_reach_probs が指定されていません")

    def solve(
        self, input_data: Optional[OptimizeInputData] = None
    ) -> OptimizeOutputData:
        """
        最適化問題を解く

        Args:
            input_data: 最適化問題の入力データ（Noneの場合は初期化時のデータを使用）

        Returns:
            OptimizeOutputData: 最適化結果
        """
        start_time = time.time()

        # input_dataが指定された場合は更新
        if input_data is not None:
            self.input_data = input_data
            # estimatorも更新
            if input_data.individual_reach_probs is not None:
                self.estimator = ProgramUniqueReachEstimator(
                    input_data.individual_reach_probs
                )

        # 貪欲法で最適解を探索
        objective_value, selected_programs = self._greedy_search(
            budget=self.input_data.budget,
            costs=self.input_data.costs,
        )

        # 解の作成
        selection_matrix = np.zeros((1, self.input_data.program_count))
        if selected_programs:
            # プログラム番号は1-indexedなので、0-indexedに変換
            selection_matrix[0, np.array(selected_programs) - 1] = 1

        total_cost = float(self.input_data.costs @ selection_matrix.flatten())

        self.solution = SolutionData(
            selected_programs=selected_programs,
            selection_matrix=selection_matrix,
            total_cost=total_cost,
        )

        computation_time = time.time() - start_time

        self.output_data = OptimizeOutputData(
            solution=self.solution,
            objective_value=objective_value,
            computation_time=computation_time,
            method="greedy_search",
            metadata={
                "budget": self.input_data.budget,
                "program_count": self.input_data.program_count,
                "selected_program_count": len(selected_programs),
            },
        )

        return self.output_data

    def get_solution(self) -> SolutionData:
        """
        解を取得する

        Returns:
            SolutionData: 最適化の解
        """
        if self.solution is None:
            raise ValueError("まず solve() を実行してください")
        return self.solution

    def _greedy_search(
        self, budget: float, costs: np.ndarray
    ) -> Tuple[float, List[int]]:
        """
        予算内で最大のユニークリーチを達成する番組群を貪欲法で探索する

        Args:
            budget: 予算制約
            costs: 各番組のコスト

        Returns:
            Tuple[float, List[int]]: 最大ユニークリーチと選択された番組のリスト
        """
        # 番組候補の数
        program_count = costs.shape[0]
        # 選択した番組群。最初は何も選択していない。これを更新していく
        selection_matrix = np.zeros((1, program_count))
        # 選択した番組のリスト
        selected_programs: List[int] = []

        # 予算内を超えない限り番組を追加していく
        while True:
            # 番組追加前の合算ユニークリーチ
            reach_before = self.estimator.predict(selection_matrix)[0]

            # 番組を追加した場合の合算のユニークリーチの増加幅を計算
            delta = np.zeros(program_count)  # 差分
            for j in range(program_count):
                # すでに選択されている番組はスキップ
                if selection_matrix[0, j] == 1:
                    continue

                # 番組j+1を追加
                # インデックスが0始まりなので、正確にはjではなくj+1が追加される
                temp_selection = selection_matrix.copy()
                temp_selection[0, j] = 1

                # 予算内に収まらない場合はスキップ
                if costs @ temp_selection.flatten() > budget:
                    continue

                # 番組j+1を追加した場合の合算のユニークリーチの増加幅を計算
                reach_after = self.estimator.predict(temp_selection)[0]

                delta[j] = (reach_after - reach_before) / costs[j]

            # すべてのdeltaが0の場合（追加できる番組がない場合）はループを終了
            if np.all(delta == 0):
                break

            # コストパフォーマンスが最も高い番組を追加
            j_to_add = int(np.argmax(delta))
            selection_matrix[0, j_to_add] = 1

            # 予算内に収まらない場合は番組の追加を取りやめて終了
            if costs @ selection_matrix.flatten() > budget:
                selection_matrix[0, j_to_add] = 0
                break

            # インデックスは0始まりなので、1を足して追加
            selected_programs.append(j_to_add + 1)

        # 最終的な合算のユニークリーチと、それを達成する番組群を返す
        final_reach = self.estimator.predict(selection_matrix)[0]
        return final_reach, selected_programs
