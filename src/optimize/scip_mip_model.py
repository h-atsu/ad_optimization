import time
from typing import Optional

import numpy as np
from pyscipopt import Model, quicksum

from src.dataclass import OptimizeInputData, OptimizeOutputData, SolutionData
from src.optimize.model_base import SolverBase


class ScipMipModel(SolverBase):
    """SCIP MIPによる番組選択最適化クラス"""

    def __init__(self, optimize_input_data: OptimizeInputData):
        """
        Args:
            optimize_input_data: 最適化問題の入力データ
        """
        self.input_data = optimize_input_data
        self.model = Model()
        self.d = {}  # 決定変数
        self.e = {}  # 補助変数
        self.solution: Optional[SolutionData] = None

    def _add_variables(self):
        """
        変数を追加する
        """

        for j in self.input_data.list_program_index:
            self.d[j] = self.model.addVar(vtype="BINARY", name=f"d_{j}")

        for i in self.input_data.list_indivisual_index:
            for k in [0] + self.input_data.list_program_index:
                self.e[i, k] = self.model.addVar(
                    vtype="CONTINUOUS", lb=0, name=f"e_{i}_{k}"
                )

        return self

    def _add_constraints(self):
        """
        制約を追加する
        """

        # 予算制約
        self.model.addCons(
            quicksum(
                self.d[j] * self.input_data.costs[j - 1]
                for j in self.input_data.list_program_index
            )
            <= self.input_data.budget
        )

        # 補助変数の漸化式制約
        for i in self.input_data.list_indivisual_index:
            self.model.addCons(self.e[i, 0] == 1)

            for k in self.input_data.list_program_index:
                self.model.addCons(-self.d[k] <= self.e[i, k] - self.e[i, k - 1])
                self.model.addCons(self.e[i, k] - self.e[i, k - 1] <= self.d[k])

                self.model.addCons(
                    -(1 - self.d[k])
                    <= self.e[i, k]
                    - self.e[i, k - 1]
                    * (1 - self.input_data.individual_reach_probs[i - 1, k - 1])
                )
                self.model.addCons(
                    self.e[i, k]
                    - self.e[i, k - 1]
                    * (1 - self.input_data.individual_reach_probs[i - 1, k - 1])
                    <= (1 - self.d[k])
                )

        return self

    def _add_objective(self):
        """
        目的関数を追加する
        """
        self.model.setObjective(
            1
            - quicksum(
                self.e[i, self.input_data.program_count]
                for i in self.input_data.list_indivisual_index
            )
            / self.input_data.individual_count,
            sense="maximize",
        )
        return self

    def solve(self) -> OptimizeOutputData:
        """
        最適化問題を解く
        """
        start_time = time.time()

        self._add_variables()._add_constraints()._add_objective()

        # self.model.setRealParam("limits/time", 600)
        # self.model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
        self.model.optimize()

        computation_time = time.time() - start_time

        # ソルバーの結果から解を取得
        best_solution = self.model.getBestSol()

        if best_solution is not None:
            # 選択された番組を取得
            selected_programs = []
            for j in self.input_data.list_program_index:
                if (
                    self.model.getSolVal(best_solution, self.d[j]) > 0.5
                ):  # バイナリ変数なので0.5でしきい値
                    selected_programs.append(j)

            # 目的関数値を取得
            objective_value = self.model.getSolObjVal(best_solution)

            # 解の作成
            selection_matrix = np.zeros((1, self.input_data.program_count))
            if selected_programs:
                # プログラム番号は1-indexedなので、0-indexedに変換
                program_indices = [
                    j - 1 for j in selected_programs
                ]  # 1-indexedから0-indexedに変換
                selection_matrix[0, program_indices] = 1

            # 総コストを計算
            total_cost = float(self.input_data.costs @ selection_matrix.flatten())
        else:
            # 解が見つからなかった場合
            selected_programs = []
            objective_value = 0.0
            selection_matrix = np.zeros((1, self.input_data.program_count))
            total_cost = 0.0

        self.solution = SolutionData(
            selected_programs=selected_programs,
            selection_matrix=selection_matrix,
            total_cost=total_cost,
        )

        return OptimizeOutputData(
            solution=self.solution,
            objective_value=objective_value,
            computation_time=computation_time,
            method="scip_mip",
            metadata={
                "budget": self.input_data.budget,
                "program_count": self.input_data.program_count,
                "selected_program_count": len(selected_programs),
            },
        )

    def get_solution(self) -> SolutionData:
        """
        解を取得する

        Returns:
            SolutionData: 最適化の解
        """
        if self.solution is None:
            raise ValueError("まず solve() を実行してください")
        return self.solution
