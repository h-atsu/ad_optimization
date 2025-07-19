import os
import time
from typing import Optional

import gurobipy as gp
import numpy as np
from dotenv import load_dotenv
from gurobipy import GRB

from src.dataclass import OptimizeInputData, OptimizeOutputData, SolutionData
from src.optimize.model_base import SolverBase


class GurobiModel(SolverBase):
    """Gurobiによる番組選択最適化クラス"""

    def __init__(self, optimize_input_data: OptimizeInputData):
        """
        Args:
            optimize_input_data: 最適化問題の入力データ
        """
        self.input_data = optimize_input_data

        # 環境変数の読み込み
        load_dotenv()

        # Gurobiライセンスオプションの設定
        options = {}
        wls_access_id = os.getenv("GUROBI_WLSACCESSID")
        if wls_access_id:
            options["WLSACCESSID"] = wls_access_id

        wls_secret = os.getenv("GUROBI_WLSSECRET")
        if wls_secret:
            options["WLSSECRET"] = wls_secret

        license_id = os.getenv("GUROBI_LICENSEID")
        if license_id:
            options["LICENSEID"] = int(license_id)

        # 環境とモデルの作成
        self.env = gp.Env(params=options)
        self.model = gp.Model(env=self.env)
        self.d = {}  # 決定変数
        self.e = {}  # 補助変数
        self.solution: Optional[SolutionData] = None

    def _add_variables(self):
        """
        変数を追加する
        """

        for j in self.input_data.list_program_index:
            self.d[j] = self.model.addVar(vtype=GRB.BINARY, name=f"d_{j}")

        for i in self.input_data.list_indivisual_index:
            for k in [0] + self.input_data.list_program_index:
                self.e[i, k] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, name=f"e_{i}_{k}"
                )

        return self

    def _add_constraints(self):
        """
        制約を追加する
        """

        # 予算制約
        self.model.addConstr(
            gp.quicksum(
                self.d[j] * self.input_data.costs[j - 1]
                for j in self.input_data.list_program_index
            )
            <= self.input_data.budget,
            name="budget_constraint",
        )

        # 補助変数の漸化式制約
        for i in self.input_data.list_indivisual_index:
            self.model.addConstr(self.e[i, 0] == 1, name=f"initial_constraint_{i}")

            for k in self.input_data.list_program_index:
                self.model.addConstr(
                    -self.d[k] <= self.e[i, k] - self.e[i, k - 1],
                    name=f"constraint_1_{i}_{k}",
                )
                self.model.addConstr(
                    self.e[i, k] - self.e[i, k - 1] <= self.d[k],
                    name=f"constraint_2_{i}_{k}",
                )

                self.model.addConstr(
                    -(1 - self.d[k])
                    <= self.e[i, k]
                    - self.e[i, k - 1]
                    * (1 - self.input_data.individual_reach_probs[i - 1, k - 1]),
                    name=f"constraint_3_{i}_{k}",
                )
                self.model.addConstr(
                    self.e[i, k]
                    - self.e[i, k - 1]
                    * (1 - self.input_data.individual_reach_probs[i - 1, k - 1])
                    <= (1 - self.d[k]),
                    name=f"constraint_4_{i}_{k}",
                )

        return self

    def _add_objective(self):
        """
        目的関数を追加する
        """
        self.model.setObjective(
            1
            - gp.quicksum(
                self.e[i, self.input_data.program_count]
                for i in self.input_data.list_indivisual_index
            )
            / self.input_data.individual_count,
            GRB.MAXIMIZE,
        )
        return self

    def solve(self) -> OptimizeOutputData:
        """
        最適化問題を解く
        """
        start_time = time.time()

        self._add_variables()._add_constraints()._add_objective()

        # 時間制限を設定（オプション）
        # self.model.setParam('TimeLimit', 600)

        # 最適化実行
        self.model.optimize()

        computation_time = time.time() - start_time

        if self.model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            # 選択された番組を取得
            selected_programs = []
            for j in self.input_data.list_program_index:
                if self.d[j].X > 0.5:  # バイナリ変数なので0.5でしきい値
                    selected_programs.append(j)

            # 目的関数値を取得
            objective_value = self.model.ObjVal

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

        # 環境をクリーンアップ
        self.model.dispose()
        self.env.dispose()

        return OptimizeOutputData(
            solution=self.solution,
            objective_value=objective_value,
            computation_time=computation_time,
            method="gurobi",
            metadata={
                "budget": self.input_data.budget,
                "program_count": self.input_data.program_count,
                "selected_program_count": len(selected_programs),
                "solver_status": self.model.Status,
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
