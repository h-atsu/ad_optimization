from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel


class OptimizeInputData(BaseModel):
    """最適化問題の入力データを格納するクラス"""

    budget: float  # 予算またはリソース制約
    costs: np.ndarray  # 各番組のコスト
    program_count: int  # 番組数
    individual_count: int  # 個人数
    list_indivisual_index: List[int]  # 個人のインデックス e.g. [1, 2, 3 , 4, 5]
    list_program_index: List[int]  # 番組のインデックス e.g. [1, 2, 3, 4, 5]
    individual_reach_probs: np.ndarray  # 個人ごとの各番組への接触確率

    class Config:
        arbitrary_types_allowed = True


class SolutionData(BaseModel):
    """最適化問題の解を格納するクラス"""

    selected_programs: List[int]  # 選択された番組のリスト
    selection_matrix: np.ndarray  # 選択された番組のダミー変数行列
    total_cost: float  # 総コスト

    class Config:
        arbitrary_types_allowed = True


class OptimizeOutputData(BaseModel):
    """最適化結果の評価値を格納するクラス"""

    solution: SolutionData
    objective_value: float  # 目的関数値（ユニークリーチ）
    computation_time: float  # 計算時間
    method: str  # 使用した最適化手法
    metadata: Dict[str, Any]  # その他の情報
