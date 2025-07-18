from abc import ABC, abstractmethod

import numpy as np


class EstimatorBase(ABC):
    """予測モデルの基底クラス"""

    def __init__(self, individual_reach_probs: np.ndarray) -> None:
        """
        Args:
            individual_reach_probs (np.ndarray): 個人ごとの各番組のCMへの接触確率
        """
        self.individual_reach_probs_ = individual_reach_probs
        self.program_count_ = individual_reach_probs.shape[1]

    @abstractmethod
    def predict(self, selection_matrix: np.ndarray) -> np.ndarray:
        """
        選択した番組群に対する予測を行う

        Args:
            selection_matrix (np.ndarray): ダミー変数形式の選択した番組群

        Returns:
            np.ndarray: 予測値
        """
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """
        モデルの学習を行う
        """
        pass
