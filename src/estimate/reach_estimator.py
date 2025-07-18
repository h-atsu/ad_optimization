import numpy as np
import polars as pl

from .estimator_base import EstimatorBase


class ProgramUniqueReachEstimator(EstimatorBase):
    """選択した番組群に対するユニークリーチを予測するモデル"""

    def __init__(self, individual_reach_probs: np.ndarray) -> None:
        """
        Args:
            individual_reach_probs (np.ndarray): 個人ごとの各番組のCMへの接触確率
        """
        super().__init__(individual_reach_probs)

    def fit(self) -> None:
        """学習処理（このモデルでは特に行わない）"""
        pass

    def predict(self, selection_matrix: np.ndarray) -> np.ndarray:
        """
        選択した番組群に対するユニークリーチを予測

        Args:
            selection_matrix (np.ndarray): ダミー変数形式の選択した番組群

        Returns:
            np.ndarray: 合算のユニークリーチの予測値
        """
        # (1 - PI)のD乗を計算したいので、shapeをあわせる
        # (n, m)を(n, m, 1)に
        pi_ = self.individual_reach_probs_[:, :, np.newaxis]
        # (l, m)を(1, m, l)に
        d_ = selection_matrix.T[np.newaxis, :, :]

        # これで、(1 - pi_) ** d_のshapeが(n, m, l)になる
        reach_pred = 1 - ((1 - pi_) ** d_).prod(1).mean(0)

        return reach_pred

    def decompose_contribution(self, selection_matrix: np.ndarray) -> pl.DataFrame:
        """
        ユニークリーチの貢献度を分解する

        Args:
            selection_matrix (np.ndarray): 選択した番組群。1行m列を想定

        Returns:
            pl.DataFrame: 貢献度の分解結果
        """
        # 選択した番組群のインデックス
        j_indices = np.where(selection_matrix)[1]
        # 選択した番組群の接触確率だけにデータを限定
        pi = self.individual_reach_probs_[:, j_indices]

        # 番組単体のユニークリーチ
        r_single = pi.mean(0)
        # 選択した番組群の合算のユニークリーチ
        r_full = self.predict(selection_matrix)[0]  # floatにするため[0]で取り出す

        ### 貢献度1: 番組単体のユニークリーチによる貢献度 ###
        phi_single = r_single

        ### 貢献度2: 番組固有のユニークリーチによる貢献度 ###
        # 番組を一つ外した場合に合算のユニークリーチがどれだけ減少するかで測定
        phi_loo = np.zeros(len(j_indices))
        for i, j_index in enumerate(j_indices):
            d_ = selection_matrix.copy()
            d_[0, j_index] = 0
            phi_loo[i] = r_full - self.predict(d_)[0]

        ### 貢献度3: Shapley値による貢献度（1階近似） ###
        # 1 - Π_j (1 - π_ij)を計算
        # PIを割る際にshapeが(n, 1)になるようにkeepdims=Trueを指定
        r = 1 - (1 - pi).prod(1, keepdims=True)

        # π_ij / Σ_j π_ij を計算
        sum_pi = pi.sum(1, keepdims=True)
        pi_divided_sum_pi = np.nan_to_num(
            pi / np.where(sum_pi == 0, np.nan, sum_pi),
            nan=0,
        )  # 分母が0の場合は0に置換

        # 2つのパートをかけ合わせて平均をとることで貢献度を計算
        phi_shapley = (pi_divided_sum_pi * r).mean(0)

        # 貢献度をDataFrameにまとめて出力
        return pl.DataFrame(
            {
                "j": j_indices + 1,  # インデックスは0始まりなので1を足す
                "single": phi_single,
                "loo": phi_loo,
                "shapley": phi_shapley,
            }
        )
