import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import betabinom, binom

from .estimator_base import EstimatorBase
from .reach_estimator import ProgramUniqueReachEstimator


class BinomialFrequencyEstimator(EstimatorBase):
    """二項分布を仮定したフリークエンシーの分布を推定するモデル"""

    def __init__(self, individual_reach_probs: np.ndarray) -> None:
        """
        Args:
            individual_reach_probs (np.ndarray): 個人iが番組jのCMに接触する確率
        """
        super().__init__(individual_reach_probs)
        self.pi_ = None

    def fit(self) -> None:
        """学習処理（特になし）"""
        pass

    def predict(self, selection_matrix: np.ndarray) -> np.ndarray:
        """
        フリークエンシーの分布を予測

        Args:
            selection_matrix (np.ndarray): 選択した番組群。1行m列を想定

        Returns:
            np.ndarray: フリークエンシーの分布の予測値
        """
        # 選択した番組数
        sum_d = selection_matrix.sum()

        # CM接触確率の全体平均。学習結果は`pi_`に格納
        self.pi_ = np.mean(
            (self.individual_reach_probs_ * selection_matrix).sum(1) / sum_d
        )

        # 二項分布を用いてフリークエンシーの分布を予測
        return binom.pmf(k=np.arange(sum_d + 1), n=sum_d, p=self.pi_)


class BetaBinomialFrequencyEstimator(ProgramUniqueReachEstimator):
    """ベータ二項分布でフリークエンシーの分布を推定するモデル"""

    def __init__(self, individual_reach_probs: np.ndarray) -> None:
        """
        Args:
            individual_reach_probs (np.ndarray): 個人ごとの各番組のCMへの接触確率
        """
        super().__init__(individual_reach_probs)
        self.mu_ = None
        self.nu_ = None

    def fit(self) -> None:
        """学習処理（特になし）"""
        pass

    def predict_frequency_distribution(
        self, selection_matrix: np.ndarray
    ) -> np.ndarray:
        """
        フリークエンシーの分布を予測

        Args:
            selection_matrix (np.ndarray): 選択した番組群。1行m列を想定

        Returns:
            np.ndarray: フリークエンシーの分布の予測値
        """
        sum_d = int(selection_matrix.sum())
        r_ = float(self.predict(selection_matrix)[0])
        self.mu_ = float(
            np.mean((self.individual_reach_probs_ * selection_matrix).sum(1) / sum_d)
        )

        # ベータ二項分布のパラメータnuを推定
        self.nu_ = minimize_scalar(
            fun=lambda nu: (
                (1 - r_)
                - betabinom.pmf(
                    k=0,
                    n=sum_d,
                    a=self.mu_ * nu,
                    b=(1 - self.mu_) * nu,
                )
            )
            ** 2,
            bounds=(0, 50),
            method="bounded",
        ).x

        return betabinom.pmf(
            k=np.arange(sum_d + 1),
            n=sum_d,
            a=self.mu_ * self.nu_,
            b=(1 - self.mu_) * self.nu_,
        )
