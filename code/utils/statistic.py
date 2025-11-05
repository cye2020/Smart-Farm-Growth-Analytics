# 표준 라이브러리
from abc import ABC, abstractmethod
from dataclasses import dataclass

# 서드파티 라이브러리
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

# 통계 분석
from scipy import stats
from scipy.stats import shapiro, ttest_ind  
from scipy.stats import mannwhitneyu


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_interpretation: str
    conclusion: str
    metadata: dict

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha
    
    def to_dict(self) -> dict:
        return self.__dict__


class StatisticalTest(ABC):
    """
    모델 통계 검정의 기반 추상 클래스
    """
    
    @abstractmethod
    def execute(self, data: pd.DataFrame, iv_col: str, dp_col: list, **kwargs):
        pass
    
    @abstractmethod
    def interpret(self, result):
        pass


class TTest(StatisticalTest):
    """
    t-검정 (독립/대응)
    """
    
    def __init__(self):
        pass
    
    def execute(self, data: pd.DataFrame, iv_col: str, dv_col: str, alpha: float = 0.5, equal_var=True):
        """
        두 그룹 간 평균 차이에 대한 가설검정을 수행하는 함수.
        (정규성에 따라 t-검정 또는 Mann-Whitney U 검정을 자동 선택)

        Parameters
        ----------
        data: 원본 데이터
        iv_col : array-like
            독립 변수 컬럼명
        dv_col : array-like
            종속 변수 컬럼명
        is_normal_iv : bool
            그룹 0의 정규성 충족 여부
        is_normal_dv : bool
            그룹 1의 정규성 충족 여부
        equal_var : bool, optional
            두 그룹의 분산이 같은지 여부 (default=True)
        alpha : float, optional
            유의수준 (default=0.05)

        Returns
        -------
        result : dict
            {
                'test_name': str,
                'statistic': float,
                'p_value': float,
                'effect_size': float or None,
                'effect_interpretation': str or None,
                'conclusion': str
            }
        """
        
        iv_data = data[iv_col]
        dv_data = data[dv_col]
        
        is_normal_iv = self.check_normality_simple(iv_data, iv_col)
        is_normal_dv = self.check_normality_simple(dv_data, dv_col)
        
        print("\n[가설검정]")
        print("-" * 40)

        if is_normal_iv and is_normal_dv:
            print("H₀: μ₀ = μ₁ (두 클래스의 평균이 같다)")
            print("H₁: μ₀ ≠ μ₁ (두 클래스의 평균이 다르다)")
        else:
            print("H₀: 두 클래스의 분포가 같다 (중앙값 차이가 없다)")
            print("H₁: 두 클래스의 분포가 다르다 (중앙값 차이가 있다)")
            
        print(f"유의수준: α = {alpha}\n")

        # --- 검정 수행 ---
        if is_normal_iv and is_normal_dv:
            # 모수 검정
            test_name = "Student's t-test" if equal_var else "Welch's t-test"
            t_stat, p_value = ttest_ind(iv_data, dv_data, equal_var=equal_var)
            print(f"{test_name} 결과:")
            print(f"t = {t_stat:.4f}, p = {p_value:.4f}")
            
            # Cohen's d 계산
            pooled_std = np.sqrt((iv_data.var() + dv_data.var()) / 2)
            cohens_d = (iv_data.mean() - dv_data.mean()) / pooled_std
            abs_d = abs(cohens_d)

            if abs_d < 0.2:
                effect = "매우 작은 효과"
            elif abs_d < 0.5:
                effect = "작은 효과"
            elif abs_d < 0.8:
                effect = "중간 효과"
            else:
                effect = "큰 효과"

            print(f"Cohen's d = {cohens_d:.3f} ({effect})")

            test_stat = t_stat
            effect_size = cohens_d
            effect_interpretation = effect

        else:
            # 비모수 검정
            test_name = "Mann-Whitney U test"
            u_stat, p_value = mannwhitneyu(iv_data, dv_data, alternative='two-sided')
            print(f"{test_name} 결과:")
            print(f"U = {u_stat:.4f}, p = {p_value:.4f}")

            test_stat = u_stat
            effect_size = None
            effect_interpretation = None

        # --- 결론 ---
        print("\n[결론]")
        if p_value < alpha:
            conclusion = f"✅ p-value({p_value:.4f}) < {alpha} → 귀무가설 기각\n   두 클래스에 유의한 차이가 있음"
        else:
            conclusion = f"❌ p-value({p_value:.4f}) ≥ {alpha} → 귀무가설 채택\n   두 클래스에 유의한 차이가 없음"

        print(conclusion)
        
        # 결과 반환
        return TestResult(
            test_name=test_name,
            statistic=test_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_interpretation=effect_interpretation,
            conclusion=conclusion,
            metadata=None
        )
    
    def interpret(self, result: TestResult, alpha: float = 0.05):
        pass

    def check_normality_simple(self, data: ArrayLike, col="데이터"):
        """
        데이터의 정규성을 검정하는 함수: t-test용

        Parameters
        ----------
        data : array-like
            정규성을 검정할 데이터 (NaN은 자동 제거)
        name : str, default="데이터"
            출력 시 표시될 데이터 이름

        Returns
        -------
        bool
            정규분포 가정 충족 여부
            - True: 정규분포 가정 가능 (모수 검정)
            - False: 정규분포 가정 위반 (비모수 검정)

        검정 기준
        ---------
        - n < 30: Shapiro-Wilk 검정 (p > 0.05)
        - 30 ≤ n < 100: 왜도/첨도 우선, 필요시 Shapiro-Wilk
        - n ≥ 100: 왜도 기준 (|왜도| < 2, 중심극한정리)
        """
        # NaN 체크
        if pd.isna(data).any():
            print(f"⚠️ 경고: {col}에 NaN 값이 {pd.isna(data).sum()}개 포함됨")
            data = data.dropna()
            print(f"   → NaN 제거 후 n={len(data)}")

        n = len(data)

        print(f"\n[{col} 정규성 검정] n={n}")
        print("-"*40)

        # 왜도와 첨도
        skew = stats.skew(data)
        kurt = stats.kurtosis(data, fisher=True)
        print(f"왜도(Skewness): {skew:.3f}")
        print(f"첨도(Kurtosis): {kurt:.3f}")

        # 표본 크기에 따른 판단
        if n < 30:
            stat, p = shapiro(data)
            print(f"Shapiro-Wilk p-value: {p:.4f}")
            is_normal = p > 0.05
            reason = f"Shapiro p={'>' if is_normal else '≤'}0.05"
        elif n < 100:
            if abs(skew) < 1 and abs(kurt) < 2:
                is_normal = True
                reason = "|왜도|<1, |첨도|<2"
            else:
                stat, p = shapiro(data)
                print(f"추가 Shapiro-Wilk p-value: {p:.4f}")
                is_normal = p > 0.05
                reason = f"Shapiro p={'>' if is_normal else '≤'}0.05"
        else:
            is_normal = abs(skew) < 2
            reason = f"|왜도|{'<' if is_normal else '≥'}2 (중심극한정리)"

        print(f"결과: {'✅ 정규분포 가정 충족' if is_normal else '❌ 정규분포 가정 위반'} ({reason})")
        return is_normal