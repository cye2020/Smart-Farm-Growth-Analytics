# =========================================================
# 통계 검정 클래스
# ========================================================="


# 1. 표준 라이브러리
from abc import ABC, abstractmethod
from dataclasses import dataclass

# 2. 서드파티 라이브러리
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

# 2-1. 시각화 라이브러이
import matplotlib.pyplot as plt

# 2-2. 통계 분석
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind  
from scipy.stats import mannwhitneyu, norm


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
    def execute(self):
        pass
    
    @abstractmethod
    def interpret(self):
        pass


class TTest(StatisticalTest):
    """
    t-검정 (독립/대응)
    """
    
    def __init__(self):
        pass
    
    def execute(self, class0_data: pd.Series, class1_data: pd.Series, iv_col: str, dv_col: str, labels: list = [0, 1], alpha: float = 0.5):
        """
        두 그룹 간 평균 차이에 대한 가설검정을 수행하는 함수.
        (정규성에 따라 t-검정 또는 Mann-Whitney U 검정을 자동 선택)

        Parameters
        ----------
        class0_data: pd.DataFrame
            독립 변수의 값이 0인 데이터
        class1_data: pd.DataFrame
            독립 변수의 값이 1인 데이터
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
        
        self.plot(class0_data, class1_data, labels, iv_col, dv_col)
        
        equal_var = self.check_homosecedasticity(class0_data, class1_data)
        
        is_normal_0 = self.check_normality_simple(class0_data)
        is_normal_1 = self.check_normality_simple(class1_data)
        
        print("\n[가설검정]")
        print("-" * 40)

        if is_normal_0 and is_normal_1:
            print("H₀: μ₀ = μ₁ (두 클래스의 평균이 같다)")
            print("H₁: μ₀ ≠ μ₁ (두 클래스의 평균이 다르다)")
        else:
            print("H₀: 두 클래스의 분포가 같다 (중앙값 차이가 없다)")
            print("H₁: 두 클래스의 분포가 다르다 (중앙값 차이가 있다)")
            
        print(f"유의수준: α = {alpha}\n")

        # --- 검정 수행 ---
        if is_normal_0 and is_normal_1:
            # 모수 검정
            test_name = "Student's t-test" if equal_var else "Welch's t-test"
            t_stat, p_value = ttest_ind(class0_data, class1_data, equal_var=equal_var)
            print(f"{test_name} 결과:")
            print(f"t = {t_stat:.4f}, p = {p_value:.4f}")
            
            # Cohen's d 계산
            pooled_std = np.sqrt((class0_data.var() + class1_data.var()) / 2)
            cohens_d = (class0_data.mean() - class1_data.mean()) / pooled_std
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
            u_stat, p_value = mannwhitneyu(class0_data, class1_data, alternative='two-sided')
            print(f"{test_name} 결과:")
            print(f"U = {u_stat:.4f}, p = {p_value:.4f}")
            
            # 총 샘플 크기 (N)
            n0 = len(class0_data)
            n1 = len(class1_data)
            N = n0 + n1
            
            # 효과 크기 계산 (rank-biserial crrelation)
            r_rb = (2 * u_stat) / (n0 * n1) - 1
            abs_rb = abs(r_rb)
            
            if abs_rb < 0.1:
                effect = "매우 작은 효과"
            elif abs_rb < 0.3:
                effect = "작은 효과"
            elif abs_rb < 0.5:
                effect = "중간 효과"
            else:
                effect = "큰 효과"
            
            test_stat = u_stat
            effect_size = r_rb
            effect_interpretation = effect

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
    
    def check_homosecedasticity(self, class0_data, class1_data):
        print("\n[등분산성 검정]")
        print("-"*40)
        stat, p_levene = levene(class0_data, class1_data)
        print(p_levene)
        print(f"Levene's test p-value: {p_levene:.4f}")
        equal_var = p_levene > 0.05
        return equal_var
    
    def plot(self, class0_data, class1_data, labels, iv_col, dv_col):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        # 박스플롯
        bp = axes[0].boxplot([class0_data, class1_data],
                            labels=labels,
                            patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[0].set_ylabel(dv_col)
        axes[0].set_title(f'{dv_col} 분포')
        axes[0].grid(True, alpha=0.3)

        # 히스토그램
        axes[1].hist(class0_data, bins=10, alpha=0.6, label=f'{iv_col} - {labels[0]}', 
                    color='blue', density=True, edgecolor='black')
        axes[1].hist(class1_data, bins=10, alpha=0.6, label=f'{iv_col} - {labels[1]}', 
                    color='red', density=True, edgecolor='black')
        axes[1].set_xlabel(dv_col)
        axes[1].set_ylabel('밀도')
        axes[1].set_title(f'{dv_col} 분포 비교')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Q-Q plot (Class 0)
        stats.probplot(class0_data, dist="norm", plot=axes[2])
        axes[2].set_title(f'Q-Q Plot ({labels[0]})')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()