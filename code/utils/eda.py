# 표준 라이브러리
from IPython.display import display
from typing import List
from itertools import combinations

# 서드파티 라이브러리
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# 로컬 모듈
from .config import growth_map


def eda_missing_data(data: pd.DataFrame, ext: str = 'ipynb') -> None:
    """결측치 탐색 함수

    Args:
        data (pd.DataFrame): 입력 데이터
    """
    
    # 만약, jupyter notebook에서 코드를 실행한다면, display로 출력
    if ext == 'ipynb':
        print = display
    else:
        pass
    
    # 결측치 시각화: 대략적으로 보기
    msno.matrix(data)
    plt.show()
    
    # 결측치 출력: 정확한 개수 알아보기
    print(data.isnull().sum())


def eda_duplicates(data: pd.DataFrame, columns: List[str], ext: str = 'ipynb') -> None:
    # 만약, jupyter notebook에서 코드를 실행한다면, display로 출력
    if ext == 'ipynb':
        print = display
    else:
        pass
    
    num_cols = len(columns)
    
    for i in range(num_cols):
        print('='*50)
        print(f'컬럼 {i+1}개 PK 검사')
        print('='*50)
        
        for cols in combinations(columns, i+1):
            # print('-'*30)
            print(f'{"와 ".join([growth_map[col] for col in cols])}의 PK 검사')
            # print('-'*30)
            print(data.duplicated(subset=cols).value_counts())
            

def plot_one_feature(data: pd.DataFrame, feature, color, axes):
    sns.boxplot(data, y=feature, color=color, ax=axes[0])
    axes[0].set_title('Box Plot')
    axes[0].grid(True, alpha=0.3)
    stats.probplot(data[feature], dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    sns.histplot(data, x=feature, color=color, ax=axes[2])
    axes[2].set_title('히스토그램')
    axes[2].grid(True, alpha=0.3)

def plot_features(data: pd.DataFrame, features, colors):
    n = len(features)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
    
    for i in range(n):
        feature, color = features[i], colors[i]
        ax = axes if n == 1 else axes[i]
        plot_one_feature(data, feature, color, ax)

    plt.tight_layout()
    plt.show()