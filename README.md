# 축산 유성분검사성적 데이터 분석 프로젝트

목표: 우유 생산성 및 품질 최적화 방안 도출

## 프로젝트 디렉토리 구조

```plaintext
Project/
├── data/
│   ├── raw/              # 원본 데이터 (Git 제외)
│   └── processed/        # 최종 분석용 데이터
├── code/
│   ├── eda/              # 탐색적 데이터 분석
│   ├── data/             # 데이터 로드/전처리 함수
│   ├── models/           # 모델 학습/예측
│   └── utils/            # 공통 유틸리티
├── models/               # 학습된 모델 파일 (.pkl)
├── reports/
│   ├── figures/          # 그래프 이미지
│   └── presentations/    # 발표자료
├── tableau/              # 태블로 파일
├── .gitignore            # Git 제외 파일 목록
├── README.md             # 프로젝트 소개
└── requirements.txt      # Python 패키지 목록
```

## 원본 데이터 준비

[스마트팜코리아 OPENAPI](https://www.smartfarmkorea.net/openApi/openApiList.do?menuId=M11040303)

1. 낙농 분석용 데이터 유성분검사성적서 정보
2. 낙농 분석용 데이터 로봇착유기 정보
3. 낙농 분석용 데이터 ICT착유기 정보

## 파이썬 환경 설치

Install packages with:

```bash
pip install -r requirements.txt
```
