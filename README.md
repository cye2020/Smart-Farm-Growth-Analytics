# 스마트팜 생육 데이터 분석 프로젝트

## 프로젝트 디렉토리 구조
```plaintext
Project/
├── data/
│   ├── raw/              # 원본 데이터 (Git 제외)
│   ├── interim/          # 중간 처리 데이터
│   └── processed/        # 최종 분석용 데이터
├── code/
│   ├── eda/              # 탐색적 데이터 분석
│   ├── data/             # 데이터 로드/전처리 함수
│   ├── features/         # 피처 엔지니어링
│   ├── models/           # 모델 학습/예측
│   └── utils/            # 공통 유틸리티
├── models/               # 학습된 모델 파일 (.pkl, .h5)
├── reports/
│   ├── figures/          # 그래프 이미지
│   └── presentations/    # 발표자료
├── tableau/
│   ├── workbooks/        # .twb 파일
│   └── exports/          # 이미지, CSV 추출물
├── .gitignore            # Git 제외 파일 목록
├── README.md             # 프로젝트 소개
├── requirements.txt      # Python 패키지 목록
└── .env.example          # 환경변수 템플릿
```