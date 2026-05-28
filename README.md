# MSL
머신러닝 팀 프로젝트
ml-project-team1/                  ← 깃허브 루트
├── project_1/                     ← 🎯 제출용 (이대로 zip)
│   ├── main.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── final_model.pkl
│   ├── data/
│   │   └── data.csv               ← 용량 크면 .gitignore 처리
│   ├── result/
│   │   ├── prediction.csv
│   │   ├── metrics.txt
│   │   └── confusion_matrix.png
│   ├── README.docx                ← 공지 양식
│   └── requirements.txt
│
├── notebooks/                     ← 🧪 실험용 (제출 X)
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_tuning.ipynb
│
├── docs/                          ← 📄 문서·발표자료
│   ├── 1_발표자료.pptx
│   ├── meeting_notes.md
│   └── references.md
│
├── .gitignore
└── README.md                      ← 깃허브용 (제출 README와 별개)
