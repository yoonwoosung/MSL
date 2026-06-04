"""
결과 시각화 스크립트 (make_result_charts.py)

result/model_comparison.csv 와 result/metrics.txt 를 읽어
보고서/발표용 막대그래프 이미지를 생성한다.
  - result/model_comparison_chart.png : 3개 모델 지표 비교 (그룹 막대)
  - result/metrics_chart.png          : 최종 모델 클래스별 지표
"""

import os
import csv

import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = "result"


def load_comparison(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def chart_model_comparison(rows, out_path):
    """3개 모델을 Accuracy/Precision/Recall/F1 4개 지표로 비교하는 그룹 막대그래프."""
    models = [r["model"] for r in rows]
    metrics = ["test_accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]

    data = np.array([[float(r[m]) for m in metrics] for r in rows])  # (모델수, 4)

    x = np.arange(len(labels))
    width = 0.8 / len(models)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, data[i], width, label=model, color=colors[i % len(colors)])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Test Set)")
    # 0.98 부근에 몰려 있어 차이가 잘 보이도록 y축 확대
    ymin = max(0.0, data.min() - 0.01)
    ax.set_ylim(ymin, 1.002)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장됨: {out_path}")


def chart_final_metrics(rows, out_path):
    """최종 모델(RandomForest)의 핵심 지표를 단일 막대그래프로."""
    # 최종 모델은 RandomForest로 고정한다. (F1 최댓값으로 고르면 소수점 4째 자리
    #  차이로 DecisionTree가 뽑혀 슬라이드 제목이 잘못 나오는 문제가 있었음)
    best = next((r for r in rows if r["model"] == "RandomForest"),
                max(rows, key=lambda r: float(r["f1"])))
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [float(best["test_accuracy"]), float(best["precision"]),
              float(best["recall"]), float(best["f1"])]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0.98, 1.002)
    ax.set_ylabel("Score")
    ax.set_title(f"Final Model Metrics — {best['model']}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장됨: {out_path}")


def main():
    csv_path = os.path.join(RESULT_DIR, "model_comparison.csv")
    if not os.path.exists(csv_path):
        print(f"에러: {csv_path} 가 없습니다. 먼저 main.py를 실행하세요.")
        return
    rows = load_comparison(csv_path)
    chart_model_comparison(rows, os.path.join(RESULT_DIR, "model_comparison_chart.png"))
    chart_final_metrics(rows, os.path.join(RESULT_DIR, "metrics_chart.png"))


if __name__ == "__main__":
    main()
