#!/usr/bin/env python3
"""
analyze_results.py  ―  最適化結果の可視化・分析
================================================================
optimize_ai.py 実行後に使用。

必要ライブラリ:
  pip install optuna matplotlib pandas
"""

import json
import optuna
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 文字化け回避

STUDY_DB_PATH    = "sqlite:///optuna_study.db"
BEST_CONFIG_PATH = "best_ai_config.json"


def main():
    # Study の読み込み
    study = optuna.load_study(
        study_name="tank_ai_opt",
        storage=STUDY_DB_PATH,
    )

    trials_df = study.trials_dataframe()
    completed = trials_df[trials_df["state"] == "COMPLETE"]

    print(f"完了試行数: {len(completed)}")
    print(f"最良勝率:   {completed['value'].max():.4f}")
    print(f"平均勝率:   {completed['value'].mean():.4f}")

    # ── 図1: 勝率の推移 ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AIConfig Optimization Results (CMA-ES)", fontsize=14)

    ax = axes[0, 0]
    ax.plot(completed["number"], completed["value"], alpha=0.6, lw=0.8, label="win rate")
    # ベストスコアの推移
    best_so_far = completed["value"].cummax()
    ax.plot(completed["number"], best_so_far, color="red", lw=1.5, label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 図2: パラメータ重要度 ─────────────────────────────────────────
    try:
        importance = optuna.importance.get_param_importances(study)
        ax = axes[0, 1]
        params = list(importance.keys())[:10]   # 上位10件
        values = [importance[p] for p in params]
        ax.barh(params[::-1], values[::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Parameter Importance (Top 10)")
        ax.grid(True, alpha=0.3, axis='x')
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f"重要度計算エラー\n{e}",
                        ha='center', va='center', transform=axes[0, 1].transAxes)

    # ── 図3: 勝率のヒストグラム ───────────────────────────────────────
    ax = axes[1, 0]
    ax.hist(completed["value"], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(completed["value"].max(), color='red', linestyle='--', label=f"Best: {completed['value'].max():.3f}")
    ax.set_xlabel("Win Rate")
    ax.set_ylabel("Count")
    ax.set_title("Win Rate Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 図4: THREAT_ZONE_1 vs THREAT_ZONE_2 散布図 ──────────────────
    ax = axes[1, 1]
    z1_col = "params_THREAT_ZONE_1"
    z2_col = "params_THREAT_ZONE_2"
    if z1_col in completed.columns and z2_col in completed.columns:
        sc = ax.scatter(
            completed[z1_col], completed[z2_col],
            c=completed["value"], cmap="RdYlGn",
            alpha=0.7, s=30
        )
        plt.colorbar(sc, ax=ax, label="Win Rate")
        ax.set_xlabel("THREAT_ZONE_1")
        ax.set_ylabel("THREAT_ZONE_2")
        ax.set_title("Zone Parameter Space")
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "ゾーンパラメータなし",
                        ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig("optimization_results.png", dpi=150, bbox_inches="tight")
    print("\nグラフ保存: optimization_results.png")
    plt.show()

    # ── 最良パラメータの表示 ──────────────────────────────────────────
    print("\n--- 最良パラメータ (best_ai_config.json) ---")
    try:
        with open(BEST_CONFIG_PATH, encoding="utf-8") as f:
            best = json.load(f)
        for k, v in best.items():
            print(f"  {k:30s} = {v:.6f}")
    except FileNotFoundError:
        print("  (まだ生成されていません。optimize_ai.py を先に実行してください)")

    # ── デフォルト値との比較 ──────────────────────────────────────────
    defaults = {
        "THREAT_ZONE_1": 0.3, "THREAT_ZONE_2": 0.6,
        "P01_Z1_AA_AC_2_O": 0.9, "P02_Z1_AE_DT_2_U": 0.2,
        "P03_Z1_ER_DT_2_U": 0.6, "P05_Z2_CA_AT_1_O": 1.0,
        "P06_Z2_AA_DC_2_U": 0.3, "P07_Z2_AE_DT_2_U": 0.3,
        "P08_Z2_EE_DT_2_O": 0.7, "P10_Z3_CC_DT_2_U": 0.3,
        "P11_Z3_CA_DT_2_U": 0.3, "RANK_1_THREAT": 1.0,
        "RANK_2_THREAT": 0.4, "RANK_3_THREAT": 0.2, "RANK_4_THREAT": 0.1,
    }

    try:
        with open(BEST_CONFIG_PATH, encoding="utf-8") as f:
            best = json.load(f)
        print("\n--- デフォルト値との差分 ---")
        print(f"  {'パラメータ':30s}  {'デフォルト':>12}  {'最適値':>12}  {'変化':>10}")
        print("  " + "-" * 68)
        for k in defaults:
            d = defaults[k]
            o = best.get(k, d)
            diff = o - d
            marker = "▲" if diff > 0.05 else ("▼" if diff < -0.05 else " ")
            print(f"  {k:30s}  {d:>12.4f}  {o:>12.4f}  {marker}{diff:>+9.4f}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
