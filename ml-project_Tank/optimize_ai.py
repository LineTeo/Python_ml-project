#!/usr/bin/env python3
"""
optimize_ai.py  ―  CMA-ES による AIConfig パラメータ最適化
================================================================

動作フロー:
  1. Optuna (CmaEsSampler) がパラメータ候補を生成
  2. JSON に書き出し → Java プロセスを subprocess で起動
  3. Java 側が "WIN_RATE:0.6750" を標準出力 → Python が読み取る
  4. 勝率を目的関数値として Optuna にフィードバック
  5. n_trials 回繰り返して最良パラメータを保存

必要ライブラリ:
  pip install optuna

Java 側のビルド・実行コマンドを JAVA_CMD に設定してください。
"""

import json
import subprocess
import tempfile
import os
import optuna
#from optuna.samplers import CmaEsSampler
from optuna.samplers import CmaEsSampler, RandomSampler 

# ======================================================================
# 設定
# ======================================================================

# Javaの実行コマンド（クラスパスは環境に合わせて変更）
# 例: javac でビルド済みの場合
JAVA_CMD = [
    "java",
    "-cp", "../../../../pleiades/2024-12/workspace/ClassTrain/bin",          # ← クラスファイルのディレクトリに変更
    "war.main.BattleEvaluator"
]

# 1試合の評価に使うゲーム数（多いほど精度↑・時間↑）
GAMES_PER_TRIAL = 5000

# 最適化の試行回数
N_TRIALS = 1000

# 結果保存先
BEST_CONFIG_PATH = "best_ai_config.json"
STUDY_DB_PATH    = "sqlite:///optuna_study.db"   # 途中再開用DB
"""
sqlite:///optuna_study.db は「SQLiteのDBに接続するためのURL形式」
スラッシュの数の違いがポイント：
sqlite:///optuna_study.db
→ 相対パス（今いるフォルダ基準）
sqlite:////absolute/path/to/optuna_study.db
→ 絶対パス（/ が1個増える）
これはOptunaが内部でつかうDB接続ライブラリ（SQLAlchemy）のDB接続形式
例えば：
MySQL → mysql://user:pass@host/db
PostgreSQL → postgresql://...
SQLite → sqlite:///...
"""


# ======================================================================
# 目的関数
# ======================================================================

def objective(trial: optuna.Trial) -> float:
    """
    ": optuna.Trial"は この引数がoptuna.Trialだよっていう補助情報。なくても動作する。
    1 trial = パラメータ1セットで GAMES_PER_TRIAL 試合を評価。
    返り値: 敵軍勝率（最大化）
    """

    # ── パラメータ候補を生成 ──────────────────────────────────────────
    # suggest_float(name, low, high) で探索範囲を指定
    # THREAT_ZONE_1 < THREAT_ZONE_2 の制約を守るため調整 (辞書リテラルの中ではやらない方がいいので外に出す)
    zone1 = trial.suggest_float("THREAT_ZONE_1", 0.1, 0.5)
    zone2 = trial.suggest_float("THREAT_ZONE_2", zone1 + 0.1, min(zone1 + 0.5, 0.9))

    config = {
        "THREAT_ZONE_1":    zone1,
        "THREAT_ZONE_2":    zone2,
        "P01_Z1_AA_AC_2_O": trial.suggest_float("P01_Z1_AA_AC_2_O", 0.0, 1.0),
        "P02_Z1_AE_DT_2_U": trial.suggest_float("P02_Z1_AE_DT_2_U", 0.0, 1.0),
        "P03_Z1_ER_DT_2_U": trial.suggest_float("P03_Z1_ER_DT_2_U", 0.0, 1.0),
        "P05_Z2_CA_AT_1_O": trial.suggest_float("P05_Z2_CA_AT_1_O", 0.0, 1.0),
        "P06_Z2_AA_DC_2_U": trial.suggest_float("P06_Z2_AA_DC_2_U", 0.0, 1.0),
        "P07_Z2_AE_DT_2_U": trial.suggest_float("P07_Z2_AE_DT_2_U", 0.0, 1.0),
        "P08_Z2_EE_DT_2_O": trial.suggest_float("P08_Z2_EE_DT_2_O", 0.0, 1.0),
        "P10_Z3_CC_DT_2_U": trial.suggest_float("P10_Z3_CC_DT_2_U", 0.0, 1.0),
        "P11_Z3_CA_DT_2_U": trial.suggest_float("P11_Z3_CA_DT_2_U", 0.0, 1.0),
        # RANK系: 順位が高いほど脅威大 という制約を維持
        "RANK_1_THREAT":    1.0,  # 1位は固定（基準値）
        "RANK_2_THREAT":    trial.suggest_float("RANK_2_THREAT", 0.0, 1.0),
        "RANK_3_THREAT":    trial.suggest_float("RANK_3_THREAT", 0.0, 0.5),
        "RANK_4_THREAT":    trial.suggest_float("RANK_4_THREAT", 0.0, 0.3),
    }

    # ── 一時JSONファイルに書き出し ────────────────────────────────────
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2)
        tmp_path = f.name

    # ── Java プロセス起動 ────────────────────────────────────────────
    try:
        cmd = JAVA_CMD + [tmp_path, str(GAMES_PER_TRIAL)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120   # 2分タイムアウト
        )

        if result.returncode != 0:
            print(f"[Trial {trial.number}] Java エラー:\n{result.stderr}")
            return 0.0  # エラー時はペナルティ

        # "WIN_RATE:0.6750" の行を探してパース
        win_rate = _parse_win_rate(result.stdout)
        print(f"[Trial {trial.number:>4}] 勝率={win_rate:.4f}  params={_short_params(config)}")
        return win_rate

    except subprocess.TimeoutExpired:
        print(f"[Trial {trial.number}] タイムアウト")
        return 0.0
    finally:
        os.unlink(tmp_path)  # 一時ファイルを削除


def _parse_win_rate(stdout: str) -> float:
    """標準出力から 'WIN_RATE:0.6750' 形式の行を読み取る"""
    for line in stdout.splitlines():
        if line.startswith("WIN_RATE:"):
            return float(line.split(":")[1])
    raise ValueError(f"WIN_RATE 行が見つかりません:\n{stdout}")


def _short_params(cfg: dict) -> str:
    """ログ出力用に主要パラメータを短縮表示"""
    return (f"Z1={cfg['THREAT_ZONE_1']:.2f} Z2={cfg['THREAT_ZONE_2']:.2f} "
            f"P01={cfg['P01_Z1_AA_AC_2_O']:.2f}")


# ======================================================================
# メイン
# ======================================================================

def main():
    print("=" * 60)
    print("AIConfig 最適化開始（CMA-ES）")
    print(f"  試行回数    : {N_TRIALS}")
    print(f"  試合数/試行 : {GAMES_PER_TRIAL}")
    print(f"  結果保存先  : {BEST_CONFIG_PATH}")
    print("=" * 60)

    # Study 作成（DB指定で途中再開可能）
    # CMA-ESはウォームアップ試行が必要なため n_startup_trials を指定する
    # (これがないと最初のsample_relativeで初期化エラーが発生する)
    sampler = CmaEsSampler(
        seed=42,
        n_startup_trials=20,           # 最初の20試行はランダム探索でウォームアップ
        warn_independent_sampling=False,
        restart_strategy="ipop",       # 局所解にはまったら自動リスタート
    )
    study = optuna.create_study(
        direction="maximize",       # 勝率を最大化
        sampler=sampler,            # ウォームアップ試行を含むCMA-ESサンプラー
    #    sampler=CmaEsSampler(seed=42),
        study_name="tank_ai_opt",
        storage=STUDY_DB_PATH,
        load_if_exists=True,        # 既存studyがあれば再開
    )

    # 最適化実行
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # ── 結果表示 ──────────────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"最適化完了！  最良勝率: {best.value:.4f}  (Trial #{best.number})")
    print("=" * 60)
    print("\n最良パラメータ:")
    for k, v in best.params.items():
        print(f"  {k:30s} = {v:.6f}")

    # RANK_1_THREAT は固定値なので追記
    best_config = dict(best.params)
    best_config["RANK_1_THREAT"] = 1.0

    # JSON保存
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    print(f"\n最良パラメータを保存: {BEST_CONFIG_PATH}")

    # ── 上位10件の勝率推移を表示 ──────────────────────────────────────
    print("\n--- 上位10試行 ---")
    top10 = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:10]
    for t in top10:
        print(f"  Trial#{t.number:>4}  勝率={t.value:.4f}")


if __name__ == "__main__":
    main()
