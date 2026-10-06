import pandas as pd
from scipy import stats

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv("reports/unified_results.csv")


# ============================================================
# SPLIT BY CPD SIGNAL STRENGTH
# ============================================================
#
# После применения tolerance=1 год при ансамблировании CPD
# обнаруживается во всех 85 регионах. Группа «без CPD» пуста,
# и сравнение «с CPD / без CPD» невозможно.
#
# Вместо этого делим регионы по силе сигнала CPD:
#   - Strong CPD: cpd_max_signal выше медианы
#   - Weak CPD:   cpd_max_signal ниже или равен медиане
#

signal = df["cpd_max_signal"].dropna()
median_signal = signal.median()

print("=" * 60)
print("CPD SIGNAL STRENGTH ANALYSIS")
print("=" * 60)
print(f"Regions with CPD signal: {len(signal)} из {len(df)}")
print(f"Median cpd_max_signal: {median_signal:.4f}")
print(f"Min: {signal.min():.4f}, Max: {signal.max():.4f}")


strong = df[df["cpd_max_signal"] > median_signal]["xgboost_sMAPE"].dropna()
weak = df[df["cpd_max_signal"] <= median_signal]["xgboost_sMAPE"].dropna()

print(f"\nStrong CPD (signal > median): n={len(strong)}, "
      f"mean sMAPE={strong.mean():.2f}%, median={strong.median():.2f}%")
print(f"Weak CPD (signal <= median):  n={len(weak)}, "
      f"mean sMAPE={weak.mean():.2f}%, median={weak.median():.2f}%")


# ============================================================
# MANN-WHITNEY U TEST
# ============================================================

stat, p_val = stats.mannwhitneyu(strong, weak, alternative="two-sided")

print(f"\nMann-Whitney U: stat={stat:.2f}, p-value={p_val:.4f}")

if p_val > 0.05:
    print("-> Разница в sMAPE между сильным и слабым CPD "
          "статистически НЕЗНАЧИМА (p > 0.05).")
else:
    print("-> Разница в sMAPE между сильным и слабым CPD "
          "статистически ЗНАЧИМА (p <= 0.05).")


# ============================================================
# ADDITIONAL CONTEXT
# ============================================================

print("\n=== ДОПОЛНИТЕЛЬНО: УСТОЙЧИВОСТЬ РАЗНИЦЫ ===")

# Разница медиан — устойчивее к выбросам, чем разница средних
diff_mean = weak.mean() - strong.mean()
diff_median = weak.median() - strong.median()

print(f"Разница mean:   {diff_mean:+.2f} п.п.")
print(f"Разница median: {diff_median:+.2f} п.п.")


# ============================================================
# INTERPRETATION
# ============================================================

print("\n=== ИНТЕРПРЕТАЦИЯ ===")

if p_val > 0.05:
    print(
        "Статистически значимой разницы в качестве прогноза XGBoost "
        "между регионами с сильным и слабым сигналом CPD не обнаружено.\n"
        "Причинная связь между силой структурных сдвигов и качеством "
        "прогнозирования не установлена."
    )
else:
    print(
        "Обнаружена статистически значимая разница. Регионы с сильным "
        "сигналом CPD имеют другой уровень ошибки прогноза."
    )