import numpy as np
import json
from GSDIJ_AHP_unified import GSDIJ_AHP

from pathlib import Path
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
json_file = BASE_DIR / "data" / "article_example1.json"



try:
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"  Файл {json_file} загружен")
    print(f"  Критериев: {len(data.get('criteria', []))}")
    print(f"  Экспертов: {len(data.get('dms', []))}")
except FileNotFoundError:
    print(f"Файл {json_file} не найден!")

tt = 5
ss = 0.3

print("WGMM (Weighted Geometric Mean Matrix)")

gsd_wgmm = GSDIJ_AHP(debug_mode=True, center_method="wgmm")
gsd_wgmm.load_from_json(json_file)

results_wgmm = gsd_wgmm.run_complete_analysis(t=tt, s=ss)
gsd_wgmm.save_interval_and_ranking("results_unified_wgmm.json")

print(f"  λ_opt = {results_wgmm['lambda_opt']:.4f}")
print(f"  U = {results_wgmm['U']:.4f}")
print(f"  GSI = {results_wgmm['GSI']:.4f}")

print("RowSum (Row Sum Method)")

gsd_rowsum = GSDIJ_AHP(debug_mode=True, center_method="rowsum")
gsd_rowsum.load_from_json(json_file)

results_rowsum = gsd_rowsum.run_complete_analysis(t=tt, s=ss)
gsd_rowsum.save_interval_and_ranking("results_unified_rowsum.json")

print(f"  λ_opt = {results_rowsum['lambda_opt']:.4f}")
print(f"  U = {results_rowsum['U']:.4f}")
print(f"  GSI = {results_rowsum['GSI']:.4f}")

print("Tropical (Tropical log-Chebyshev Center)")

gsd_tropical = GSDIJ_AHP(debug_mode=True, center_method="tropical", tropical_variant="worst")
gsd_tropical.load_from_json(json_file)

results_tropical = gsd_tropical.run_complete_analysis(t=tt, s=ss)
gsd_tropical.save_interval_and_ranking("results_unified_tropical.json")

print(f"  λ_opt = {results_tropical['lambda_opt']:.4f}")
print(f"  U = {results_tropical['U']:.4f}")
print(f"  GSI = {results_tropical['GSI']:.4f}")

print("СРАВНЕНИЕ ТРЁХ МЕТОДОВ")

import pandas as pd

comparison_data = {
    'Метод': ['WGMM', 'RowSum', 'Tropical'],
    'λ_opt': [
        results_wgmm['lambda_opt'],
        results_rowsum['lambda_opt'],
        results_tropical['lambda_opt']
    ],
    'U': [
        results_wgmm['U'],
        results_rowsum['U'],
        results_tropical['U']
    ],
    'GSI': [
        results_wgmm['GSI'],
        results_rowsum['GSI'],
        results_tropical['GSI']
    ],
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))


criteria_names = results_wgmm['criteria_names']
weights_wgmm = results_wgmm['weights']
weights_rowsum = results_rowsum['weights']
weights_tropical = results_tropical['weights']

weights_comparison = pd.DataFrame({
    'Критерий': criteria_names,
    'WGMM': [f"{w:.4f}" for w in weights_wgmm],
    'RowSum': [f"{w:.4f}" for w in weights_rowsum],
    'Tropical': [f"{w:.4f}" for w in weights_tropical],
})

print("\n" + weights_comparison.to_string(index=False))

print("КОРРЕЛЯЦИЯ WGMM С RowSum И Tropical")

from scipy.stats import pearsonr, spearmanr
import numpy as np

wgmm_weights = np.array(results_wgmm['weights'])
rowsum_weights = np.array(results_rowsum['weights'])
tropical_weights = np.array(results_tropical['weights'])

# 1. ВЫЧИСЛЯЕМ РАНГИ (1 = самый важный, N = наименее важный)
def get_ranks(weights):
    """Преобразует веса в ранги (1 = самый важный)"""
    sorted_indices = np.argsort(weights)[::-1]  
    ranks = np.zeros_like(sorted_indices, dtype=float)
    
    # Присваиваем ранги от 1 до N
    for rank, idx in enumerate(sorted_indices, 1):
        ranks[idx] = rank
    
    return ranks

wgmm_ranks = get_ranks(wgmm_weights)
rowsum_ranks = get_ranks(rowsum_weights)
tropical_ranks = get_ranks(tropical_weights)

# 2. ВЫЧИСЛЯЕМ КОРРЕЛЯЦИИ
print("\n1. КОРРЕЛЯЦИЯ WGMM С RowSum:")

wgmm_rowsum_rank_corr = spearmanr(wgmm_ranks, rowsum_ranks)[0]

print(f"Корреляция по рангам (Спирмен): {wgmm_rowsum_rank_corr:.4f}")

print("\n2. КОРРЕЛЯЦИЯ WGMM С Tropical:")

# По рангам (Спирмен)
wgmm_tropical_rank_corr = spearmanr(wgmm_ranks, tropical_ranks)[0]

print(f"Корреляция по рангам (Спирмен): {wgmm_tropical_rank_corr:.4f}")

# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ КОРРЕЛЯЦИЙ
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")

correlation_results = {
    'wgmm_rowsum': {
        'rank_correlation': float(wgmm_rowsum_rank_corr)
    },
    'wgmm_tropical': {
        'rank_correlation': float(wgmm_tropical_rank_corr)
    },
    'comparison': {
        'closer_by_ranks': 'RowSum' if wgmm_rowsum_rank_corr > wgmm_tropical_rank_corr else 'Tropical',
        'rank_correlation_difference': float(abs(wgmm_rowsum_rank_corr - wgmm_tropical_rank_corr))
    },
    'parameters': {
        'lambda_wgmm': results_wgmm['lambda_opt'],
        'lambda_rowsum': results_rowsum['lambda_opt'],
        'lambda_tropical': results_tropical['lambda_opt'],
        'U_wgmm': results_wgmm['U'],
        'U_rowsum': results_rowsum['U'],
        'U_tropical': results_tropical['U'],
        'GSI_wgmm': results_wgmm['GSI'],
        'GSI_rowsum': results_rowsum['GSI'],
        'GSI_tropical': results_tropical['GSI']
    }
}

with open("correlation_wgmm_comparison.json", "w", encoding="utf-8") as f:
    json.dump(correlation_results, f, ensure_ascii=False, indent=2)

print("Результаты корреляций сохранены в correlation_wgmm_comparison.json")


print("СРАВНЕНИЕ РАНЖИРОВАНИЯ КРИТЕРИЕВ")
print("\nWGMM:")
for i, name in enumerate(results_wgmm['ranking_names'], 1):
    print(f"  {i}. {name}")

print("\nRowSum:")
for i, name in enumerate(results_rowsum['ranking_names'], 1):
    print(f"  {i}. {name}")

print("\nTropical:")
for i, name in enumerate(results_tropical['ranking_names'], 1):
    print(f"  {i}. {name}")

