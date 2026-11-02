"""Paired age benefit by delay cell and original presentation, from saved predictions.

No model is refitted. Presentation summaries average within-presentation AP
differences over the 25 delay cells and three seed-specific out-of-fold predictions.
They are descriptive, not 66 independent observations or a presentation bootstrap.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from staleness.cohort import build_cohort
from staleness.experiment import CELLS
from staleness.inference import load_pooled


def export_seed_table():
    # Seed means are complete partitions, rather than treating folds as seeds.
    fresh = pd.read_csv('results/runs_fresh.csv')
    fresh = fresh[(fresh.d_vle == 0) & (fresh.d_asm == 0)]
    metrics = ['pr_auc','roc_auc','macro_f1','recall','brier']
    seed = fresh.groupby(['model','seed'])[metrics].mean().reset_index()
    seed.to_csv('results/baseline_seed_means.csv',index=False)
    # LaTeX table used in the revised paper, including SD over complete seed means.
    mean = seed.groupby('model')[metrics].mean()
    sd = seed.groupby('model')[metrics].std(ddof=1)
    lines = [r'\begin{tabular}{@{}lcccccc@{}}\toprule',
             r'Method & PR-AUC & ROC-AUC & Macro-F1 & Recall & Brier & $p$ \\ \midrule']
    for model, label in [('B0','B0 majority'),('B1','B1 logistic'),('B2','B2 RBF-SVM'),('B3','B3 boosting')]:
        values = [f'{mean.loc[model,c]:.3f} $\\pm$ {sd.loc[model,c]:.3f}' for c in metrics]
        if model == 'B3':
            values = [r'\textbf{'+v+'}' for v in values]
        lines.append(label+' & '+' & '.join(values)+
                     (r' & reference \\' if model=='B3' else r' & $\le0.001$ \\'))
    lines.append(r'\bottomrule\end{tabular}')
    with open('results/paper_seed_table.tex','w') as fh:
        fh.write('\n'.join(lines)+'\n')
    full = pd.read_csv('results/runs_fresh.csv')
    surface = full.groupby(['model','d_vle','d_asm']).pr_auc.mean()
    panels = []
    for model, label in [('B1','B1 logistic'),('B2','B2 RBF-SVM'),('B3','B3 boosting')]:
        panels += [r'\begin{minipage}[t]{0.32\linewidth}\centering\footnotesize',
                   r'\textbf{'+label+r'}\\[3pt]',r'\setlength{\tabcolsep}{1.9pt}',
                   r'\begin{tabular}{@{}lccccc@{}}\toprule',
                   r'$\delta_V\backslash\delta_A$ & 0 & 1 & 3 & 7 & 14 \\ \midrule']
        for dv in [0,1,3,7,14]:
            panels.append(str(dv)+' & '+' & '.join(f'{surface.loc[(model,dv,da)]:.3f}'
                           for da in [0,1,3,7,14])+r' \\')
        panels += [r'\bottomrule\end{tabular}\end{minipage}'+
                   (r'\hfill' if model!='B3' else '')]
    with open('results/paper_surface_panels.tex','w') as fh:
        fh.write('\n'.join(panels)+'\n')


def export_adjacent_contrasts():
    """B3 adjacent contrasts and per-day finite differences on the fresh-trained surface.

    S(d_j) is mean test AP over the 15 folds at one source delay d_j with the other source
    fresh. Adjacent contrast = S(d_{j+1}) - S(d_j); per-day difference divides it by the
    unequal step d_{j+1} - d_j. These differ from the fresh-reference contrast S(d) - S(0).
    """
    full = pd.read_csv('results/runs_fresh.csv')
    surface = full[full.model == 'B3'].groupby(['d_vle', 'd_asm']).pr_auc.mean()
    levels = [0, 1, 3, 7, 14]
    rows = []
    for a, b in zip(levels, levels[1:]):
        asm = surface.loc[(0, b)] - surface.loc[(0, a)]
        vle = surface.loc[(b, 0)] - surface.loc[(a, 0)]
        rows.append(dict(step=f'{a} to {b}', days=b - a, assessment_contrast=asm,
                         assessment_per_day=asm / (b - a), vle_contrast=vle,
                         vle_per_day=vle / (b - a)))
    table = pd.DataFrame(rows)
    table.to_csv('results/adjacent_contrasts_B3.csv', index=False)

    def fmt(v):
        v = round(v, 4)
        if v == 0:
            return '0.0000'
        return ('$-$' if v < 0 else '+') + f'{abs(v):.4f}'

    lines = [r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
             r'Step (days) & Length & Assessment & Per day & VLE & Per day \\ \midrule']
    for r in rows:
        lines.append(f"{r['step']} & {r['days']} & {fmt(r['assessment_contrast'])} & "
                     f"{fmt(r['assessment_per_day'])} & {fmt(r['vle_contrast'])} & "
                     f"{fmt(r['vle_per_day'])} \\\\")
    lines.append(r'\bottomrule\end{tabular}')
    with open('results/paper_adjacent_table.tex', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def main():
    if "--seed-only" in sys.argv:
        export_seed_table()
        export_adjacent_contrasts()
        return
    cohort, _ = build_cohort('data/oulad')
    y = cohort.y.to_numpy()
    pred, _ = load_pooled('augmented', len(cohort))
    runs = pd.read_csv('results/runs_augmented.csv')
    paired = runs.pivot(index=['seed', 'fold', 'model', 'd_vle', 'd_asm'],
                        columns='condition', values='pr_auc')
    paired['gain'] = paired.freshness_aware - paired.augmented
    cell = paired['gain'].groupby(['model','d_vle','d_asm']).agg(['mean','std']).reset_index()
    cell.to_csv('results/paired_gain_by_delay.csv', index=False)
    marginal = []
    for source, column in [('VLE','d_vle'), ('assessment','d_asm')]:
        z = paired['gain'].groupby(['seed','fold','model',column]).mean()
        z = z.groupby(['model',column]).agg(['mean','std']).reset_index().rename(columns={column:'delay'})
        z.insert(0,'source',source)
        marginal.append(z)
    pd.concat(marginal).to_csv('results/paired_gain_by_source.csv',index=False)
    detail = []
    for presentation in sorted(cohort.presentation.unique()):
        idx = np.flatnonzero(cohort.presentation.to_numpy() == presentation)
        for model in ['B1','B3']:
            for seed in range(3):
                aware = pred[(model,'freshness_aware')][seed][idx]
                aug = pred[(model,'augmented')][seed][idx]
                for j, (dv, da) in enumerate(CELLS):
                    a = average_precision_score(y[idx],aware[:,j])
                    b = average_precision_score(y[idx],aug[:,j])
                    detail.append(dict(presentation=presentation,model=model,seed=seed,
                                       d_vle=dv,d_asm=da,n=len(idx),positives=int(y[idx].sum()),
                                       aware_ap=a,augmented_ap=b,gain=a-b))
    detail = pd.DataFrame(detail)
    detail.to_csv('results/paired_gain_presentation_detail.csv',index=False)
    summary = detail.groupby(['presentation','model']).agg(n=('n','first'),positives=('positives','first'),
                      augmented_ap=('augmented_ap','mean'),aware_ap=('aware_ap','mean'),gain=('gain','mean')).reset_index()
    summary.to_csv('results/paired_gain_by_presentation.csv',index=False)
    export_seed_table()
    export_adjacent_contrasts()
    print('Wrote delay, source, presentation and seed summaries from saved predictions.')


if __name__ == '__main__':
    main()
