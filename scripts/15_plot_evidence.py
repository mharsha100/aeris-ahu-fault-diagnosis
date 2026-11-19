from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/end_to_end_20261116';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(name):plt.savefig(FIG/name,dpi=160,bbox_inches='tight');plt.close()
def main():
    selection=json.loads((OUT/'selection.json').read_text());chosen=selection['per_architecture']
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for col,(v,name) in enumerate(chosen.items()):
        ax=axes[0,col]
        for path in sorted(OUT.glob(f'e2e_*_{v}*/learning_curve.csv')):
            d=pd.read_csv(path);label=path.parent.name.replace('e2e_','');ax.plot(d.epoch,d.validation_macro_f1,label=label,alpha=.8)
            axes[1,col].plot(d.epoch,d.train_weighted_loss,label=label,alpha=.8)
        ax.set(title=v.upper(),xlabel='Completed epoch',ylabel='Development macro F1');ax.legend(fontsize=6)
        axes[1,col].set(xlabel='Completed epoch',ylabel='Training loss (run-specific weighting)')
    fig.suptitle('Fresh temporal training — source development only');fig.tight_layout();save('learning_curves.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,(v,name) in zip(axes,chosen.items()):
        m=json.loads((OUT/f'{name}_hybrid_metrics.json').read_text());cm=np.array(m['confusion_matrix']);ax.imshow(cm,cmap='Blues')
        for (i,j),z in np.ndenumerate(cm):ax.text(j,i,str(z),ha='center',va='center',color='white' if z>cm.max()*.55 else 'black')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=['N','D','C','H'],yticklabels=['N','D','C','H'],xlabel='Predicted class',ylabel='True class',title=v.upper())
    fig.suptitle('6,930 development windows — not independent cases');fig.tight_layout();save('confusion_matrices.png')
    d=pd.read_csv(OUT/'reliability.csv');fig,ax=plt.subplots(figsize=(6,4));ax.plot([0,1],[0,1],'k--',label='Ideal')
    for name,g in d.groupby('bundle'):ax.plot(g.confidence,g.accuracy,'o-',label=name.replace('e2e_',''))
    ax.set(xlabel='Mean confidence within bin',ylabel='Observed window accuracy',title='Source development calibration',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8);save('reliability.png')
    d=pd.read_csv(OUT/'coverage.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,g in d.groupby('bundle'):
        axes[0].plot(g.coverage,g.accepted_false_normal,'o-',label=name.replace('e2e_',''));axes[1].plot(g.threshold,g.review_windows/g.windows,'o-',label=name.replace('e2e_',''))
    axes[0].set(xlabel='Accepted coverage',ylabel='Accepted fault-to-normal windows',title='Missed faults versus coverage');axes[1].set(xlabel='Confidence threshold',ylabel='Review fraction',title='Review workload');axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);save('coverage_tradeoff.png')
    d=pd.read_csv(OUT/'before_after.csv');fig,ax=plt.subplots(figsize=(10,4));ax.bar(range(len(d)),d.macro_f1);ax.set(xticks=range(len(d)),xticklabels=d.name.str.replace('e2e_',''),ylabel='Source development macro F1',ylim=(.5,.8),title='Full-budget anchors and three-seed finalists');ax.tick_params(axis='x',rotation=55);save('before_after.png')
if __name__=='__main__':main()
