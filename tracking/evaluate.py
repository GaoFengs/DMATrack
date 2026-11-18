import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset


def evaluate(results, sequences=None):
    dataset = get_dataset('lasher')
    if sequences is not None:
        dataset = [s for s in dataset if s.name in sequences]
        if len(dataset) != len(set(sequences)):
            raise ValueError('Unknown sequence')
    rows = []
    thresholds = torch.arange(0, 1.05, .05, dtype=torch.float64)
    for seq in dataset:
        boxes = np.loadtxt(Path(results)/(seq.name+'.txt'), ndmin=2)
        if boxes.shape != seq.ground_truth_rect.shape:
            raise ValueError('Prediction/annotation shape mismatch: '+seq.name)
        overlap, center, normalized, _ = calc_seq_err_robust(
            torch.tensor(boxes, dtype=torch.float64),
            torch.tensor(seq.ground_truth_rect, dtype=torch.float64), seq.dataset)
        rows.append(dict(sequence=seq.name, frames=len(boxes),
                         Precision=100*float((center <= 20).double().mean()),
                         NormPrecision=100*float((normalized <= .2).double().mean()),
                         AUC=100*float((overlap[:, None] > thresholds[None]).double().mean())))
    if not rows:
        raise ValueError('No sequences to evaluate')
    return dict(protocol='project: sequence-mean P@20, NP@0.2, overlap AUC (0:0.05:1)',
                sequences=len(rows), frames=sum(r['frames'] for r in rows),
                metrics={k:float(np.mean([r[k] for r in rows]))
                         for k in ('Precision', 'NormPrecision', 'AUC')}, per_sequence=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--sequences', nargs='+')
    args = parser.parse_args()
    report = evaluate(args.results, args.sequences)
    (Path(args.results)/'evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_sequence'}, indent=2))


if __name__ == '__main__':
    main()
