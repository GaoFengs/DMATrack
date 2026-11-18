"""Verify that only the three score heads differ between two checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

GATE_BRANCHES = ('score_branch_rgb', 'score_branch_t', 'score_branch_rgbt')


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', default='checkpoints/DMATrack_ep0010.pth.tar')
    parser.add_argument('--after', default='checkpoints/DMATrack_gate_retrained_ep0010.pth.tar')
    parser.add_argument('--output', default='checkpoints/gate_weight_verification.json')
    args = parser.parse_args()
    before_path, after_path = Path(args.before), Path(args.after)
    before = torch.load(str(before_path), map_location='cpu')['net']
    after = torch.load(str(after_path), map_location='cpu')['net']
    report = dict(before=dict(path=str(before_path), sha256=checksum(before_path)),
                  after=dict(path=str(after_path), sha256=checksum(after_path)),
                  gate_branches=list(GATE_BRANCHES),
                  missing_after=sorted(set(before)-set(after)),
                  unexpected_after=sorted(set(after)-set(before)),
                  non_gate_tensors=0, non_gate_elements=0, non_gate_differences=[],
                  gate_tensors=0, gate_changed=[], gate_unchanged=[],
                  gate_branch_changes={name:0 for name in GATE_BRANCHES})
    frozen_hashes = [hashlib.sha256(), hashlib.sha256()]
    for name in sorted(set(before) & set(after)):
        left, right = before[name], after[name]
        gate = name.split('.')[0] in GATE_BRANCHES
        same_structure = left.shape == right.shape and left.dtype == right.dtype
        left_bytes = left.detach().cpu().contiguous().numpy().tobytes()
        right_bytes = right.detach().cpu().contiguous().numpy().tobytes()
        identical = same_structure and left_bytes == right_bytes
        if gate:
            report['gate_tensors'] += 1
            report['gate_unchanged' if identical else 'gate_changed'].append(name)
            if not identical:
                report['gate_branch_changes'][name.split('.')[0]] += 1
        else:
            report['non_gate_tensors'] += 1
            report['non_gate_elements'] += left.numel()
            if not identical:
                report['non_gate_differences'].append(name)
            for digest, tensor, data in zip(frozen_hashes, (left, right), (left_bytes, right_bytes)):
                digest.update((name+'|'+str(tensor.dtype)+'|'+str(tuple(tensor.shape))).encode())
                digest.update(data)
    report['non_gate_sha256'] = dict(before=frozen_hashes[0].hexdigest(), after=frozen_hashes[1].hexdigest())
    report['non_gate_bitwise_identical'] = not (
        report['missing_after'] or report['unexpected_after'] or report['non_gate_differences'])
    report['passed'] = report['non_gate_bitwise_identical'] and all(report['gate_branch_changes'].values())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ('gate_changed', 'gate_unchanged')}, indent=2))
    if not report['passed']:
        raise SystemExit('Checkpoint verification failed')


if __name__ == '__main__':
    main()
