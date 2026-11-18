import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED_SHA = '1b3b5818641af9bcfee74c96eebb6602457f2b51badc1e7610976fb4782e249b'
STATE = {}


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _initialize_worker(gpus, options):
    worker = mp.current_process()._identity[0]-1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus[worker % len(gpus)])
    import torch
    from lib.test.evaluation import get_dataset
    from lib.test.parameter.ckd_repro_7578 import parameters
    from lib.test.tracker.ckd_repro_7578 import HistoricalCKD
    torch.set_num_threads(2)
    log = open(Path(options['output'])/('worker_%d.log'%worker), 'w', buffering=1)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        params = parameters('CKD_ba_rgbt_base_gate', options['checkpoint'])
        params.debug = 0
        params.inference_strategy = options['strategy']
        params.template_update_interval = 50
        params.template_score_threshold = options['score']
        params.template_iou_threshold = options['iou']
        params.tau_rgb, params.tau_tir = options['tau_rgb'], options['tau_tir']
        tracker = HistoricalCKD(params, 'lasher')
    STATE.update(tracker=tracker, dataset=get_dataset('lasher'), options=options, log=log)
    sources = {}
    for name, module in list(sys.modules.items()):
        if name == 'lib' or name.startswith(('lib.', 'external.')):
            location = getattr(module, '__file__', None)
            if location:
                path = Path(location).resolve()
                try:
                    relative = str(path.relative_to(ROOT))
                except ValueError:
                    raise RuntimeError('External source dependency: '+str(path))
                sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(Path(options['output'])/('sources_worker_%d.json'%worker), sources)


def initialize_worker(gpus, options):
    try:
        _initialize_worker(gpus, options)
    except BaseException:
        STATE['initialization_error'] = traceback.format_exc()


def read_pair(paths):
    import cv2
    pair = [cv2.imread(str(path)) for path in paths]
    if any(image is None for image in pair):
        raise IOError('Unreadable frame pair: '+str(paths))
    return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in pair]


def run_sequence(name):
    if 'initialization_error' in STATE:
        raise RuntimeError(STATE['initialization_error'])
    import numpy as np
    seq = STATE['dataset'][name]
    tracker = STATE['tracker']
    directory = Path(STATE['options']['output'])/'results'
    with (directory/(name+'_updates.log')).open('w', buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = tracker.initialize(*read_pair(seq.frames[0]), seq.init_info())
        boxes, gates, experts = [seq.init_info()['init_bbox']], [result['source_gate']], [result['expert']]
        for paths in seq.frames[1:]:
            assert tracker.template_update_interval == 50
            result = tracker.track(*read_pair(paths))
            boxes.append(result['target_bbox'])
            gates.append(result['source_gate'])
            experts.append(result['expert'])
    np.savetxt(directory/(name+'.txt'), np.round(np.asarray(boxes, dtype=np.float64), 6),
               fmt='%.6f', delimiter='\t')
    (directory/(name+'_source_gate.txt')).write_text('\n'.join(gates)+'\n')
    (directory/(name+'_expert.txt')).write_text('\n'.join(experts)+'\n')
    log = (directory/(name+'_updates.log')).read_text()
    checks = set(map(int, re.findall(r'\[template_update\] frame=(\d+)', log)))
    assert checks == set(range(50, len(boxes), 50)), 'Template update schedule mismatch'
    return dict(sequence=name, frames=len(boxes), update_schedule_valid=True,
                updates_accepted=log.count(' success:'),
                updates_rejected=log.count(' rollback:'), updates_skipped=log.count(' skipped:'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.environ.get('DMATRACK_CHECKPOINT', str(ROOT/'checkpoints/DMATrack_ep0010.pth.tar')))
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--workers-per-gpu', type=int, default=3)
    parser.add_argument('--sequences', nargs='+')
    parser.add_argument('--reference', help='Optional existing results, opened only AFTER fresh inference')
    parser.add_argument('--strategy', choices=['always_rgbt', 'hard_switch'], default='always_rgbt')
    parser.add_argument('--score', type=float, default=.3)
    parser.add_argument('--iou', type=float, default=.2)
    parser.add_argument('--tau-rgb', type=float, default=.99)
    parser.add_argument('--tau-tir', type=float, default=.95)
    parser.add_argument('--verify-weight', action='store_true')
    args = parser.parse_args()
    if args.workers_per_gpu < 1 or len(set(args.gpus)) != len(args.gpus):
        raise ValueError('Invalid GPU/worker configuration')
    output = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    if args.verify_weight and digest.hexdigest() != EXPECTED_SHA:
        raise ValueError('The 76.11 reproduction requires the published original ep10 weight')
    from lib.test.evaluation import get_dataset
    dataset = get_dataset('lasher')
    if args.sequences:
        if not set(args.sequences).issubset({s.name for s in dataset}):
            raise ValueError('Unknown sequence')
        dataset = [s for s in dataset if s.name in args.sequences]
    output.mkdir(parents=True, exist_ok=False)
    (output/'results').mkdir()
    options = vars(args).copy()
    options.update(checkpoint=str(checkpoint), output=str(output))
    started = time.time()
    plan = dict(**options, checkpoint_sha256=digest.hexdigest(), template_update_interval=50,
                sequence_count=len(dataset), frames=sum(len(s.frames) for s in dataset))
    write_json(output/'plan.json', plan)
    try:
        rows = []
        write_json(output/'progress.json', dict(stage='inference', finished=0, total=len(dataset)))
        with mp.get_context('spawn').Pool(len(args.gpus)*args.workers_per_gpu,
                initialize_worker, (args.gpus, options)) as pool:
            for row in pool.imap_unordered(run_sequence,
                    [s.name for s in sorted(dataset, key=lambda s:len(s.frames), reverse=True)]):
                rows.append(row)
                progress = dict(stage='inference', finished=len(rows), total=len(dataset),
                                last_sequence=row['sequence'], elapsed_seconds=time.time()-started)
                write_json(output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        write_json(output/'interval_audit.json', dict(passed=True, sequences=rows))
        from tracking.evaluate import evaluate
        evaluation = evaluate(output/'results', [s.name for s in dataset])
        write_json(output/'results/evaluation.json', evaluation)
        if args.verify_weight and args.strategy == 'always_rgbt' and args.score == .3 and args.iou == .2:
            golden = json.loads((ROOT/'assets/reproduction_7611.json').read_text())
            checks = []
            for seq in dataset:
                for suffix in ('.txt', '_source_gate.txt', '_expert.txt'):
                    filename = seq.name+suffix
                    checksum = hashlib.sha256((output/'results'/filename).read_bytes()).hexdigest()
                    checks.append(dict(file=filename, identical=checksum == golden['files'][filename]))
            audit = dict(files=len(checks), identical_files=sum(c['identical'] for c in checks),
                         all_byte_identical=all(c['identical'] for c in checks), details=checks)
            write_json(output/'reproduction_audit.json', audit)
            if not audit['all_byte_identical']:
                raise RuntimeError('Fresh results differ from the verified 76.11 result; see reproduction_audit.json')
        if args.reference:
            reference = Path(args.reference)
            comparison = []
            for s in dataset:
                for suffix in ('.txt', '_source_gate.txt', '_expert.txt'):
                    filename = s.name+suffix
                    comparison.append(dict(file=filename, identical=
                        (reference/filename).read_bytes() == (output/'results'/filename).read_bytes()))
            write_json(output/'comparison.json', dict(files=len(comparison),
                       identical_files=sum(r['identical'] for r in comparison),
                       all_byte_identical=all(r['identical'] for r in comparison), details=comparison))
        summary = dict(stage='complete', metrics=evaluation['metrics'], sequences=len(dataset),
                       frames=evaluation['frames'], elapsed_seconds=time.time()-started)
        write_json(output/'progress.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException:
        write_json(output/'progress.json', dict(stage='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
