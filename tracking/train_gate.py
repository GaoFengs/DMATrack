"""Train only freshly initialized score branches; torch.distributed.run supported."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from lib.test.parameter.ckd_repro_7578 import parameters
from lib.train_gate.model import ScoreOnlyModel, SCORES, frozen_digest, is_score, aligned_iou
from lib.train_gate.data import make_sampler, collate_pairs

def write_json(path, obj):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(obj, indent=2))
    temporary.replace(path)

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

def load_batch(batch):
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    z, x = batch['template_images'], batch['search_images']
    return [z[:, :3], z[:, 3:]], [x[:, :3], x[:, 3:]], batch['template_anno'], batch['search_anno']

def smoke(model, batch, optimizer):
    inputs = load_batch(batch)
    frozen_before = frozen_digest(model.network)
    score_before = {n: p.detach().clone() for n, p in model.network.named_parameters() if is_score(n)}

    with torch.no_grad():
        verify_z, verify_x = [[v[:1] for v in group] for group in inputs[:2]]
        _, _, boxes = model.tracking_outputs(verify_z, verify_x)
        original = model.network(template=verify_z, search=verify_x, gt_bboxs_template=inputs[2][:1],
                                 inference_strategy='hard_switch', return_analysis=True)['analysis']
        errors = [float((box['pred_boxes']-original[key]['pred_boxes']).abs().max())
                  for box, key in zip(boxes, ('rgb_out', 'tir_out', 'rgbt_out'))]
        assert max(errors) == 0, errors
    loss, logits, labels, iou = model(*inputs)
    assert torch.equal(labels.bool(), model.make_labels(iou))
    boundary = torch.tensor([model.threshold-1e-4, model.threshold, model.threshold+1e-4], device=iou.device)
    expected = [False, True, True] if model.positive_boundary == 'ge' else [False, False, True]
    assert model.make_labels(boundary).tolist() == expected
    branch_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction='none').mean(0)
    expected_loss = branch_losses.sum() if model.branch_reduction == 'sum' else branch_losses.mean()
    assert torch.allclose(loss.detach(), expected_loss)
    loss.backward()
    gradients = {name: sum(float(p.grad.abs().sum()) for p in getattr(model.network, name).parameters() if p.grad is not None) for name in SCORES}
    assert all(value > 0 and np.isfinite(value) for value in gradients.values()), gradients
    assert all(p.grad is None for n, p in model.network.named_parameters() if not is_score(n))
    optimizer.step()
    assert frozen_digest(model.network) == frozen_before == model.initial_frozen_digest
    changed = [name for name, p in model.network.named_parameters() if is_score(name) and not torch.equal(p, score_before[name])]
    assert all(any(n.startswith(branch+'.') for n in changed) for branch in SCORES)

    unit = torch.tensor([[.5, .5, 1., 1.], [2., 2., 1., 1.]])
    assert torch.equal(aligned_iou(unit, torch.tensor([[0., 0., 1., 1.]]).repeat(2, 1)), torch.tensor([1., 0.]))
    input_digest = hashlib.sha256()
    for key in sorted(batch):
        input_digest.update(batch[key].numpy().tobytes())
    return dict(passed=True, fusion=model.fusion, loss=float(loss), frozen_sha256=frozen_before,
                branch_reduction=model.branch_reduction, positive_boundary=model.positive_boundary,
                label_threshold=model.threshold,
                boundary_labels=model.make_labels(boundary).tolist(), branch_losses=branch_losses.tolist(),
                input_sha256=input_digest.hexdigest(), initial_score_sha256=model.initial_score_digests,
                trainable_parameters={n:sum(p.numel() for p in getattr(model.network,n).parameters()) for n in SCORES},
                inference_box_max_abs_errors=errors, score_gradients=gradients,
                changed_score_tensors=len(changed), labels=labels.tolist(), iou=iou.tolist())

@torch.no_grad()
def validate(model, loader, world):
    model.eval()
    totals = torch.zeros(8, device='cuda', dtype=torch.float64)
    for batch in loader:
        with torch.cuda.amp.autocast(enabled=True):
            loss, logits, labels, _ = model(*load_batch(batch))
        totals[0] += loss.double()
        totals[1:4] += labels.double().mean(0)
        totals[4:7] += ((logits.sigmoid() > .5) == labels.bool()).double().mean(0)
        totals[7] += 1
    if world > 1:
        dist.all_reduce(totals)
    return dict(loss=float(totals[0]/totals[7]), positive_ratio=(totals[1:4]/totals[7]).tolist(),
                binary_accuracy=(totals[4:7]/totals[7]).tolist(),
                note='Sampled-pair classification validation, not tracking Precision')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/gate_iou08_retrain')
    parser.add_argument('--checkpoint', default=os.environ.get('DMATRACK_CHECKPOINT', 'checkpoints/DMATrack_ep0010.pth.tar'))
    parser.add_argument('--dataset', default=os.environ.get('LASHER_ROOT', 'data/LasHeR'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--fusion', choices=['sum', 'concat'], default='sum')
    parser.add_argument('--branch-reduction', choices=['mean', 'sum'], default='sum')
    parser.add_argument('--positive-boundary', choices=['gt', 'ge'], default='ge')
    parser.add_argument('--label-threshold', type=float, default=.5)
    parser.add_argument('--lr-drop-after', type=int, default=7,
                        help='Completed epochs before LR decay; milestone 16 means epoch 17 starts lower')
    parser.add_argument('--lr-schedule', choices=['step', 'single'], default='step')
    parser.add_argument('--batch-size', type=int, default=128, help='Global batch across all GPUs')
    parser.add_argument('--micro-batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2, help='Workers per GPU')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-batch-size', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    rank, world, local = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1)), int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(24)
    np.random.seed(24+rank)
    random.seed(24+rank)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.stdout = (output/('rank_%d.log'%rank)).open('a', buffering=1)
    sys.stderr = sys.stdout
    params = parameters('CKD_ba_rgbt_base_gate', args.checkpoint)
    cfg = params.cfg
    if args.epochs <= 0:
        raise ValueError('Epoch count must be positive')
    if args.batch_size % (world*args.micro_batch):
        raise ValueError('Global batch must be divisible by world size * micro batch')
    accumulation = args.batch_size // (world*args.micro_batch)
    model = ScoreOnlyModel(cfg, args.checkpoint, reset_scores=True, fusion=args.fusion,
                           branch_reduction=args.branch_reduction,
                           positive_boundary=args.positive_boundary, threshold=args.label_threshold).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg.TRAIN.LEARN.LR, weight_decay=cfg.TRAIN.LEARN.WEIGHT_DECAY)
    if args.lr_drop_after <= 0:
        raise ValueError('LR decay epoch must be positive')
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_drop_after], gamma=.1)
                 if args.lr_schedule == 'single' else
                 torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_after, gamma=.1))
    if args.smoke:
        names = (Path(args.dataset)/'trainingsetList.txt').read_text().splitlines()[:2]

        torch.manual_seed(24)
        np.random.seed(24)
        random.seed(24)
        sampler = make_sampler(cfg, args.dataset, 'train', samples=args.smoke_batch_size, sequence_names=names)
        batch = collate_pairs([sampler[i] for i in range(args.smoke_batch_size)])
        report = smoke(model, batch, optimizer)
        write_json(output/'smoke.json', report)
        print(json.dumps(report), flush=True)
        return
    train_set = make_sampler(cfg, args.dataset, 'train')
    train_names = set(train_set.datasets[0].sequence_list)
    test_names = set((Path(args.dataset)/'testingsetList.txt').read_text().splitlines())
    assert not train_names & test_names, 'Training and test sequences must not overlap'
    training_sampler = DistributedSampler(train_set, world, rank, seed=24, drop_last=True)
    def loader(dataset, sampler):
        generator = torch.Generator()
        generator.manual_seed(24+1000*rank+(100 if dataset is not train_set else 0))
        return DataLoader(dataset, batch_size=args.micro_batch, sampler=sampler, drop_last=True,
                          num_workers=args.workers, collate_fn=collate_pairs, pin_memory=True,
                          worker_init_fn=seed_worker, multiprocessing_context='spawn' if args.workers else None,
                          persistent_workers=args.workers > 0, generator=generator)
    train_loader = loader(train_set, training_sampler)
    val_set = make_sampler(cfg, args.dataset, 'test')
    val_sampler = DistributedSampler(val_set, world, rank, seed=24, shuffle=False, drop_last=True)
    val_loader = loader(val_set, val_sampler)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    start_epoch = 1
    if args.resume:
        latest = sorted(output.glob('OSTrack_CKD_ep*.pth.tar'))[-1]
        checkpoint = torch.load(latest, map_location='cpu')
        old_plan = checkpoint['training_plan']
        assert old_plan.get('rgbt_score_fusion', 'sum') == args.fusion
        assert old_plan['epochs'] == args.epochs, 'Resume only the same experiment plan'
        assert old_plan.get('branch_reduction', 'mean') == args.branch_reduction
        assert old_plan.get('positive_boundary', 'gt') == args.positive_boundary
        assert old_plan.get('label_threshold', .8) == args.label_threshold
        assert old_plan.get('lr_schedule', 'step') == args.lr_schedule
        assert old_plan['lr_drop_epoch'] == args.lr_drop_after
        assert old_plan['global_batch'] == args.batch_size and old_plan['world_size'] == world
        model.network.load_state_dict(checkpoint['net'], strict=True)
        assert checkpoint['frozen_sha256'] == frozen_digest(model.network) == model.initial_frozen_digest
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']+1
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'])
    elif any(output.glob('OSTrack_CKD_ep*.pth.tar')):
        raise FileExistsError('Existing training checkpoints; use --resume or a new directory')
    distributed = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    manifest = dict(checkpoint=args.checkpoint, reset_score_branches=list(SCORES),
                    rgbt_score_fusion=args.fusion,
                    rgbt_score_input_dim=1536 if args.fusion=='concat' else 768,
                    initial_score_sha256=model.initial_score_digests,
                    trainable_parameters={n:sum(p.numel() for p in getattr(model.network,n).parameters()) for n in SCORES},
                    label='IoU %s %g' % ('>=' if args.positive_boundary == 'ge' else '>', args.label_threshold),
                    label_threshold=args.label_threshold,
                    loss='BCEWithLogitsLoss batch mean per branch; branch '+args.branch_reduction,
                    positive_boundary=args.positive_boundary, branch_reduction=args.branch_reduction,
                    epochs=args.epochs, global_batch=args.batch_size, micro_batch=args.micro_batch,
                    world_size=world, accumulation=accumulation, learning_rate=cfg.TRAIN.LEARN.LR,
                    weight_decay=cfg.TRAIN.LEARN.WEIGHT_DECAY, lr_drop_epoch=args.lr_drop_after,
                    lr_schedule=args.lr_schedule, lr_gamma=.1,
                    samples_per_epoch=len(train_set), training_sequences=len(train_names),
                    validation_samples=len(val_set), validation_epoch_interval=cfg.TRAIN.TRAINER.VAL_EPOCH_INTERVAL,
                    test_sequences=len(test_names), splits_disjoint=True, seed=24,
                    dataloader_seed_rule='24 + 1000*rank (+100 for validation)',
                    frozen_sha256=model.initial_frozen_digest, amp=True,
                    frozen_forward='float32, eval, no_grad', gpu_visibility=os.getenv('CUDA_VISIBLE_DEVICES'))
    if rank == 0:
        write_json(output/'training_plan.json', manifest)
        write_json(output/'data_audit.json', dict(policy='Original frame IDs; exclude only unavailable pairs',
                   training_sequences=len(train_names),
                   missing_frames=train_set.datasets[0].missing_frame_audit))
    for epoch in range(start_epoch, args.epochs+1):
        epoch_lr = optimizer.param_groups[0]['lr']
        model.train()
        training_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(8, device='cuda', dtype=torch.float64)
        started = time.time()
        usable = len(train_loader)//accumulation*accumulation
        for step, batch in enumerate(train_loader):
            if step >= usable:
                break
            if epoch == 1 and step == 0:
                digest = hashlib.sha256()
                for key in sorted(batch):
                    digest.update(batch[key].numpy().tobytes())
                hashes = [digest.hexdigest()]
                if world > 1:
                    hashes = [None]*world
                    dist.all_gather_object(hashes, digest.hexdigest())
                assert len(set(hashes)) == world, 'Duplicate random training batches across ranks'
                if rank == 0:
                    write_json(output/'first_batch_audit.json', dict(rank_hashes=hashes, all_distinct=True))
            with torch.cuda.amp.autocast(enabled=True):
                loss, logits, labels, iou = distributed(*load_batch(batch))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite classification loss')
            scaler.scale(loss/accumulation).backward()
            if (step+1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], .1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            totals[0] += loss.detach().double()
            totals[1:4] += labels.double().mean(0)
            totals[4:7] += ((logits.sigmoid() > .5) == labels.bool()).double().mean(0)
            totals[7] += 1
            if rank == 0 and (step % 25 == 0 or step+1 == usable):
                progress = dict(stage='training', epoch=epoch, epochs=args.epochs, micro_step=step+1,
                                micro_steps=usable, loss=float(loss), lr=optimizer.param_groups[0]['lr'],
                                elapsed_seconds=time.time()-started)
                write_json(output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        if world > 1:
            dist.all_reduce(totals)
        validation = validate(model, val_loader, world) if epoch % cfg.TRAIN.TRAINER.VAL_EPOCH_INTERVAL == 0 else None
        assert frozen_digest(model.network) == model.initial_frozen_digest, 'Frozen network changed'
        scheduler.step()
        if rank == 0:
            record = dict(epoch=epoch, learning_rate=epoch_lr, next_learning_rate=optimizer.param_groups[0]['lr'],
                          loss=float(totals[0]/totals[7]),
                          positive_ratio=(totals[1:4]/totals[7]).tolist(),
                          binary_accuracy=(totals[4:7]/totals[7]).tolist(),
                          elapsed_seconds=time.time()-started, frozen_unchanged=True)
            if validation is not None:
                record['validation'] = validation
            write_json(output/('epoch_%02d.json'%epoch), record)
            checkpoint = dict(net={k:v.detach().cpu() for k,v in model.network.state_dict().items()},
                              optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                              scaler=scaler.state_dict(), epoch=epoch, training_plan=manifest,
                              frozen_sha256=model.initial_frozen_digest, torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state())
            destination = output/('OSTrack_CKD_ep%04d.pth.tar'%epoch)
            temporary = destination.with_suffix('.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(destination)
            print(json.dumps(record), flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        write_json(output/'progress.json', dict(stage='training_complete', epochs=args.epochs))
    if world > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
