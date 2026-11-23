"""Ordered TBPTT, one update per complete-flight batch, no test sources."""
from dataclasses import fields
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from provenance import stable_hash, write_json
from simulink_sequence_data import collate, fit_scaler, source_receipt, validate_partitions
from simulink_sequence_model import build_model, loss_for_flights
from simulink_realtime import SEMANTICS
from simulink_realtime_evaluation import choose_policy, probability_sequences


def detach_memory(memory):
    if memory is None:
        return None
    def det(value):
        if value is None:
            return None
        return tuple(det(v) for v in value) if isinstance(value, tuple) else value.detach()
    return type(memory)(**{f.name: det(getattr(memory, f.name)) for f in fields(memory)})


def select_memory(memory, indices):
    if memory is None:
        return None
    data = {}
    for field in fields(memory):
        value = getattr(memory, field.name)
        if value is None:
            data[field.name] = None
        elif field.name == "hidden":
            data[field.name] = tuple(v.index_select(1, indices) for v in value)
        elif isinstance(value, tuple):
            data[field.name] = tuple(v.index_select(0, indices) for v in value)
        else:
            data[field.name] = value.index_select(0, indices)
    return type(memory)(**data)


def sample_weights(flights, steps, config, device):
    weights = np.zeros((len(flights), steps), dtype=np.float32)
    for i, f in enumerate(flights):
        n = len(f.x)
        weights[i, :n] = 1.
        if f.label:
            current, _ = f.targets()
            early = (f.time >= f.fault_start_s-1e-9) & (f.time < f.fault_start_s+config["early_span_s"]-1e-9) & (current > 0)
            weights[i, :n][early] = config["early_weight"]
    return torch.as_tensor(weights, device=device)


def flight_loss_scales(flights, config, device):
    """Per-flight balancing multipliers to counter post-onset 1:6 healthy vs mild-fault prior skew."""
    healthy_w = float(config.get("healthy_flight_weight", 1.0))
    mild_w = float(config.get("mild_flight_weight", 1.0))
    scales = []
    for f in flights:
        if f.label == 0:
            scales.append(healthy_w)
        elif 13 <= f.label <= 18:
            scales.append(mild_w)
        else:
            scales.append(1.0)
    return torch.as_tensor(scales, dtype=torch.float32, device=device)


def segmented_backward(model, x, lengths, current, seen, labels, weights, config,
                       phase_tags=None, flight_scales=None):
    """Returns point counts. Padding never advances a finished record's state.

    Parameters stay fixed until all chunks of this batch have been processed;
    detached memory carries numerical history but no cross-boundary gradients.
    Per-flight time weights sum to one, independent of chunk boundaries and length.
    """
    batch, steps = x.shape[:2]
    normalized = weights/weights.sum(1, keepdim=True)
    if flight_scales is None:
        flight_scales = torch.ones((batch,), dtype=weights.dtype, device=x.device)
    binary_w = float(config.get("binary_fault_loss_weight", 0.0))
    memory, previous = None, list(range(batch))
    stats = dict(loss=0., valid_forward_points=0, current_supervised_points=0, chunks=0,
                 record_terminals=0, weighted_mass=0.)
    phase_sums = {name:[0.,0] for name in ("healthy","early","sustained")}
    for start in range(0, steps, config["segment_samples"]):
        active = [i for i in previous if int(lengths[i]) > start]
        if not active:
            break
        if active != previous:
            memory = select_memory(memory, torch.as_tensor([previous.index(i) for i in active], device=x.device))
        ix = torch.as_tensor(active, device=x.device)
        end = min(start+config["segment_samples"], steps)
        lens = (lengths[ix]-start).clamp(max=end-start)
        output = model(x[ix, start:end], memory, lens)
        target_seg = current[ix, start:end]
        loss_map = F.cross_entropy(output["current_logits"].transpose(1, 2), target_seg, reduction="none")
        coeff = normalized[ix, start:end]
        scaled_coeff = coeff * flight_scales[ix, None]
        if phase_tags is not None:
            for phase,name in enumerate(phase_sums):
                mask = (phase_tags[ix,start:end] == phase) & (coeff > 0)
                phase_sums[name][0] += float(loss_map.detach()[mask].sum())
                phase_sums[name][1] += int(mask.sum())
        loss = config["current_loss_weight"]*(loss_map*scaled_coeff).sum()/batch
        if binary_w > 0.:
            log_p = F.log_softmax(output["current_logits"], dim=-1)
            log_p_healthy = log_p[..., 0]
            log_p_fault = torch.logsumexp(log_p[..., 1:], dim=-1)
            is_fault = (target_seg > 0).to(log_p.dtype)
            bin_loss = -(is_fault*log_p_fault + (1.-is_fault)*log_p_healthy)
            loss = loss + binary_w*(bin_loss*scaled_coeff).sum()/batch
        if config.get("prefix_loss_weight", 0):
            values = F.cross_entropy(output["prefix_logits"].transpose(1, 2), seen[ix, start:end], reduction="none")
            loss = loss+config["prefix_loss_weight"]*(values*scaled_coeff).sum()/batch
        terminal = lengths[ix] <= end
        if config.get("record_loss_weight", 0) and bool(terminal.any()):
            loss = loss+config["record_loss_weight"]*F.cross_entropy(output["record_logits"][terminal], labels[ix][terminal], reduction="sum")/batch
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite segmented loss")
        loss.backward()
        stats["loss"] += float(loss.detach())
        stats["valid_forward_points"] += int(lens.sum())
        stats["current_supervised_points"] += int(lens.sum())
        stats["record_terminals"] += int(terminal.sum())
        stats["weighted_mass"] += float(coeff.sum())/batch
        stats["chunks"] += 1
        memory, previous = detach_memory(output["memory"]), active
    stats["phase_sums"] = phase_sums
    return stats


def _save(path, payload):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_realtime_checkpoint(path, device="cpu", allow_legacy_control=False):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("semantics") != SEMANTICS:
        raise ValueError("Legacy checkpoint has no current-state-stream-v2 semantics. Use the legacy runner explicitly.")
    if saved["config"]["objective"] == "legacy_record" and not allow_legacy_control:
        raise ValueError("Record-primary experimental control requires explicit allow_legacy_control")
    model = build_model(saved["config"]).to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), saved["scaler"], saved


def train_job(config, flights, output, identity, device="cuda", resume=False):
    """No test flights accepted. Source/code identity must match to resume."""
    validate_partitions(flights)
    if any(f.role == "test" for f in flights):
        raise PermissionError("Test measurements cannot enter development training")
    train = [f for f in flights if f.role == "train"]
    validation = [f for f in flights if f.role == "validation"]
    if {f.label for f in train} != set(range(19)) or not validation:
        raise ValueError("Need all training classes and separate validation records")
    if min(config["segment_samples"], config["max_epochs"], config["batch_flights"]) < 1 or config["early_weight"] <= 0:
        raise ValueError("Invalid training geometry")
    output = Path(output)
    signature = stable_hash(dict(config=config, identity=identity))
    if output.exists() and any(output.iterdir()):
        if not resume or not (output/"last.pt").exists():
            raise FileExistsError("Use a new job directory or identical --resume")
        last = torch.load(output/"last.pt", map_location="cpu", weights_only=False)
        if last["signature"] != signature:
            raise ValueError("Config/code/source identity changed")
        if (output/"summary.json").exists():
            import json
            return json.loads((output/"summary.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(config["seed"])
    model = build_model(config).to(device)
    initial_hash = stable_hash({k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()})
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["max_epochs"],
            eta_min=config["learning_rate"] * float(config.get("min_lr_ratio", 0.1)),
        )
        if config.get("use_cosine_lr", False) else None
    )
    scaler = fit_scaler(
        train,
        feature_mode=config.get("feature_mode", "raw12"),
        scaling_mode=config.get("scaling_mode", "standard"),
    )
    history, best_rank, updates, first_epoch, best_epoch = [], None, 0, 1, None
    best_policy = None
    wall_before = 0.
    if resume and (output/"last.pt").exists():
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        for state in optimizer.state.values():
            for name, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[name] = val.to(device)
        if scheduler is not None and last.get("scheduler") is not None:
            scheduler.load_state_dict(last["scheduler"])
        scaler = last["scaler"]
        history, updates = last["history"], last["updates"]
        first_epoch, best_rank = last["epoch"]+1, tuple(last["best_rank"]) if last["best_rank"] is not None else None
        best_epoch, best_policy = last["best_epoch"], last["best_policy"]
        rng.bit_generator.state = last["sampler_rng"]
        torch.set_rng_state(last["torch_rng"].cpu())
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in last["cuda_rng"]])
        wall_before = last["elapsed_s"]
    write_json(output/"job.json", dict(signature=signature, config=config, identity=identity,
                                      initial_parameters_sha256=initial_hash, semantics=SEMANTICS))
    started = time.perf_counter()
    for epoch in range(first_epoch, config["max_epochs"]+1):
        epoch_start = time.perf_counter()
        order = rng.permutation(len(train))
        stats = []
        max_gradient = 0.
        model.train()
        for start in range(0, len(order), config["batch_flights"]):
            selected = [train[i] for i in order[start:start+config["batch_flights"]]]
            x, lengths, current, seen, labels = collate(selected, scaler, device)
            optimizer.zero_grad(set_to_none=True)
            if config["objective"] == "current_state":
                weights = sample_weights(selected, x.shape[1], config, device)
                f_scales = flight_loss_scales(selected, config, device)
                tags = torch.full(current.shape,-1,dtype=torch.long,device=device)
                for i,flight in enumerate(selected):
                    target,_ = flight.targets()
                    phase = np.zeros(len(target),dtype=np.int64)
                    phase[target>0] = 2
                    if flight.label:
                        phase[(target>0)&(flight.time<flight.fault_start_s+config.get("phase_reporting_span_s",3.)-1e-9)] = 1
                    tags[i,:len(target)] = torch.as_tensor(phase,device=device)
                row = segmented_backward(model, x, lengths, current, seen, labels, weights, config, tags, f_scales)
            elif config["objective"] == "legacy_record":
                output_full = model(x, lengths=lengths)
                weights = config["legacy_control_loss_weights"]
                loss, parts = loss_for_flights(output_full, lengths, current, seen, labels,
                    period=config["decision_samples"], record_weight=weights["record"],
                    prefix_weight=weights["prefix"], current_weight=weights["current"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite legacy-control loss")
                loss.backward()
                row = dict(loss=float(loss.detach()), valid_forward_points=int(lengths.sum()),
                    current_supervised_points=sum(int((int(n)-1)//config["decision_samples"]+1) for n in lengths),
                    chunks=1, weighted_mass=1., record_terminals=len(selected), **parts)
            else:
                raise ValueError("Unknown training objective")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            max_gradient = max(max_gradient, float(norm))
            optimizer.step()
            updates += 1
            stats.append(row)
        if scheduler is not None:
            scheduler.step()
        row = dict(epoch=epoch, optimizer_updates=updates, lr=float(optimizer.param_groups[0]["lr"]),
            loss=float(np.mean([r["loss"] for r in stats])),
            forward_points=sum(r["valid_forward_points"] for r in stats),
            current_supervised_points=sum(r["current_supervised_points"] for r in stats),
            compute_chunks=sum(r["chunks"] for r in stats),
            record_terminals=sum(r["record_terminals"] for r in stats),
            max_preclip_gradient_norm=max_gradient, training_wall_s=time.perf_counter()-epoch_start)
        if config["objective"] == "current_state":
            row["phase_training_CE"] = {name: sum(r["phase_sums"][name][0] for r in stats)/max(1,sum(r["phase_sums"][name][1] for r in stats))
                                        for name in ("healthy","early","sustained")}
        if row["forward_points"] != sum(len(f.x) for f in train) or row["record_terminals"] != len(train):
            raise AssertionError("Training dropped or repeated valid points/record endpoints")
        improved = False
        if epoch % config["validation_every_epochs"] == 0 or epoch == config["max_epochs"]:
            probabilities = probability_sequences(model, scaler, validation, device, config["batch_flights"])
            selection = choose_policy(validation, probabilities, config["policy_candidates"], config["decision_samples"])
            rank = tuple(selection["selected"]["rank"])
            row["validation"] = selection
            improved = best_rank is None or rank > best_rank
            if improved:
                best_rank, best_epoch, best_policy = rank, epoch, selection["selected"]["policy"]
            write_json(output/f"validation_epoch_{epoch:03d}.json", selection)
        row["epoch_wall_s"] = time.perf_counter()-epoch_start
        history.append(row)
        payload = dict(semantics=SEMANTICS, signature=signature, config=config, model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            scaler=scaler, epoch=epoch, updates=updates, history=history,
            best_rank=best_rank, best_epoch=best_epoch, best_policy=best_policy,
            initial_parameters_sha256=initial_hash,
            sampler_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
            elapsed_s=wall_before+time.perf_counter()-started)
        if improved:
            _save(output/"best.pt", payload)
        _save(output/"last.pt", payload)
        write_json(output/"history.json", history)
        val_text = ""
        if "validation" in row:
            vm = row["validation"]["selected"]["metrics"]
            m80 = vm.get("by_severity", {}).get("partial_80pct", {})
            val_text = (
                f" stateF1={row['validation']['states']['all']['macro_f1']:.4f}"
                f" loc1={vm['localization_within_1s']:.4f}"
                f" mild80_det1={(m80.get('detection_within_1s') or 0.):.4f}"
                f" mild80_loc1={(m80.get('localization_within_1s') or 0.):.4f}"
                f" hFA={vm['healthy_false_alarm_events']}"
            )
        print(f"REALTIME_EPOCH {config['model']} {config['objective']} seed={config['seed']} epoch={epoch} updates={updates} loss={row['loss']:.5f} seconds={row['epoch_wall_s']:.2f}{val_text}", flush=True)
        model.train()
    best = torch.load(output/"best.pt", map_location="cpu", weights_only=False)
    train_source = source_receipt(train)
    summary = dict(status="completed", signature=signature, semantics=SEMANTICS,
        model=config["model"], objective=config["objective"], seed=config["seed"],
        train_records=len(train), validation_records=len(validation),
        train_points_per_traversal=sum(len(f.x) for f in train),
        record_traversals=len(history), record_presentations=len(train)*len(history),
        total_valid_forward_points=sum(r["forward_points"] for r in history),
        total_current_supervised_points=sum(r["current_supervised_points"] for r in history),
        optimizer_updates=updates, compute_chunks=sum(r["compute_chunks"] for r in history),
        selected_epoch=best_epoch, selected_policy=best_policy,
        selected_validation=best["history"][-1]["validation"],
        parameters=sum(p.numel() for p in model.parameters()),
        train_validation_wall_s=wall_before+time.perf_counter()-started,
        scaler_fit_points=int(scaler.n_samples_seen_), scaler_fit_sources=[f.source_id for f in train],
        train_measured_identity=stable_hash(train_source), initial_parameters_sha256=initial_hash,
        GAN_used=False, ROS_used=False, test_used=False)
    write_json(output/"summary.json", summary)
    return summary