"""Opt-in training benchmarks for the existing Hugging Face Trainer loop."""
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


class TokenCountingCollator:
    """Counts batch tokens as delivered by the DataLoader (may include prefetch)."""
    def __init__(self, collate):
        self.collate = collate
        self.total_input_tokens = 0
        self.total_nonpadding_tokens = 0

    def __call__(self, rows):
        batch = self.collate(rows)
        self.total_input_tokens += int(batch['input_ids'].numel())
        self.total_nonpadding_tokens += int(batch['attention_mask'].sum().item())
        return batch


class BenchmarkCallback(TrainerCallback):
    """Times optimizer steps, with CUDA synchronization at step boundaries.

    Evaluation and checkpoint saving run between optimizer steps and used to be
    silently charged to the following step and to wall time. instrument_trainer
    reports them separately, so step times are pure training time and the
    wall-time breakdown shows where non-training time went.
    """
    def __init__(self, output_file, warmup_steps=10, token_counter=None):
        self.output_file = Path(output_file)
        self.warmup_steps = warmup_steps
        self.token_counter = token_counter
        self.step_times = []
        self.started = None
        self.last_step = None
        self.step_records = []
        self.phase_seconds = {'evaluation': 0.0, 'checkpoint_save': 0.0}
        self.phase_calls = {'evaluation': 0, 'checkpoint_save': 0}
        self._phase_since_step = 0.0
        self._last_tokens = 0

    def add_phase(self, name, seconds):
        self.phase_seconds[name] += seconds
        self.phase_calls[name] += 1
        self._phase_since_step += seconds

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        self.last_step = self.started
        self.step_times = []
        self.step_records = []
        self._phase_since_step = 0.0

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        duration = now - self.last_step - self._phase_since_step
        self._phase_since_step = 0.0
        self.step_times.append(duration)
        record = {'step': int(state.global_step), 'seconds': duration}
        if self.token_counter is not None:
            tokens = self.token_counter.total_nonpadding_tokens
            record['collated_nonpadding_tokens'] = tokens - self._last_tokens
            self._last_tokens = tokens
        self.step_records.append(record)
        self.last_step = now

    def on_train_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        warm = self.step_times[:self.warmup_steps]
        steady = self.step_times[self.warmup_steps:]
        steady_seconds = sum(steady)
        phases = sum(self.phase_seconds.values())
        output = {
            'optimizer_steps': len(self.step_times),
            'training_wall_seconds': elapsed,
            'warmup_steps_excluded': min(self.warmup_steps, len(self.step_times)),
            'median_steady_step_seconds': statistics.median(steady) if steady else None,
            'steady_optimizer_steps_per_second': len(steady) / steady_seconds if steady_seconds else None,
            'wall_time_breakdown_seconds': {
                'warmup_steps': sum(warm),
                'steady_steps': steady_seconds,
                'evaluation': self.phase_seconds['evaluation'],
                'checkpoint_save': self.phase_seconds['checkpoint_save'],
                'other': elapsed - sum(self.step_times) - phases,
            },
            'evaluation_calls': self.phase_calls['evaluation'],
            'checkpoint_save_calls': self.phase_calls['checkpoint_save'],
            'warmup_step_seconds': [round(t, 2) for t in warm],
            'peak_pytorch_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_pytorch_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            'step_timings': self.step_records,
            'notes': [
                'Step timings include gradient accumulation and Trainer overhead, excluding evaluation and checkpoint saving.',
                'PyTorch peak excludes allocations outside its CUDA caching allocator.',
                'Token counts from the collator may include prefetched batches; do not interpret as exact consumed tokens.',
                'Wall time includes any initialization inside Trainer.train and callback overhead.',
            ],
        }
        if self.token_counter is not None:
            output['collated_input_tokens_including_prefetch'] = self.token_counter.total_input_tokens
            output['collated_nonpadding_tokens_including_prefetch'] = self.token_counter.total_nonpadding_tokens
            steady_tokens = sum(r.get('collated_nonpadding_tokens', 0) for r in self.step_records[self.warmup_steps:])
            output['approx_steady_nonpadding_tokens_per_second'] = steady_tokens / steady_seconds if steady_seconds else None
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(json.dumps(output, indent=2) + '\n')
        print('\n===== CQ TRAINING BENCHMARK =====')
        print(json.dumps({k: v for k, v in output.items() if k != 'step_timings'}, indent=2))


def instrument_trainer(trainer, callback):
    """Time Trainer.evaluate and checkpoint saving into callback phases."""
    def timed(name, fn):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                callback.add_phase(name, time.perf_counter() - t0)
        return wrapper
    trainer.evaluate = timed('evaluation', trainer.evaluate)
    if hasattr(trainer, '_save_checkpoint'):
        trainer._save_checkpoint = timed('checkpoint_save', trainer._save_checkpoint)
    return trainer
