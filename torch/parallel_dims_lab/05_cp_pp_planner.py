"""Plan CP token ranges, PP layer ranges, and a simple fill schedule."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from parallel_dims_common import seq_len_divisor


def split_ranges(length: int, parts: int) -> list[tuple[int, int]]:
    """Split ``range(length)`` into contiguous, nearly equal half-open ranges."""

    if length < 1 or parts < 1:
        raise ValueError("length and parts must be positive")
    base, remainder = divmod(length, parts)
    ranges = []
    start = 0
    for part in range(parts):
        size = base + (part < remainder)
        ranges.append((start, start + size))
        start += size
    return ranges


def context_parallel_ranges(seq_len: int, cp: int) -> list[tuple[int, int]]:
    """Return the token range assigned to each CP rank."""

    return split_ranges(seq_len, cp)


@dataclass(frozen=True)
class PipelineStage:
    """A PP stage's zero-based, half-open layer interval."""

    stage: int
    start: int
    end: int

    @property
    def layer_count(self) -> int:
        return self.end - self.start


def pipeline_stages(num_layers: int, pp: int) -> list[PipelineStage]:
    """Distribute layers across stages, giving early stages remainders."""

    return [
        PipelineStage(stage, start, end)
        for stage, (start, end) in enumerate(split_ranges(num_layers, pp))
    ]


def pipeline_schedule(pp: int, num_microbatches: int) -> list[list[tuple[int, int]]]:
    """Build a forward-only fill schedule for a teaching example.

    At time ``t``, stage ``s`` handles microbatch ``t - s`` when that index is
    valid.  This shows the pipeline fill order without claiming to implement a
    production 1F1B or interleaved schedule.
    """

    if pp < 1 or num_microbatches < 1:
        raise ValueError("pp and num_microbatches must be positive")
    schedule = []
    for time in range(num_microbatches + pp - 1):
        events = []
        for stage in range(pp):
            microbatch = time - stage
            if 0 <= microbatch < num_microbatches:
                events.append((stage, microbatch))
        schedule.append(events)
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--num-microbatches", type=int, default=8)
    args = parser.parse_args()

    token_ranges = context_parallel_ranges(args.seq_len, args.cp)
    divisor = seq_len_divisor(args.tp, args.cp)
    stages = pipeline_stages(args.num_layers, args.pp)

    print(f"CP token ranges (half-open): {token_ranges}")
    print(f"tokens per CP rank: {[end - start for start, end in token_ranges]}")
    print(f"seq_len_divisor = tp x (2 x cp) = {divisor}")
    print(
        f"sequence check: {args.seq_len} % {divisor} = "
        f"{args.seq_len % divisor} -> {'VALID' if args.seq_len % divisor == 0 else 'ERROR'}"
    )
    print("PP stages:")
    for stage in stages:
        print(
            f"  stage {stage.stage}: layers [{stage.start}, {stage.end - 1}] "
            f"({stage.layer_count} layers)"
        )
    print("pipeline fill schedule:")
    for time, events in enumerate(pipeline_schedule(args.pp, args.num_microbatches)):
        text = ", ".join(
            f"stage{stage} -> microbatch{microbatch}" for stage, microbatch in events
        )
        print(f"  time {time}: {text}")


if __name__ == "__main__":
    main()
