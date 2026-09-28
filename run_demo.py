"""Small, honest single-step CBF-LLM mechanism demonstration.

Uses a small causal LM and the sentiment classifier from the paper. The
finite candidate-search budget is intentional and reported in outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent


def display_model_path(model_path: Path) -> str:
    try:
        return str(model_path.resolve().relative_to(ROOT))
    except ValueError:
        return str(model_path)


DEFAULT_PROMPTS = [
    "I really appreciate your help, but",
    "Today was a wonderful day, although",
    "I love this idea, but",
    "You did a great job, even though",
    "I feel hopeful about the future, but",
    "Thank you for trying; however",
]


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def label_indices(config) -> tuple[int, int, int]:
    labels = {str(label).lower(): int(index) for index, label in config.id2label.items()}
    selected = {}
    for target in ("negative", "neutral", "positive"):
        matches = [index for label, index in labels.items() if target in label]
        if len(matches) != 1:
            raise ValueError(f"Cannot identify unique {target} label from {config.id2label}")
        selected[target] = matches[0]
    return selected["negative"], selected["neutral"], selected["positive"]


class SentimentBarrier:
    def __init__(self, model_dir: Path, device: str, batch_size: int = 16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        ).to(device).eval()
        self.neg, self.neu, self.pos = label_indices(self.model.config)
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def score_many(self, texts: list[str]) -> list[float]:
        results: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            enc = {key: value.to(self.device) for key, value in enc.items()}
            probabilities = self.model(**enc).logits.float().softmax(dim=-1)
            h = probabilities[:, self.pos] - torch.maximum(
                probabilities[:, self.neg], probabilities[:, self.neu]
            )
            results.extend(float(value) for value in h.cpu())
        return results

    def score(self, text: str) -> float:
        return self.score_many([text])[0]


class Generator:
    def __init__(self, model_dir: Path, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True
        ).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def next_logits(self, ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        return self.model(input_ids=x).logits[0, -1].float().cpu()

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def sample_by_original_logits(
    candidates: list[int], logits: torch.Tensor, rng: torch.Generator
) -> int:
    selected_logits = logits[torch.tensor(candidates, dtype=torch.long)]
    relative_probabilities = torch.softmax(selected_logits, dim=0)
    chosen_position = int(torch.multinomial(relative_probabilities, 1, generator=rng))
    return candidates[chosen_position]


def run_once(
    generator: Generator,
    barrier: SentimentBarrier,
    prompt: str,
    mode: str,
    gamma: float,
    seed: int,
    max_new_tokens: int,
    safe_k: int,
    search_limit: int,
) -> dict:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ids = generator.tokenizer.encode(prompt, add_special_tokens=False)
    h_current = barrier.score(generator.decode(ids))
    if h_current < 0:
        raise ValueError(f"Initial prompt must have h >= 0; got {h_current:.4f}")
    trajectory = [h_current]
    steps = []
    started = time.perf_counter()
    stop_reason = "max_new_tokens"

    for step_number in range(1, max_new_tokens + 1):
        logits = generator.next_logits(ids)
        ranked = torch.topk(logits, min(search_limit, logits.numel())).indices.tolist()
        rejected = 0
        inspected = 0

        if mode == "baseline":
            allowed = ranked[:safe_k]
        else:
            candidate_texts = [generator.decode(ids + [candidate]) for candidate in ranked]
            candidate_scores = barrier.score_many(candidate_texts)
            allowed = []
            threshold = gamma * h_current
            for candidate, next_h in zip(ranked, candidate_scores):
                inspected += 1
                if next_h + 1e-7 >= threshold:
                    allowed.append(candidate)
                    if len(allowed) == safe_k:
                        break
                else:
                    rejected += 1
            if not allowed:
                stop_reason = "no_feasible_candidate_within_search_limit"
                break

        chosen = sample_by_original_logits(allowed, logits, rng)
        if chosen == generator.tokenizer.eos_token_id:
            stop_reason = "eos_token"
            break
        ids.append(chosen)
        full_text = generator.decode(ids)
        new_h = barrier.score(full_text)
        if mode == "cbf" and new_h + 1e-6 < gamma * h_current:
            raise AssertionError(
                f"CBF inequality failed at step {step_number}: "
                f"new h={new_h:.6f}, gamma*h={gamma * h_current:.6f}"
            )
        token_text = generator.tokenizer.decode([chosen], skip_special_tokens=True)
        steps.append(
            {
                "step": step_number,
                "token_id": chosen,
                "token_text": token_text,
                "h_before": h_current,
                "h_after": new_h,
                "threshold": gamma * h_current if mode == "cbf" else None,
                "rejected_candidates": rejected,
                "inspected_candidates": inspected,
                "allowed_candidates": len(allowed),
            }
        )
        trajectory.append(new_h)
        h_current = new_h

    return {
        "mode": mode,
        "gamma": gamma if mode == "cbf" else None,
        "prompt": prompt,
        "seed": seed,
        "text": generator.decode(ids),
        "generated_text": generator.decode(ids)[len(prompt) :],
        "h_trajectory": trajectory,
        "min_h": min(trajectory),
        "final_h": trajectory[-1],
        "rejected_total": sum(step["rejected_candidates"] for step in steps),
        "inspected_total": sum(step["inspected_candidates"] for step in steps),
        "generated_tokens": len(steps),
        "stop_reason": stop_reason,
        "seconds": time.perf_counter() - started,
        "steps": steps,
    }


def plot_trajectories(runs: list[dict], output_path: Path) -> None:
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9.5, 5.3), dpi=160)
    palette = {"baseline": "#303030", "0.0": "#2F80B5", "0.4": "#779B22"}
    for run in runs:
        key = "baseline" if run["mode"] == "baseline" else f'{run["gamma"]:.1f}'
        label = "No intervention" if key == "baseline" else f"CBF gamma={key}"
        values = run["h_trajectory"]
        ax.plot(range(len(values)), values, marker="o", markersize=3.5, linewidth=2, label=label, color=palette.get(key))
    ax.axhline(0, color="#C94A34", linestyle="--", linewidth=1.5, label="h = 0 boundary")
    ax.set_xlabel("Generated token step")
    ax.set_ylabel("Language constraint score h(x)")
    ax.set_title("Single-step CBF-LLM mechanism demo")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_outputs(runs: list[dict], metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "gamma", "step", "h"])
        for run in runs:
            for step, h in enumerate(run["h_trajectory"]):
                writer.writerow([run["mode"], run["gamma"], step, f"{h:.7f}"])
    plot_trajectories(runs, output_dir / "h_trajectory.png")

    lines = [
        "# 小模型单步 CBF-LLM 实验结果",
        "",
        "> 这是机制级演示。生成模型、数据量和有限候选搜索与论文正式实验不同。",
        "",
        f'- 生成模型：`{metadata["generator_model"]}`',
        f'- 情感模型：`{metadata["sentiment_model"]}`',
        f'- 设备：`{metadata["device"]}`；随机种子：`{metadata["seed"]}`',
        f'- 起始文本：`{metadata["prompt"]}`',
        f'- 每步收集合格候选数：{metadata["safe_k"]}；最多检查候选数：{metadata["search_limit"]}',
        "",
        "| 方法 | 生成 token | 最低 h | 最终 h | 拒绝候选 | 用时 | 停止原因 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        method = "无干预" if run["mode"] == "baseline" else f'CBF γ={run["gamma"]}'
        lines.append(
            f'| {method} | {run["generated_tokens"]} | {run["min_h"]:.3f} | '
            f'{run["final_h"]:.3f} | {run["rejected_total"]} | '
            f'{run["seconds"]:.1f}s | {run["stop_reason"]} |'
        )
    baseline = next(run for run in runs if run["mode"] == "baseline")
    first_crossing = next(
        (step for step, value in enumerate(baseline["h_trajectory"]) if value < 0), None
    )
    lines.extend(["", "## 本次观察", ""])
    if first_crossing is None:
        lines.append("- 无干预轨迹在本次运行中没有跌破 `h=0`。")
    else:
        lines.append(f"- 无干预轨迹在第 **{first_crossing}** 个生成 token 后首次跌破 `h=0`。")
    for run in runs:
        if run["mode"] == "cbf":
            incomplete = sum(step["allowed_candidates"] < metadata["safe_k"] for step in run["steps"])
            lines.append(
                f'- CBF `γ={run["gamma"]}` 的最低分数为 **{run["min_h"]:.3f}**，'
                f'共拒绝 **{run["rejected_total"]}** 个候选；'
                f'有 {incomplete} 步未在搜索上限内收满 {metadata["safe_k"]} 个合格候选。'
            )
    lines.extend(["", "![h 分数轨迹](h_trajectory.png)", ""])
    for run in runs:
        method = "无干预" if run["mode"] == "baseline" else f'CBF γ={run["gamma"]}'
        lines.extend([f"## {method} 的完整文本", "", "> " + run["text"].replace("\n", "\n>").rstrip(), ""])
    lines.extend(
        [
            "## 如何解读",
            "",
            "- 受控轨迹的每一步都由程序核对 `h_next >= γ × h_current`。若没有合格候选，程序停止，不放行违规 token。",
            "- 红色虚线是 `h=0`。如果无干预轨迹越线而受控轨迹没有越线，可直观看到本文单步机制。",
            "- `0%` 或轨迹未越线只代表这些提示词与随机种子下的现象，不能代表所有输入。",
            "- 这里的 `h` 由情感模型定义；“正面”不是一般意义的安全。",
            "- γ 更大表示每一步的通过条件更严格；不同 γ 会走上不同文本轨迹，最终 h 不必按 γ 大小排序。",
            "",
        ]
    )
    (output_dir / "实验结果.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-model", type=Path, default=ROOT / "models/distilgpt2")
    parser.add_argument("--sentiment-model", type=Path, default=ROOT / "models/roberta-sentiment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--safe-k", type=int, default=5)
    parser.add_argument("--search-limit", type=int, default=50)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.4])
    args = parser.parse_args()
    if not args.generator_model.is_dir() or not args.sentiment_model.is_dir():
        raise FileNotFoundError("Model directories missing. Follow README.md download commands.")
    if not 1 <= args.safe_k <= args.search_limit:
        raise ValueError("Require 1 <= safe-k <= search-limit")
    if any(not 0 <= gamma <= 1 for gamma in args.gammas):
        raise ValueError("All gamma values must lie in [0,1]")

    torch.set_num_threads(min(6, torch.get_num_threads()))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"Loading sentiment model on {device}...", flush=True)
    barrier = SentimentBarrier(args.sentiment_model, device)
    if args.prompt is None:
        scores = barrier.score_many(DEFAULT_PROMPTS)
        valid = [(prompt, score) for prompt, score in zip(DEFAULT_PROMPTS, scores) if score >= 0]
        if not valid:
            raise RuntimeError("No predefined prompt has nonnegative h; provide --prompt")
        prompt, initial_h = min(valid, key=lambda item: item[1])
        print(f"Selected lowest positive h among preset prompts: {prompt!r}, h={initial_h:.4f}", flush=True)
    else:
        prompt = args.prompt
        initial_h = barrier.score(prompt)
        print(f"Prompt h={initial_h:.4f}", flush=True)
    print("Loading generator...", flush=True)
    generator = Generator(args.generator_model, device)

    runs = []
    for mode, gamma in [("baseline", 0.0)] + [("cbf", g) for g in args.gammas]:
        label = "No intervention" if mode == "baseline" else f"CBF gamma={gamma}"
        print(f"Running {label}...", flush=True)
        run = run_once(
            generator, barrier, prompt, mode, gamma, args.seed,
            args.max_new_tokens, args.safe_k, args.search_limit,
        )
        runs.append(run)
        print(
            f"  tokens={run['generated_tokens']}, min_h={run['min_h']:.3f}, "
            f"rejected={run['rejected_total']}, seconds={run['seconds']:.1f}",
            flush=True,
        )
    metadata = {
        "generator_model": display_model_path(args.generator_model),
        "sentiment_model": display_model_path(args.sentiment_model),
        "device": device,
        "seed": args.seed,
        "prompt": prompt,
        "initial_h": initial_h,
        "safe_k": args.safe_k,
        "search_limit": args.search_limit,
        "max_new_tokens": args.max_new_tokens,
        "torch_version": torch.__version__,
    }
    write_outputs(runs, metadata, args.output_dir)
    print(f"Done. Results: {args.output_dir / '实验结果.md'}", flush=True)


if __name__ == "__main__":
    main()
