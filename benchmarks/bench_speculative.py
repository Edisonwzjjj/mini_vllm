"""Benchmark: greedy vs chain-PLD vs tree-dummy vs tree-PLD speculative decode.

Four runs on the same prompts:
  1. greedy baseline (no spec)
  2. chain spec + PLD draft (real 0-cost draft from prompt n-grams)
  3. tree spec + dummy draft (depth-3 leftmost = repeat(last_token))
  4. tree spec + PLD multi-branch draft (option-A 7-slot topology)

Variable separation:
  - chain vs greedy           -> chain spec end-to-end value (with real draft)
  - tree-dummy vs greedy      -> isolates verify-forward overhead when draft
    quality is ~0 (dummy chain is repeat-token, almost always rejected).
    A speedup < 1.0x quantifies the verify-forward tax.
  - tree-PLD vs tree-dummy    -> draft-quality contribution within tree mode
  - tree-PLD vs chain-PLD     -> tree-topology value (does fanning out help?)
  - chain - tree-dummy gap    -> chain draft-quality contribution

Correctness sanity (temperature=0, greedy):
  output token ids must equal the greedy baseline for every prompt.
"""

import time
from mini_vllm import LLM, SamplingParams

MODEL_PATH = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 128
DRAFT_LEN = 4
TREE_TOPK = 2
TREE_DEPTH = 3

# Prompts crafted with strong literal repetition / list / code structure
# so PLD has real ngram hits. The last one is open-ended prose to give
# PLD a tougher case.
PROMPTS = [
    "List the days of the week: Monday, Tuesday, Wednesday, Thursday, Friday, "
    "Saturday, Sunday. List the days of the week again: Monday, Tuesday,",

    "def add(a, b): return a + b\n"
    "def sub(a, b): return a - b\n"
    "def mul(a, b): return a * b\n"
    "def div(a, b): return",

    "Shopping list:\n- apple\n- banana\n- cherry\n- date\n- elderberry\n"
    "Shopping list:\n- apple\n- banana\n- cherry\n-",

    "Q: What is 2+2? A: 4.\nQ: What is 3+3? A: 6.\nQ: What is 4+4? A: 8.\n"
    "Q: What is 5+5? A:",

    "Once upon a time in a small village there lived",
]


def make_llm(**eagle_kwargs):
    return LLM(
        model_path=MODEL_PATH,
        block_size=16,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.5,
        deterministic=False,
        **eagle_kwargs,
    )


def run(llm, prompts, sp):
    t0 = time.time()
    outputs = llm.engine.generate(prompts, sp)
    elapsed = time.time() - t0
    total = sum(len(o["token_ids"]) for o in outputs)
    return outputs, total, elapsed


def fmt_metrics(m: dict, draft_len_ref: int) -> str:
    proposed = m["draft_tokens_proposed"]
    accepted = m["draft_tokens_accepted"]
    bonus = m["bonus_tokens"]
    spec = m["spec_steps"]
    fallback = m["fallback_steps"]
    total_steps = spec + fallback
    acc = accepted / proposed * 100 if proposed else 0.0
    bonus_rate = bonus / spec * 100 if spec else 0.0
    fb_share = fallback / total_steps * 100 if total_steps else 0.0
    avg_acc = accepted / spec if spec else 0.0
    return (
        f"  spec_steps={spec}  fallback={fallback} ({fb_share:.1f}%)\n"
        f"  proposed={proposed}  accepted={accepted}  bonus={bonus}\n"
        f"  acceptance_rate={acc:.1f}%  bonus_rate={bonus_rate:.1f}%  "
        f"avg_accept/spec={avg_acc:.2f} (of {draft_len_ref})"
    )


def main():
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    print("=" * 72)
    print("[1/3] greedy baseline (no spec)")
    print("=" * 72)
    llm_g = make_llm()
    g_out, g_tok, g_t = run(llm_g, PROMPTS, sp)
    g_tps = g_tok / g_t
    print(f"tokens={g_tok}  time={g_t:.2f}s  tps={g_tps:.2f}")
    del llm_g

    print()
    print("=" * 72)
    print(f"[2/3] chain spec + PLD draft (draft_len={DRAFT_LEN})")
    print("=" * 72)
    llm_c = make_llm(
        enable_eagle=True,
        eagle_mode="chain",
        eagle_draft_len=DRAFT_LEN,
        eagle_use_pld=True,
        eagle_pld_max_ngram=3,
        eagle_pld_min_ngram=2,
    )
    runner_c = llm_c.engine.spec_runner
    runner_c.reset_metrics()
    c_out, c_tok, c_t = run(llm_c, PROMPTS, sp)
    c_tps = c_tok / c_t
    print(f"tokens={c_tok}  time={c_t:.2f}s  tps={c_tps:.2f}")
    print(fmt_metrics(runner_c.metrics, DRAFT_LEN))
    chain_metrics = dict(runner_c.metrics)
    del llm_c

    print()
    print("=" * 72)
    print(f"[3/4] tree spec + dummy draft (topk={TREE_TOPK}, depth={TREE_DEPTH})")
    print("=" * 72)
    llm_t = make_llm(
        enable_eagle=True,
        eagle_mode="tree",
        eagle_topk=TREE_TOPK,
        eagle_spec_steps=TREE_DEPTH,
    )
    runner_t = llm_t.engine.spec_runner
    runner_t.reset_metrics()
    t_out, t_tok, t_t = run(llm_t, PROMPTS, sp)
    t_tps = t_tok / t_t
    print(f"tokens={t_tok}  time={t_t:.2f}s  tps={t_tps:.2f}")
    print(fmt_metrics(runner_t.metrics, TREE_DEPTH))
    tree_metrics = dict(runner_t.metrics)
    del llm_t

    print()
    print("=" * 72)
    print(f"[4/4] tree spec + PLD multi-branch (topk={TREE_TOPK}, depth={TREE_DEPTH})")
    print("=" * 72)
    llm_tp = make_llm(
        enable_eagle=True,
        eagle_mode="tree",
        eagle_topk=TREE_TOPK,
        eagle_spec_steps=TREE_DEPTH,
        eagle_use_pld=True,
        eagle_pld_max_ngram=3,
        eagle_pld_min_ngram=2,
    )
    runner_tp = llm_tp.engine.spec_runner
    runner_tp.reset_metrics()
    tp_out, tp_tok, tp_t = run(llm_tp, PROMPTS, sp)
    tp_tps = tp_tok / tp_t
    print(f"tokens={tp_tok}  time={tp_t:.2f}s  tps={tp_tps:.2f}")
    print(fmt_metrics(runner_tp.metrics, TREE_DEPTH))
    tree_pld_metrics = dict(runner_tp.metrics)
    del llm_tp

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'method':<32}{'tokens/s':>12}{'speedup':>10}")
    print(f"{'greedy baseline':<32}{g_tps:>12.2f}{'1.00x':>10}")
    print(f"{'chain spec (PLD draft)':<32}{c_tps:>12.2f}{f'{c_tps/g_tps:.2f}x':>10}")
    print(f"{'tree spec (dummy draft)':<32}{t_tps:>12.2f}{f'{t_tps/g_tps:.2f}x':>10}")
    print(f"{'tree spec (PLD draft)':<32}{tp_tps:>12.2f}{f'{tp_tps/g_tps:.2f}x':>10}")

    print()
    print("Variable separation:")
    print(f"  verify-forward tax (tree-dummy / greedy)        = {t_tps/g_tps:.2f}x  "
          "(< 1.0 means verify forward is slower than single-token decode)")
    print(f"  tree draft-quality (tree-PLD - tree-dummy)      = {tp_tps/g_tps - t_tps/g_tps:+.2f}  "
          "(speedup gained by replacing dummy with PLD inside tree mode)")
    print(f"  tree-topology value (tree-PLD - chain-PLD)      = {tp_tps/g_tps - c_tps/g_tps:+.2f}  "
          "(does fanning out branches beat a single chain?)")
    print(f"  chain draft-quality (chain-PLD - tree-dummy)    = {c_tps/g_tps - t_tps/g_tps:+.2f}  "
          "(chain with real PLD vs no-draft-quality tree baseline)")

    print()
    # Correctness sanity vs greedy
    def cmp(label, a_out, b_out):
        bad = 0
        for i, (a, b) in enumerate(zip(a_out, b_out)):
            if a["token_ids"] != b["token_ids"]:
                bad += 1
                # find first divergence
                k = 0
                for x, y in zip(a["token_ids"], b["token_ids"]):
                    if x != y:
                        break
                    k += 1
                print(f"  [MISMATCH] {label} prompt {i}: diverge@{k}, "
                      f"len {len(a['token_ids'])} vs {len(b['token_ids'])}")
        if bad == 0:
            print(f"  {label}: OK ({len(a_out)}/{len(a_out)} match greedy)")
        else:
            print(f"  {label}: FAIL ({bad} mismatches)")

    print("Correctness vs greedy:")
    cmp("chain spec    ", g_out, c_out)
    cmp("tree spec     ", g_out, t_out)
    cmp("tree spec PLD ", g_out, tp_out)

    print()
    print("Per-prompt PLD acceptance (chain spec, isolated runs):")
    print(f"  {'idx':>3}  {'len':>4}  {'acc':>6}  {'fb':>4}  preview")
    llm_c2 = make_llm(
        enable_eagle=True,
        eagle_mode="chain",
        eagle_draft_len=DRAFT_LEN,
        eagle_use_pld=True,
        eagle_pld_max_ngram=3,
        eagle_pld_min_ngram=2,
    )
    runner_c2 = llm_c2.engine.spec_runner
    for i, prompt in enumerate(PROMPTS):
        runner_c2.reset_metrics()
        out = llm_c2.engine.generate([prompt], sp)
        m = runner_c2.metrics
        a, p, f = m["draft_tokens_accepted"], m["draft_tokens_proposed"], m["fallback_steps"]
        rate = a / p * 100 if p else 0.0
        preview = out[0]["text"].replace("\n", " ")[:36]
        print(f"  {i:>3}  {len(out[0]['token_ids']):>4}  {rate:>5.1f}%  {f:>4}  {preview!r}")


if __name__ == "__main__":
    main()
