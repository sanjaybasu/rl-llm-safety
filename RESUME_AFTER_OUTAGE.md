# Resume After Outage

If local internet drops or the laptop sleeps, Modal jobs continue running independently. To check state and resume after any disruption, run these commands from any internet-connected terminal.

## 1. Check current state

```bash
# Apps (look for "ephemeral" = still running, "stopped" = done)
modal app list | grep rl-llm-safety-v3

# Local CSVs (Phase 1+5 output; 7 per run_id when complete)
modal volume ls rl-llm-safety-v3-predictions canonical

# Results files (Phase 9-12 output; empty until Phase 6 + downstream finish)
modal volume ls rl-llm-safety-v3-results
```

## 2. Canonical run_id

`3158b720-8638-4a1d-8faf-ebf391012ed9` — Phase 1+5 complete on volume. Always use this run_id when resuming.

The orphan run_id `c2b0e12e-bcb6-4d1f-9ee5-afa14f94a68d` is from a Modal worker restart; ignore it (its local CSVs are harmless but its LLM CSVs will never be produced).

## 3. Resume after Phase 6 failure (timeout or crash)

```bash
cd /Users/sanjaybasu/waymark-local/packaging/rl_llm_safety_github_v3/code/pipeline
modal run --detach modal_pipeline.py::resume_pipeline \
    --run-id 3158b720-8638-4a1d-8faf-ebf391012ed9
```

This skips Phase 0-5 (already on volume) and re-runs Phase 6 → 8 → 9 → 10 → 11 → 12 → 14.

Phase 6 is resumable: `llm_inference.run_client_on_dataset` checkpoints every 100 messages and skips already-completed `message_id` values on relaunch. A worker death loses at most 99 messages, not the full 2,000.

## 4. Skip Phase 6 entirely (if LLM CSVs already complete on volume)

```bash
modal run --detach modal_pipeline.py::resume_pipeline \
    --run-id 3158b720-8638-4a1d-8faf-ebf391012ed9 \
    --skip-phase6
```

Runs only Phase 8 → 14. Use when the two LLM CSVs already exist on the volume.

## 5. Download final outputs after pipeline completes

```bash
mkdir -p ~/Downloads/rl-llm-safety-v3-results
modal volume get rl-llm-safety-v3-results . ~/Downloads/rl-llm-safety-v3-results

mkdir -p ~/Downloads/rl-llm-safety-v3-predictions
modal volume get rl-llm-safety-v3-predictions canonical/ ~/Downloads/rl-llm-safety-v3-predictions
```

Then run the local Phase D rendering:

```bash
cd /Users/sanjaybasu/waymark-local/packaging/rl_llm_safety_github_v3
code/pipeline/phase_d_render.sh 3158b720-8638-4a1d-8faf-ebf391012ed9
```

## 6. Internet drop during this session

The Modal pipeline does not depend on the local terminal. The background monitor (a `Monitor` task in this session) will pause emitting notifications during an outage but will resume once connectivity returns. No action needed.

The deployed app (`ap-lIO7GUwSOBiXAj03UpdtCY` — `rl-llm-safety-v3-pipeline`) and the in-flight ephemeral runs persist regardless of local connectivity.
