# RecoveryAgent v0.4-A Training Readiness

Status: **CPU_DATA_READY**

- Oracle trusted decisions: 204
- Real Teacher trusted decisions: 84
- Independent Train tasks: {'combined': 52, 'oracle_reference': 52, 'teacher': 11}
- Real Teacher episodes/tasks audited: 33/16
- Matched reference rows: 204/208
- Cross-split leakage: 0
- Source files unchanged: True

## Trust and causality

Successful full-Verifier episodes only; per-decision replay/behavior evidence; uncertain, failures, redundant mutations and post-success calls excluded. Historical assistant actions remain masked.

Recorded model-visible history is authoritative. Historical calls were serialized individually by the existing Runtime; target raw multi-call groups and FIFO call_id mappings are retained.

Final-failed episodes are excluded conservatively; partial-action salvage is not attempted. Successful episodes still undergo per-call filtering. Diagnostic calls require existing positive behavior evidence; uncertain calls are excluded. Final answers require an autonomous successful stop. Privileged evidence is metadata only, never messages/tools. Historical failed calls remain visible as observed context but receive no loss.

Same task under several prompts counts once in independent task coverage. IID validation/test may share mechanisms with Train but have separate frozen variants; OOD mechanisms are disjoint.

## Official tokenizer and labels

AutoTokenizer loads the repository's unmodified Qwen3 template with enable_thinking=False. The official template renders tool schema in System and Tool Responses as user/tool_response. It does not render call_id: original IDs, group order and parameter/response bindings remain in messages and tool_bindings and are validated FIFO before rendering.

No native generation-mask markers exist in this pinned template. Full conversation IDs are checked against independent encoding of the rendered template. The official generation-prompt IDs must be an exact prefix of the full conversation. Only the final verified assistant content/tool calls and its im_end EOS receive labels; System/User/Tool/history and the generation header/empty thinking prefill are -100. No string-length offsets are used.

Every retained decision is an individual causal multi-round prefix, not an episode with all assistant actions supervised. Future results of the target action are excluded.

Encoding configuration: max_seq_length=8192, truncation=False.

Official model config native context: 40960 tokens; over native context: 0. Tokenizer model_max_length is recorded separately and is not treated as the native model limit.

| Source | Samples | P50 | P95 | P99 | Max | Over limit | Supervised tokens | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_reference | 204 | 1591 | 2017 | 2050 | 2060 | 0 | 4294 | 0.0128 |
| teacher | 84 | 1836 | 2403 | 2848 | 2848 | 0 | 3654 | 0.0230 |

Percentiles use nearest rank. Long sequences are retained and listed, never truncated or silently dropped. 8192 is a configurable candidate training length, not a measured GPU-memory optimum. Examine mask_examples.json for full Git/SQLite token/label alignment.

## Dataset coverage and exclusions

```json
{
  "domains": {
    "oracle_reference": {
      "git": 108,
      "mixed": 24,
      "sqlite": 72
    },
    "teacher": {
      "git": 27,
      "mixed": 49,
      "sqlite": 8
    }
  },
  "target_kinds": {
    "oracle_reference": {
      "final_answer": 52,
      "tool_call": 152
    },
    "teacher": {
      "final_answer": 12,
      "tool_call": 72
    }
  },
  "duplicates_removed": {},
  "exclusions": {
    "failed_episode": 109,
    "invalid_decision": 3,
    "non_train_episode": 188,
    "post_success_call": 23,
    "untrusted:invalid": 1,
    "untrusted:redundant": 4,
    "untrusted:tool_failed": 3,
    "untrusted:uncertain": 23,
    "untrusted_decision": 27
  }
}
```

Mixed export: disabled; sources remain separate.

## Before LoRA-SFT

1. Review trusted Teacher coverage and exclusion evidence; 33 episodes of repeated tasks are not 33 independent tasks. Oracle/Mock results are not real-model performance.
2. Choose Oracle/Teacher sampling weights explicitly. Existing cohorts have limited independent mechanism/variant coverage; no samples are fabricated to meet a quota.
3. Decide context handling from the measured distribution. Keep final targets and tool groups intact.
4. On the training host, pin the same tokenizer revision/template hashes, verify the training collator honors exported labels, and run a small CPU/GPU batch preflight before any training.
5. Keep IID validation, IID test and OOD out of training and hyperparameter-negative mining.

No weights, paid API requests, training code or gradient updates are part of this stage.

## Actual acceptance

- Focused tests: 6 passed in 15.54s (official tokenizer integration, no skips).
- Full tests: 269 passed in 326.91s.
- Existing core modules and frozen/original source hashes unchanged.
- Paid API calls: 0; model weights: none; training dependencies/gradient updates: none.
- Suggested initial training context for this fixed cohort: 4096 (observed maximum 2848); the preparation report threshold remains 8192. This is a length-based proposal, not a GPU memory benchmark.
