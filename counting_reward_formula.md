# Counting reward — the scale-aware graded formula

The reward used for `reward_type == "counting"` answers. Introduced by PR #40
(commit `9ac5e40`), which replaced the previous **exact-match** counting reward
with a **scale-aware graded (decaying-band)** reward: full credit inside a band
that scales with the count, then a smooth decay outside it. This doc is the
authoritative formula reference CLAUDE.md points to.

- **Code:** `counting_reward_value(pred, gt, tolerance=None, *, rel_tol, rel_band_cap, rel_sigma, min_sigma, decay_exp, graded)`
  at `verl/utils/reward_score/math_verify_reward_type_boxed.py`.
- **String wrapper:** `_counting_reward(pred_str, truth, extra_info=..., counting_reward_kwargs=...)`
  parses both sides to integers (number words and thousands separators are handled),
  returns `0.0` on a non-numeric prediction, and falls back to string match when the
  ground truth is non-numeric.
- **Routing:** per-row `extra_info["reward_type"] == "counting"` → the handler map in
  `math_verify_reward_type_multiturn_tool_calling_optional_think.py::acc_reward`.
- **Ablation harness / plots:** `vero2_scripts/debug_scripts/counting_reward_ablation.py`
  (regenerates every figure and table in `docs/vero2/counting-reward-ablation.md`).

## The formula

Let `d = |pred − gt|`. With the five knobs (defaults in the table below):

1. **Exact** — `d == 0` → **1.0**.
2. **Intrinsic band** — `band = floor(rel_tol · |gt|)`. The `floor` keeps small
   counts exact: `floor(0.05·|gt|) = 0` for `gt < 20`, so an off-by-one earns full
   credit only from `gt = 20` up, while `gt = 100` accepts `95–105`.
3. **Per-row widening (optional)** — if `extra_info["tolerance"] = tol > 0`,
   `band = max(band, min(tol, rel_band_cap · |gt|))`. The `rel_band_cap` ceiling
   means a large absolute tolerance can never loosen a small count (e.g. `tol = 2`
   on `gt = 1` still requires an exact answer). **In our current training data
   `tolerance` is always null**, so this branch is inert.
4. **In band** — `d ≤ band` → **1.0**.
5. **Decay** — otherwise
   `reward = exp( −((d − band) / σ) ^ decay_exp )`, with
   `σ = max(min_sigma, rel_sigma · |gt|)`.
   The `min_sigma` floor keeps small counts near-exact (`gt = 1, pred = 0 → ~0.018`)
   and avoids a divide-by-zero at `gt = 0`.

`graded = False` collapses step 5 to a hard **0.0** outside the band (a "band-only"
reward), which isolates the leniency band from the decay gradient.

### Reference values (default config)

| gt | pred | d | band | σ | reward |
|---|---|---|---|---|---|
| 100 | 110 | 10 | 5 | 15 | `exp(−(5/15)²)` = **0.8948** |
| 7 | 8 | 1 | 0 | 1.05 | `exp(−(1/1.05)²)` = **0.4037** |
| 1 | 0 | 1 | 0 | 0.5 | `exp(−(1/0.5)²)` = **0.0183** |

The harness asserts these to `< 5e-4` on every run
(`counting_reward_ablation.py --check-only`), and `tests/utils/reward_score/test_counting_reward_on_cpu.py`
tests them as a unit.

## The knobs

All five knobs are module constants that also thread through
`reward.custom_reward_function.reward_kwargs` (Phase B of the ablation), so a
training arm is a Hydra override, not a code edit. **Each override defaults to the
module constant — omitting all of them is bit-identical to the shipped reward.**

| Knob | Constant | Default | `reward_kwargs` key | Effect |
|---|---|---|---|---|
| Band fraction | `_COUNTING_REL_TOL` | `0.05` | `counting_rel_tol` | Width of the full-credit band as a fraction of `|gt|`. `0` ⇒ only exact answers get full credit (decay still applies). |
| Band cap | `_COUNTING_REL_BAND_CAP` | `0.15` | `counting_rel_band_cap` | Ceiling on per-row `tolerance` widening (inert while `tolerance` is null). |
| Decay width fraction | `_COUNTING_REL_SIGMA` | `0.15` | `counting_rel_sigma` | `σ` as a fraction of `|gt|`; the dominant lever on how gently the reward decays. |
| Width floor | `_COUNTING_MIN_SIGMA` | `0.5` | `counting_min_sigma` | Minimum `σ`; matters only for small counts (`rel_sigma·gt < min_sigma`). |
| Decay exponent | `_COUNTING_DECAY_EXP` | `2.0` | `counting_decay_exp` | `2` = Gaussian `exp(−x²)`; `1` = Laplacian `exp(−x)` (fatter tail). |
| Graded / band-only | *(argument)* | `True` | `counting_graded` | `False` ⇒ hard `0.0` outside the band (no decay gradient). |

### How overrides flow

```
config: reward.custom_reward_function.reward_kwargs.counting_rel_sigma
  → async_compute_score_from_data_source(counting_rel_sigma=…)
  → compute_score(counting_rel_sigma=…)                        # bundles non-default knobs
  → acc_reward(…, counting_reward_kwargs={"rel_sigma": …})     # only for reward_type=="counting"
  → _counting_reward(…, counting_reward_kwargs=…)
  → counting_reward_value(pred, gt, rel_sigma=…)
```

`_build_counting_reward_kwargs` returns `{}` when every knob is at its default, and
`acc_reward` then never forwards it — so the counting path is provably unchanged
unless a knob is explicitly overridden.

### Example overrides

```bash
# wider Gaussian tail (denser gradient on near-misses)
+reward.custom_reward_function.reward_kwargs.counting_rel_sigma=0.30

# reproduce the pre-PR#40 exact-match reward
+reward.custom_reward_function.reward_kwargs.counting_rel_tol=0.0
+reward.custom_reward_function.reward_kwargs.counting_graded=false

# Laplacian decay
+reward.custom_reward_function.reward_kwargs.counting_decay_exp=1.0
```

(Use `+` because these keys are not present in the shipped configs; they are added,
not overwritten.)

## Why this shape

Exact-match counting reward is **scale-blind**: an off-by-one is equally fatal at
`gt = 3` and `gt = 300`, even though the second is a far better answer, and large
counts become nearly impossible to get exactly right — so the reward is ~0 and
carries no gradient. On 10,585 real `(pred, gt)` pairs the exact-match mean reward
collapses from **0.73** at `gt = 1–4` to **0.065** at `gt ≥ 100`, and **48%** of all
pairs score exactly zero. The scale-aware band + decay makes the reward (a) tolerant
in proportion to the count and (b) *graded*, so a near-miss on a large count still
produces a useful gradient. See `docs/vero2/counting-reward-ablation.md` for the full
offline study and the training-arm shortlist.
