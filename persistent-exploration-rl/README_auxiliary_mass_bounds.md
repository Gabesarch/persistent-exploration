# Auxiliary mass bounds

The policy loss limits the contribution of retry branches and the critic relative to canonical trajectories. In the Qwen recipe, branch absolute advantage mass is capped at 25% of canonical mass and the critic weighted cross entropy at 5%. Branch loss weights have a maximum 20% share across the policy minibatch, and subsampling keeps at most two branch rows per parent. These limits protect the canonical policy update while still training exploration after rejected answers.

The implementation is in `verl/workers/utils/auxiliary_mass_bounds.py`, with loss integration in the actor worker. CPU tests are in `tests/workers/utils/test_auxiliary_mass_bounds_on_cpu.py`. Run them with `PYTHONPATH=$PWD python -m pytest -q tests/workers/utils/test_auxiliary_mass_bounds_on_cpu.py`.
