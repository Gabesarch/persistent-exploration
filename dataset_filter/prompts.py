"""Prompt templates for ground-truth filtering of the counting dataset.

Three strategies, designed around the core tension the user identified:

  * Blind (image only): on HARD images even a great model may miscount, so a
    naive "blind == GT" check wrongly rejects correct-but-hard examples.
  * Verify (image + answer): showing the answer risks sycophancy — the model
    rubber-stamps a wrong GT.

The "smart" pipeline combines them: an independent blind count (LLM-as-extraction
+ our own exact-match metric) decides the easy cases; only on disagreement do we
run an answer-aware *adjudication* that is explicitly told NOT to reject an
example merely because it is hard (that is what the PLAUSIBLE verdict is for).
"""

# ---------------------------------------------------------------------------
# Stage 1 — blind independent count (no ground truth shown -> no sycophancy)
# ---------------------------------------------------------------------------

BLIND_SYSTEM = (
    "You are an expert visual-counting assistant with excellent fine-grained "
    "perception. You count objects precisely and patiently, even when they are "
    "small, numerous, overlapping, blurry, or partially occluded. You never "
    "guess a round number to avoid work — you actually count."
)

BLIND_USER = """Question: {question}

Count the requested objects as accurately as you can. Scan the WHOLE image
methodically (e.g. left-to-right, top-to-bottom, in bands) so you do not miss
clusters or edge instances. Include partially visible instances that clearly
belong to the category. Do not stop early and do not round off.

Count ONLY instances you can actually see — the way a person clicking once on
each object would. Do NOT infer, estimate, or add objects that are hidden behind
others, buried inside a stack/pile, or in the unseen interior of a 3D structure.
For a solid block or pyramid of objects, count the visible ones, not an assumed
filled-in volume.

If the scene is so dense/cluttered that an exact count is impossible, still give
your single best estimate for "count", and widen [count_low, count_high] to the
range you are confident the true number falls inside.

Return ONLY a JSON object, no prose before or after:
{{
  "count": <integer best exact count>,
  "count_low": <integer lower bound; = count if you are sure>,
  "count_high": <integer upper bound; = count if you are sure>,
  "difficulty": "easy" | "medium" | "hard",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<one or two sentences: how you counted and any uncertainty>"
}}"""


# ---------------------------------------------------------------------------
# Stage 2 — answer-aware adjudication (fair auditor; hard != wrong)
# ---------------------------------------------------------------------------

VERIFY_SYSTEM = (
    "You are a strict but fair auditor for a visual-counting dataset. You decide "
    "whether a proposed ground-truth count is correct for an image+question. "
    "Hard images with many objects are the MOST valuable examples in this dataset "
    "and must be KEPT as long as the ground truth is correct or plausible. You only "
    "reject an example when you have concrete evidence the ground truth is wrong, the "
    "question is unanswerable, or the question is AMBIGUOUS / ill-posed. We do not "
    "want ambiguous examples in the dataset. You NEVER reject an example just because "
    "counting it is difficult or you are unsure — for those, you use PLAUSIBLE."
)

# `{blind_block}` = the independent blind count when available, else "".
# `{points_block}` = explanation of the drawn GT annotation marks when available, else "".
VERIFY_USER = """Question: {question}
Proposed ground-truth answer: {gt}
{blind_block}{points_block}
Re-count the objects yourself, carefully and completely, then judge the proposed
ground-truth answer of {gt}.

Count ONLY clearly visible, distinct instances — as a human annotator clicking
once on each object would. Do NOT infer or add hidden objects behind/inside a
stack, pile, or 3D structure, and do NOT assume a solid filled interior you cannot
see. The ground truth was made by marking visible objects, so over-counting unseen
ones is YOUR error, not the ground truth's.

Before counting, check the QUESTION for ambiguity: is the target category and the
region well-defined and objective? If reasonable people would disagree on WHAT to
count (subjective category, undefined region/qualifier like "innermost stripe", or
"all the windows" spread across many buildings in a painting), the example is
AMBIGUOUS — we do not want it, regardless of any count.

Decision rules (read carefully):
- CORRECT     : your own count equals {gt}, or is within ~1-2 of it (or a few
                percent on large counts), or you are confident {gt} is right.
- PLAUSIBLE   : you cannot pin down the exact count (very dense / occluded / tiny
                objects, or the category boundary is mildly fuzzy, e.g. which shade
                counts as "blue"), but the question is well-posed and {gt} sits in a
                believable range with NO strong evidence it is wrong. KEEP these —
                they are the valuable hard examples.
- WRONG       : concrete evidence the true count of VISIBLE instances differs from
                {gt} by a LARGE margin (more than ~15% AND more than 2 in absolute
                terms), OR the category in the question clearly does not match / is
                absent. State your own count.
- AMBIGUOUS   : the QUESTION itself is ill-posed or subjective — the target category
                or region is not well-defined, so no single correct count exists.
- UNANSWERABLE: the requested object class is entirely absent, or the image cannot
                support any count at all.

Critical: do NOT choose WRONG merely because the image is hard, dense, or you are
uncertain, and NOT for a small difference of 1-2 (or a few percent) — that is what
CORRECT/PLAUSIBLE are for. Reserve AMBIGUOUS for a genuinely ill-posed QUESTION, not
for a merely hard image.

Return ONLY a JSON object, no prose before or after:
{{
  "your_count": <integer best estimate>,
  "verdict": "CORRECT" | "PLAUSIBLE" | "WRONG" | "AMBIGUOUS" | "UNANSWERABLE",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<2-4 sentences: what you counted and why this verdict; if WRONG give the evidence; if AMBIGUOUS say what is undefined>"
}}"""


def points_block(n_points):
    """Explain the drawn GT annotation marks for the points-aware verify call."""
    if not n_points:
        return ""
    return (
        f"The image has the dataset's {n_points} ground-truth annotation marks drawn "
        f"as bright magenta rings — one ring per counted object, so the proposed "
        f"answer equals the number of rings. These marks are a GUIDE and may be noisy, "
        f"misplaced, doubled, or miss an object — do NOT trust them blindly. Audit "
        f"them: is each ring on a distinct, valid instance of the requested category? "
        f"Are there clearly unmarked instances, rings on empty space / the wrong "
        f"object, or two rings on one object? Weigh this against your own recount.\n"
    )


def blind_block(blind):
    """Render the optional 'independent counter said ...' hint for stage 2."""
    if not blind or blind.get("count") is None:
        return ""
    lo, hi, c = blind.get("count_low"), blind.get("count_high"), blind.get("count")
    rng = f" (range {lo}-{hi})" if lo is not None and hi is not None and (lo != c or hi != c) else ""
    return (
        f"For reference, an independent counter who did NOT see the proposed answer "
        f"counted: {c}{rng}. Treat this as a hint, not as truth — verify yourself.\n"
    )
