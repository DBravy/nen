# GPT-OSS-20B subjective-progress annotations

## Construct

An event is a short span of generated reasoning that appears to increase the
model's own expectation of future task success.  Informally, selection targets
a positive update in a latent value estimate, not objective correctness:

`routine reasoning -> perceived progress update -> reasoning under the new belief`

The event can be a realization, a decision, a hypothesis revision, a newly
noticed constraint, a diagnosis of failure, a useful intermediate result, or
an action selected because the model expects it to resolve an important
uncertainty.  Both genuine and mistaken breakthroughs are included.

## Inclusion rule

Include a span when the local trajectory supports all three claims:

1. The thought introduces or commits to information, a hypothesis, or a course
   of action that changes the subsequent direction of reasoning.
2. At generation time, the model treats that change as promising or
   uncertainty-reducing.
3. The event can be localized to a minimal, verbatim span of the assistant's
   `thinking` text.

Explicit cue words such as “wait,” “aha,” and “therefore” are neither required
nor sufficient.  Tool outputs are evidence that can prompt an event, but an
external observation alone is not labeled until the model interprets it as a
progress-making update.

## Exclusions

- Routine execution after a strategy has already been selected.
- A generic plan stated before any meaningful commitment or narrowing.
- Correct calculations that do not change the model's appraisal or direction.
- Pure surprise, confusion, or negative uncertainty with no promising update.
- Final answers that merely restate an already-established conclusion.
- Tool calls made mechanically, without evidence that the model expects them
  to resolve a consequential uncertainty.

Closely adjacent sentences expressing one update are represented as one event.
Distinct updates in the same turn can be separate events when the intervening
reasoning establishes a genuine new transition.

## Objective-status metadata

`objective_status` is assigned only after selecting the event:

- `correct`: later evidence supports the realization, diagnosis, inference, or
  strategic premise represented by the event.
- `incorrect`: later evidence contradicts that premise or shows that the move
  did not work for the reason the model believed.
- `ambiguous`: the available trajectory does not resolve the premise, or the
  event is a strategic choice without a clean truth value.

This field never determines inclusion.  In particular, a failed rollout can
contain correct local progress, and a successful rollout can contain mistaken
progress events.

## Location and alignment

Each event has three parallel location systems:

- `step` plus `[thinking_char_start, thinking_char_end)` locates the verbatim
  event in `transcript.json`; these are zero-based Python/Unicode character
  offsets, with the end excluded.
- `transcript_message_index` is the zero-based index of that assistant message
  in the transcript array.
- `[generated_token_start, generated_token_end]` is the inclusive token span in
  that step's `generated_token_ids`.
- `[absolute_token_start, absolute_token_end]` is the corresponding inclusive
  span in that replay step's `full_token_ids`.
- `anchor_generated_position` is the final generated token overlapping the
  event.  This is the default `t = 0` anchor.

As documented by `analyze_aha_rollout.py`, the saved activation at generated
position `g` is the state after consuming token `g`, causally used to predict
token `g + 1`.  The full event span is retained so analyses can instead align
to event onset, event midpoint, or every token in the event.

The existing tokenwise direction readouts are directly indexed as well:
`w_arm_row_start`, `w_arm_row_end`, and `w_arm_anchor_row_id` point into
`task_gaming_candidate_data/token_data.jsonl`.  The parallel
`w_arm_*_token_index` fields give zero-based analysis-message token positions.
This makes the annotations usable immediately with the dense W-arm series even
though the separate per-component `activations.pt` files contain coarser replay
checkpoints.

`before`, `event`, and `after` are exact transcript excerpts.  The context
fields aid qualitative review; the character and token indices are canonical
for programmatic alignment.

When the immediately preceding trajectory message is a tool or harness
observation, `prior_message_excerpt` stores an exact suffix (at most roughly
600 characters, beginning at a line boundary) plus its message index and
starting character offset.  This preserves the evidence that triggers events
occurring at the start of a fresh reasoning turn without duplicating very long
tool outputs.

`confidence` describes confidence in the annotation and retrospective
adjudication, not the model's confidence and not the event's strength.

One labeled trajectory, `precommit_hook__review_hook__s62346`, ended with a
CUDA OOM and is stored as `transcript.partial.json`.  Its seven completed
assistant turns are included and marked `trajectory_status: partial_error`.
The event locations still map to its replay token maps.  This run was omitted
from the previously assembled W-arm table, so its `w_arm_*` row fields are
null and `w_arm_alignment_available` is false; all events from the other 23
rollouts have direct W-arm rows.

## Files

- `subjective_progress_{secret,precommit,impossible}.jsonl` in the repository
  root are the hand-authored source annotations; their parallel `_coverage`
  files record inspection of every rollout.
- `events.jsonl`: one validated and token-aligned record per selected event.
- `rollout_coverage.jsonl`: one row for every available rollout, including a
  reason when no event was selected.
- `events_review.md`: human-readable event contexts and adjudications.
- `summary.json`: dataset counts and source-coverage notes.

Regenerate and validate the derived files with:

```bash
python3 build_subjective_progress_annotations.py
```
