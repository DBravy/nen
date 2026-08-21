# Task Gaming Lens Explorer

Run from the repository root:

```bash
python3 task_gaming_dashboard.py
```

The server uses only Python's standard library and opens `http://127.0.0.1:8765`.
It reads `task_gaming_candidate_data` in place; no 191 MB copy is sent to the
browser. Use `--no-open`, `--port`, or `--data` to adjust startup behavior.

Views:

- **Token lens** — select a trial and generation step, compare activation traces,
  browse the generated token stream, and inspect full J-lens top/bottom tokens.
- **SVD directions** — inspect the transported positive/negative readout and the
  observed activation distribution for each candidate.
- **Trial index** — search labels, flags, conditions, and notes, then jump directly
  into a rollout.
- **Gaming moments** — reviewed consideration, decision, rejection, action, failed,
  and accidental-proxy landmarks with excerpts and one-click jumps into the CoT.
