# Blind judging kit: stance vs reference in SV tails

Files an LLM may see (no bank information inside):
- `blind_words.txt` - 4,381 words to label R/S/F/X (instructions at top). Output `word<TAB>label`.
- `blind_tails_graded.md` - 310 tails; count stance and reference words. Output `id<TAB>n_stance<TAB>n_reference`.
- `blind_tails.md` - same 310 tails; one label each (control). Output `id<TAB>LABEL`.

Files to keep from the judge:
- `blind_tails_key.tsv` (id -> bank/SV/polarity/tier), `tail_words.jsonl` (all tails' word lists, with bank).

Scoring (needs Python 3; scipy optional for p-values):
    python score_word_judgments.py   your_word_labels.tsv      # tier table for both banks
    python score_blind_graded.py     your_graded_counts.tsv    # per-tier stance share by bank
    python score_blind_judgments.py  your_tail_labels.tsv      # categorical control
    python compare_word_labels.py    word_labels_mine.tsv your_word_labels.tsv

My baselines: `word_labels_mine.tsv`, `blind_tails_graded_mine.tsv`, `blind_tails_labels_mine.tsv`.
Results and interpretation: `../stance_vs_reference_quantified.md`.
