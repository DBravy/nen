# Stance vs reference, quantified: `unrealized_words_selectivity` vs `g_scan_sparse`

Follow-up to `top_contexts_semantics_uws_vs_gss.md`, which argued from reading that UWS's word-level
directions are organised around *stance* (how the speaker stands toward what is said) and GSS's around
*reference* (what is being talked about). This file tests that claim with a word lexicon I labelled
myself, reports the result with its limits, and ships everything needed for an LLM to redo the
judging blind (`stance_reference_judging/`, see section 6).

## 1. Method

**Unit.** The centre token of each of the 64 events in a tail (one SV, one polarity; 2,944 tails per
bank). Only *whole words* are judged: alphabetic tokens that are not word-internal pieces (69% of UWS
events, 66% of GSS events). Fragments, punctuation, digits and newlines are excluded, so this test is
about the words, not the structural slots described in the first report.

**Labels.** Every whole word with pooled count >= 10 across both banks (4,381 distinct lower-cased
words, 84% of whole-word events; 92-95% of those in the top-300 selective tails) got one label:

* **S - stance**, in seven sub-types: modal (`may might must should shall ought need gotta`),
  epistemic/hedging/focus (`perhaps allegedly potential apparently seems just only simply roughly
  actually exactly typically according ...`), degree (`very so absolutely huge tiny slightly ...`),
  evaluative (`good lovely fantastic awful significant useful unique solid relevant comprehensive ...`),
  attitude verbs (`think believe hope trust feel want love ...`), speech acts (`say claim argue suggest
  promise agree boast ...`), interjections (`yeah oh lol huh thanks please welcome ...`).
* **R - reference**: nouns of every kind (including connotation-heavy ones - `crisis`, `chaos`,
  `wonder`, `hope` as a noun are R), concrete/procedural verbs, classifying/physical adjectives,
  proper nouns, times, units.
* **F - function**: determiners, pronouns, prepositions, conjunctions, auxiliaries and contractions,
  quantifiers/numerals, negation, linking adverbials (`also however therefore`), subordinators
  (`because although`), frequency adverbs (`often sometimes always`), and the near-grammatical modals
  `will would can could`.
* **X - fragment/other**: prefixes, URLs, placeholders (`firstname`), non-English items.

Sizes: S 929 words (evaluative 488, epistemic 140, attitude 103, speech-act 91, degree 57,
interjection 41, modal 9), F 410, X 54, R 3,491. Labelling was done on one merged alphabetical list
without bank information; 87% of whole-word events in each bank ended up labelled (UWS 0.874, GSS
0.865), so coverage is balanced. Two deliberate conservatisms work *against* the essay's claim: all
nouns are R (so the "grandeur" vocabulary counts as reference), and `will/would/can/could` are F.

**Score.** Per tail, the stance ratio S/(R+S); tails with fewer than 10 labelled content words are
dropped. Banks are compared by Mann-Whitney on the per-tail ratios, by the event-pooled share
S/(S+R), and by Fisher's test on counts of stance-dominant (ratio >= 0.6) vs reference-dominant
(<= 0.4) tails.

## 2. Result

| tier | tails scored UWS/GSS | mean ratio | median | event-pooled S/(S+R) | Mann-Whitney p |
|---|---|---|---|---|---|
| all tails, both polarities | 2207 / 2273 | 0.198 / 0.163 | 0.091 / 0.051 | 0.196 / 0.163 | 9e-12 |
| all tails, selected polarity | 1099 / 1108 | 0.197 / 0.178 | 0.100 / 0.065 | 0.195 / 0.184 | 0.002 |
| top-100 selective, selected pol. | 29 / 49 | 0.261 / 0.141 | 0.192 / 0.000 | **0.320 / 0.169** | 0.015 |
| top-300 selective | 158 / 209 | 0.225 / 0.182 | 0.133 / 0.063 | 0.234 / 0.198 | 0.02 |
| top-600 selective | 394 / 456 | 0.234 / 0.175 | 0.143 / 0.065 | 0.239 / 0.186 | 4e-5 |
| top-60 broad, both polarities | 83 / 78 | 0.247 / 0.132 | 0.159 / 0.017 | **0.263 / 0.128** | 2e-4 |
| top-150 broad | 235 / 217 | 0.208 / 0.145 | 0.087 / 0.035 | 0.211 / 0.136 | 0.002 |
| layers 0-8, top-300 in band | 225 / 233 | 0.237 / 0.206 | 0.150 / 0.077 | 0.240 / 0.202 | 0.067 |
| layers 9-17, top-300 in band | 255 / 273 | 0.218 / 0.165 | 0.108 / 0.065 | 0.206 / 0.171 | 0.003 |
| layers 18-22, top-300 in band | 142 / 187 | 0.131 / 0.133 | 0.065 / 0.038 | 0.121 / 0.147 | 0.15 |

Read: UWS's tails carry a larger share of stance words at every tier, about 1.2x in the bulk and
about 2x at the extremes - the wordy top-selective tails (0.32 vs 0.17) and the broadest directions
(0.26 vs 0.13). The difference lives in layers 0-17 and is absent in layers 18-22, where the
selective tails are structural anyway (see the first report). GSS's tails carry more reference words
in absolute terms (top-300 selected-polarity content events: R 5,656 vs 4,047) and more of them are
wordy at all: of the top-40 selective tails only 5% of UWS's have 10+ content words against 30% of
GSS's; top-100: 29% vs 49%; top-300: 53% vs 70%.

**Which stance words carry it.** Shares of content events in the top-300 selected-polarity tails:

| sub-type | UWS | GSS |
|---|---|---|
| epistemic / hedging / focus | **7.5%** | 4.6% |
| evaluative | **10.8%** | 8.5% |
| degree | 1.9% | 2.1% |
| attitude verbs | 1.8% | 2.2% |
| speech acts | 0.8% | **1.6%** |
| modal | 0.45% | 0.47% |
| interjection | 0.15% | 0.26% |
| all stance | 23.4% | 19.8% |

The UWS excess is *qualification* - how certain and how good (`potential`, `perhaps`, `allegedly`,
`just`, `almost`, `nearly`, `significant`, `comprehensive`, `perfect`, `pretty`) - not speech or
attitude: GSS has proportionally more verbs of saying and thinking (`think`, `believe`, `agree`,
`argue`, `trust`, `prove`). Dropping any one sub-type leaves UWS > GSS (event-pooled, top-300:
no drop 0.234/0.198; without epistemic 0.172/0.159; without evaluative 0.141/0.123; the other drops
change little), so no single word class manufactures the result, though epistemic and evaluative
words account for most of the gap.

One refinement I floated while reading - that GSS's speech/attitude verbs are *reported* third-person
acts rather than the speaker's own - is wrong: within ~60 characters before an attitude or speech-act
word, `I/we/you` appear in 52% of GSS's top-300 events vs 36% of UWS's (all events: 42% vs 43%).
Both banks' stance verbs are mostly the speaker's voice; GSS simply has more of them.

**Exemplar stance-dominant tails found by the lexicon** (top-300, 20+ content words), as a check that
the labels pick out what the essay meant:

* UWS: `L09_SV58/neg` 0.98 (`just` `simply` `whatever` `certainly` `definitely`); `L16_SV54/neg` 0.97
  (`simply` `wonderful` `just` `useful` `enjoy` `fun` `certainly` `lovely`); `L01_SV18/neg` 0.92
  (`significant` `comprehensive` `substantial`); `L08_SV57/pos` 0.91 (`almost`:27 `nearly` `incredibly`
  `fairly` `equally` `overly` `perfectly`); `L04_SV61/pos` 0.90 (`potential`:32 `potentially` `possibly`
  `optimal` `possible` `palpable`); `L09_SV42/neg` 0.80 (`perfect`:23 `pretty`:14 `favorite` `suitable`
  `perhaps` `especially` `pleasant`); `L07_SV58/pos` 0.80 (`pretty` `relatively` `comparatively`).
* GSS: `L05_SV13/neg` 1.00 (`'t`:35 `seem` `meant` `seemed` `seems` `may` `could` `might`); `L19_SV36/neg`
  0.95 (`just`:51); `L00_SV26/pos` 0.93 (`think`:17 `believe`:17 `Feel` `know`); `L08_SV56/neg` 0.88
  (`lovely` `valuable` `love` `vital` `sure` `relevant`); `L18_SV09/neg` 0.79 (`critical` `exceptional`
  `typical` `essential`); `L12_SV51/neg` 0.77 (`you know` `grateful` `agree` `pleased` `fortunate`);
  `L15_SV59/neg` 0.74 (`truly` `quite` `clear`); `L19_SV55/neg` 0.71 (`far` `really` `quite` `accurately`
  `truly` `precisely` `honestly`).

GSS has real stance directions. The claim that survives is proportional, not categorical.

## 3. What did not work: categorical tail labels

Labelling each tail as a whole (REFERENCE if ratio <= 0.4, STANCE if >= 0.6, MIXED between,
STRUCTURAL if under 10 content words) and counting does **not** separate the banks. On the 310-item
blind sample: UWS 84 reference / 9 stance / 18 mixed / 44 structural; GSS 107 / 15 / 13 / 20 (Fisher
p = 0.66). Stance words are a minority in nearly every tail of either bank; UWS's tails are more
*mixed*, not more often stance-*dominant*. The essay's "tones vs nouns" reads too strongly as a
dichotomy; the defensible version is "more tone in UWS, more noun in GSS, in proportion".
Consequently the blind test for an LLM has to be graded (count stance and reference words), not a
single label per tail. Both formats are provided; the categorical one is there as a control.

## 4. My baseline on exactly what the blind judge will see

Applying my word labels to the up-to-20 listed words of each sample item (word types, counts
ignored - the same view an LLM gets in `blind_tails_graded.md`): all items 0.223 vs 0.176 (p = 0.02);
selective top-100 tier 0.227 vs 0.176 (p = 0.08); broad tier 0.231 vs 0.091 (p = 0.005); mid-random
tier 0.200 vs 0.231 (p = 0.74). The gap is at the extremes and not in the middle of the selectivity
range, as in the full data.

## 5. Caveats

* The lexicon is mine and the essay was mine; a shared bias could inflate the gap. That is what the
  blind kit is for. The main mitigation already in place is that labels were assigned on one merged
  word list with no bank information, and that the bank-difference was computed only afterwards.
* Word-level labels ignore context: `just`, `like`, `pretty`, `may`, `solid`, `critical`, `fine` are
  polysemous. Capitalised `May/March/Will/Bill/Mark/Frank/Grace/Faith/Hope` are forced to R.
* Boundary decisions matter and were made by rule, not per bank: nouns are always R; frequency and
  linking adverbs and `because`-type subordinators are F; `will/would/can/could` are F while
  `may/might/must/should/shall/ought/need` are S. Moving `will/can` into S or `because/however` into S
  would raise both banks' stance shares; it is not obvious which bank it would favour.
* Only 29 UWS vs 49 GSS tails in the top-100 selective tier have enough words to score; the headline
  0.32 vs 0.17 there rests on those tails. The top-600 and all-tails rows are the robust ones.
* 13% of whole-word events (rare words) are unlabelled in both banks.

## 6. Blind replication kit (`stance_reference_judging/`)

Give an LLM the blind file, save its answers, run the matching scorer; then compare with my baselines.

1. **Word level (primary test).** `blind_words.txt` - 4,381 words, alphabetical, with instructions;
   the LLM outputs `word<TAB>R|S|F|X`. Score with `python score_word_judgments.py its_labels.tsv`
   (uses `tail_words.jsonl`, the per-tail whole-word lists for both banks; prints the same tier table
   as section 2). My labels: `word_labels_mine.tsv`; `python compare_word_labels.py word_labels_mine.tsv
   its_labels.tsv` prints the agreement matrix.
2. **Tail level, graded.** `blind_tails_graded.md` - 310 shuffled items, source hidden; the LLM outputs
   `id<TAB>n_stance<TAB>n_reference`. Score with `python score_blind_graded.py its_counts.tsv`. My
   baseline: `blind_tails_graded_mine.tsv`.
3. **Tail level, categorical (control).** `blind_tails.md` - same items; the LLM outputs
   `id<TAB>REFERENCE|STANCE|MIXED|STRUCTURAL|OTHER`. Score with `python score_blind_judgments.py
   its_labels.tsv`. My baseline: `blind_tails_labels_mine.tsv`. Expect no separation (section 3).

`blind_tails_key.tsv` is the hidden key (id -> bank, SV, polarity, tier); do not show it to the judge.
Sample composition per bank: 100 most selective tails with >= 60% whole-word centres (selected
polarity), both polarities of the 15 broadest SVs, 25 random mid-selectivity wordy tails.

The prediction to check: under an independent labeller, UWS's stance share exceeds GSS's overall and
at the top-selective and broad tiers, by a factor near 1.2 overall and near 2 at the extremes, with
the excess concentrated in epistemic and evaluative words; and there is no separation in layers 18-22
or in a single-label-per-tail format.
