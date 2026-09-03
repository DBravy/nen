# Top-context differences: `unrealized_words_selectivity` vs `g_scan_sparse`

Blind comparison of the two scans' *outputs only* (`metadata.json`, `selectivity_rankings.csv`,
`top_contexts.jsonl`, `unembedding_neighbors.jsonl`). No producing script was read.
Abbreviations: **UWS** = `unrealized_words_selectivity` (bank `unrealized_words_fineweb/directions`),
**GSS** = `g_scan_sparse` (bank `g_bank/directions`).

## 0. What is being compared

The two runs are identical except for the direction bank: same model (gpt-oss-20b), same lens, same
23 layers x 64 SVs = 1,472 candidates, same FineWeb slice (2,000 docs, 441,490 content tokens,
256-token crop), same seed, same 64 top contexts per polarity. So every difference below is
attributable to the directions, not the data. Each SV contributes 64 positive + 64 negative extreme
events; "selected polarity" = the tail the CSV scores as more selective.

## 1. Headline

**The two banks disagree about *what kind of thing* an extreme, selective direction is.**

* **UWS's most selective SVs sit on *between-word prediction slots*.** The centre token is a
  semantically empty separator or closed-class word (a bare space, `(`, `-`, `.\n`, `\n`, ` the`,
  ` was`, ` It`) and what the 64 events share is the *class of token that must come next*: a digit
  string, a new line/paragraph, a content word after a determiner/copula, or the inside of a
  quotation/parenthesis. These slots recur in most documents, so the directions are sharp *and*
  ubiquitous (z>5 in hundreds to ~1,900 of 2,000 docs).
* **GSS's most selective SVs sit *inside words*.** In late layers the centre token is the first
  sub-word piece of a longer word (` Ch`|eryl, ` Ab`|normal, ` Kr`|inkov, ` i`|Pad, ` Trans`|ferring)
  and each SV is anchored to a spelling/phonological onset neighbourhood while the full words are
  diverse. In early layers they are typographic-symbol features (curly quotes, `’s`/`–`, ` &`,
  `.”\n`). These are lexical-form features: sharp but sparse (z>5 in ~20-450 docs, median 50).
* Cross-bank, the extreme events barely coincide (per-layer Jaccard ~0.04; best cross-bank match for
  a top-40 SV is usually J<0.1), i.e. these are different features, not re-rankings of shared ones.
* The **broadly-activated** SVs (top by `mean_abs_cosine`) look *similar* across banks: sign-constant
  axes contrasting ordinary word-initial tokens against word fragments/garbage, or prepositions/quote
  marks against content nouns. The banks differ in their selective tails, not their bulk.

The rest of this file is the evidence.

## 2. Where selectivity lives

| | UWS | GSS |
|---|---|---|
| max `tail_selectivity_score` | 1.34 | 0.95 |
| median / q75 over all 1,472 SVs | 0.18 / 0.55 | 0.09 / 0.27 |
| # SVs with score > 0.5 | 395 | 225 |
| layers of top-100 selective (L18-22 : L9-17 : L0-8) | 71 : 12 : 17 | 58 : 16 : 26 |
| top-60 selective: docs with z>5 peak (median [q25,q75]) | 86 [61, 210] | 50 [22, 79] |
| top-5 selective: docs with z>5 peak | 721 / 1930 / 1033 / 1217 / 1612 | 198 / 186 / 299 / 134 / 172 |
| top-60 selective: excess kurtosis (median) | 0.93 | 0.66 |
| top-60 selective: largest-centre-token share (median) | 0.30 | 0.12 |

Per-layer, UWS selectivity climbs steeply into L19-22 (56 of 64 L22 SVs score > 0.5); GSS is flatter
(36 at L22) and has real mass at L1-L3 (e.g. `L03_SV19`, `L02_SV01`, `L02_SV62`, `L01_SV01` in its top 6).

## 3. Family classification of the selective SVs

Every event was classified by a simple rule on the centre token and the character after it
(one family per SV if >= 50% of its 64 selected-polarity events agree, else MIXED):

* `DIGIT_next` - centre is whitespace/punctuation and the next character is a digit (the tokenizer
  splits ` 348` into ` ` + `348`, so this is a real position where digits must be produced)
* `NEWLINE_center` - centre token contains `\n`
* `FUNCTION_word` - centre is a complete stop-word (` the`, ` of`, ` was`, ` It`, ...)
* `OPEN_quote_paren` - centre is an opening quote/bracket
* `WORD_ONSET` - centre is alphabetic and the next character is a letter (the word is unfinished)
* `CONTENT_word` - complete alphabetic non-stop-word
* `OTHER` - mixed symbols, e.g. `’s`, `“This`, non-Latin

| family | top-40 UWS | top-40 GSS | top-100 UWS | top-100 GSS | **late L>=18, top-40** UWS | GSS | early L<=8, top-40 UWS | GSS |
|---|---|---|---|---|---|---|---|---|
| DIGIT_next | **8** | 0 | 10 | 0 | **8** | 0 | 0 | 0 |
| NEWLINE_center | **9** | 1 | 14 | 3 | **9** | 1 | 4 | 1 |
| FUNCTION_word | **9** | 1 | 26 | 2 | **11** | 1 | 0 | 0 |
| OPEN_quote_paren | 3 | 1 | 4 | 2 | 1 | 0 | 2 | 1 |
| WORD_ONSET | 3 | **21** | 8 | 35 | 5 | **25** | 3 | 0 |
| CONTENT_word | 1 | 9 | 17 | 42 | 1 | 10 | 26 | 26 |
| OTHER / PUNCT_other | 0 | 4 | 1 | 6 | 0 | 0 | 0 | 8 |
| MIXED | 7 | 3 | 20 | 8 | 5 | 3 | 5 | 4 |

Reading: 26 of UWS's top-40 are slot families (digit-next, newline, function word); 21 of GSS's
top-40 are word-onset. The split is *not* a layer-composition artefact - restricted to L>=18 it is
even sharper (28 vs 2 slot-family SVs; 5 vs 25 word-onset). In early layers both banks' top SVs are
mostly content-word lexical features and look alike, except that GSS's early tail also contains
typographic-symbol SVs (OTHER/PUNCT = 8 vs 0).

Background rates over all 1,472 SVs: `DIGIT_next` 19 (UWS) vs **1** (GSS) - the number-slot feature
essentially does not exist in the g bank; `WORD_ONSET` 74 vs 107; `NEWLINE_center` 21 vs 34;
`FUNCTION_word` 142 vs 91. So for newline/function-word slots GSS *has* such directions but they are
not its most selective ones, whereas the digit slot is genuinely absent.

## 4. UWS selective SVs, by family (exemplars; ⟦ ⟧ marks the centre token, ⏎ = newline)

**Number comes next** - `L22_SV07` (sel 1.34, centre ` `:37, ` (`:8, `-`:5, ` $`:3), `L22_SV05`,
`L22_SV34` (` `:54/64, next char digit 100%), `L22_SV02`, `L22_SV01`, `L22_SV12`, `L21_SV16`, `L21_SV34`:

```
c.80 mi⟦ (⟧130 km) long          team total of⟦ ⟧348, following      Phone: (819) 376⟦-⟧0005
Volume⟦ ⟧19, No. 5, May 2002    Wednesday, December⟦ ⟧20, 2006      by Paul on 09⟦/⟧03/09
article_news?id⟦=⟧161294454      Price (€): 116⟦.⟧94                 BDNF (0⟦.⟧75 microg/0.5
```
Five of these (`L22_SV01/02/05/07/34`) share events with each other (pairwise J up to 0.42) - the
number slot occupies a multi-SV subspace of the UWS L22 bank, not one direction.

**A new line/paragraph comes next** - `L22_SV16` (`.\n`:29, `\n`:29), `L22_SV04`, `L21_SV06`,
`L22_SV21`, `L22_SV06` (`\n`:57), `L22_SV42`, `L22_SV11`, `L01_SV01` (`.”\n`:43):

```
broader implications⟦.⏎⟧The simple answer      Jogger Set⟦⏎⟧Got a casual night out
– 17.542⟦⏎⟧8. Prize Winning Dash               situations⟦.⏎⟧- Consider someone who
```
The next character is an upper-case letter in 58-81% of events (72-81% for all but `L22_SV11`), i.e. the start of the next sentence/item.

**Function word whose complement is due** - `L21_SV19` (` of`,` a`,` the`,`The`,` in`,` for`),
`L21_SV07` / `L21_SV24` (` the`:26-34, ` The`, ` a`), `L21_SV36` (` was`:24, ` be`:11, ` is`:11,
` being`, ` are`, ` been`), `L21_SV21` (` It`,`It`,` it`,` What`,` which`), `L22_SV58`, `L21_SV51`, `L18_SV63`:

```
Compendium⟦ of⟧ knowledge of economic sciences     the motto of⟦ the⟧ Barbco BD40HP
This youthful juice branding⟦ was⟧ designed        the Salomon rig will⟦ be⟧ parked
(Neil Diamond) (1969)⏎⟦It⟧ was originally titled   a beta blocker⟦ which⟧ is sometimes effective
```

**Opening quote / bracket, content due** - `L19_SV03` (` (`:31, ` "`:11, ` '`:6), `L02_SV25`
(` (`:62/64), `L00_SV08` (` ‘`:61/64):

```
(Neil Diamond)⟦ (⟧1969)     most patients⟦ (⟧64%)     they call it⟦ "⟧Johns."     said⟦ ‘⟧really’?
```

**Document-initial, unpredictable continuation** - `L21_SV04` (sel 1.27) and `L20_SV04` (1.20),
which share events; almost all within the first ~15 tokens after `<|startoftext|>`:

```
<|startoftext|>5s / 5c /⟦ ⟧5 / SE, iPad     <|startoftext|>415-767⟦-⟧6905 | email     KEP⟦ (⟧ First Prize)
<|startoftext|> 2013⟦.B⟧orn December 18    <|startoftext|> as “⟦d⟧oyenne of humanity”
```

One outlier worth noting: `L08_SV02` (sel 0.82) fires on *duplicated or garbled* words -
`how well⟦ well⟧-organized`, `a hard copy⟦ copy⟧`, `we seem to get be⟦ getting⟧`, `still is⟦ still⟧ banging`.

Position: UWS slot families sit early in the crop (median token index: DIGIT_next 16, FUNCTION_word
12, NEWLINE 40) vs CONTENT_word 69 - titles, dates, addresses and list headers are number/newline
dense, and determiners near a document start have minimal context. Activation magnitude does not
itself trend with position within an SV's 64 events (median Spearman rho between |activation| and
position is 0.01-0.08 for the three UWS slot families and between -0.16 and +0.23 for every family in
either bank), so this is about where the slots occur, not stronger firing early.

## 5. GSS selective SVs, by family

**Unfinished word / spelling onset (late layers; 25 of the late-layer top-40).** Each SV prefers a
neighbourhood of onsets, while the words themselves are diverse (largest full-word share ~0.03):

| SV | centre pieces | full words in the tail |
|---|---|---|
| `L21_SV20` (sel 0.95, neg) | ` Ch`:19 ` ch`:9 `Ch`:7 ` Kar` ` ph` ` Cy` ` ep` | Cheryl, Chazelle, Chabad, Ch'i, Chocks, Chabot, Chori, Chane, Churning, Chosen, chins, chimes, Engraved, epigenetically, apologist, Apology, Spiritual, crux |
| `L22_SV27` (pos) | `Ab` ` Ab` `Add` `Adding` `A` `Al` | Abandoning, Abnormal, Abode, Abundance, Abominable, Abt, Abendstern, Add, Adding, Added, Alleycat, Amendment |
| `L22_SV05` (neg) | ` trans` ` Trans` ` Mon` ` Poly` ` hom` ` micro` | ultrasonographer, Monahan, Metamorphosis, Autocallable, Monitors, homestays, Granular, Transferring, Polycom, Volleyball, depersonalization, Extrajudicial, epithelium, monotherapy |
| `L22_SV03` (pos, sel 0.79) | ` bl` ` und` ` kn` ` Kr` ` tre` ` ta` `Fre` ` pl` | Krinkov, knick-knacks, treacherous, taunt, blushing, knuckleball, Fredu, taillights, plunder, undetectable, Blazing, knurled, ambidextrous, untouchable, Frederick, undulating |
| `L22_SV29` (pos) | ` i`:31/64 | iPad, iPod, iPhone, iTunes, iWeb, iSCSI, iSenpai, iSoftBet |
| `L21_SV55` (pos) | ` San` `Vol` `Sur` ` Un` ` Str` | Surrounding, Scallop, Surplus, Volumes, Burrell, Volcano, Quantitative, Virally, Stripes, Voluminous, VolunTours, Burial, Burrows, Verlander, Sonography, Synchronize |
| `L22_SV53` (neg) | ` Ar`:14 ` fl` ` cr` ` ar` `Vol` ` aw` | Arched, Aria, Aramid, Artery, Arancini, Arion, Arri, crumpled, awoke, flared, flaked, flaking, Volcano |
| `L22_SV13` (pos) | ` ph` `Bre` ` che` `Mel` `bre` ` diss` | chelsea, Melinda, Mel, peacock, breathing, breathes, Memorials, pollute, Scratches, phallic, phasor, Pepsi, Peep, melty, Brexit |
| `L22_SV19` (neg) | ` tre` ` du` ` inf` ` Mill` ` Fo` ` Son` | treacherous, trepidation, Foxtrot, Foote, duffel, duvets, Sonography, infomercials, Krinkov |
| `L22_SV48` (pos) | ` Bar` ` Bill` ` Mem` ` Bio` `Bi` ` Gra` | perfunctorily, Memovox, BioLogic, WiFi, DiS, Barneebrown, Gray-hat, Biological |

Also `L21_SV50` (fr/gr/Pe/fe: fracking, frugal, feisty, Feathers, Berlioz), `L22_SV43` (fr/bl/gr:
fracking, frugal, purview, Tempted, ungrateful), `L22_SV21` (Ch/Sc/Kh/Kr: Scallop, Krinkov, Escapes,
Entrees, Khiyron, Scuppernong), `L22_SV15`, `L22_SV20`, `L21_SV48`, `L21_SV62`, `L22_SV35`, `L22_SV55`,
`L21_SV16`, `L20_SV60`. In 13 of GSS's top-40 SVs the next character is a lowercase letter in
>= 80% of events (UWS top-40: none; its highest is 0.75). These SVs do not share events with one another (mean pairwise J = 0.007 among the
top-40; no cluster among the word-onset SVs) - they tile onset space rather than duplicate a direction.

For contrast, UWS's few late word-onset SVs (`L22_SV36/38/49`) are about rare proper names and
hyphen/initial pieces (Khaos, Nerf, Shuchman, Saracens, Ajdar, Akwa; `over-st`, `fire-b`, `All-S`),
with only `L22_SV49` looking onset-like (D-: Dis, Dusk, Ducting, Drought, Darya, Dissertation).

**Typographic symbols (early layers).** `L03_SV19` (sel 0.95): Unicode apostrophes/dashes `’s`:15,
`–`:7, `’t`:7, `’`:7, `’ve`:4, `‐`:3, `’re`:3 (`Early⟦‐⟧Life`, `self⟦–⟧contained`, `n⟦’t⟧`, `you⟦’ve⟧`).
`L02_SV01`, `L02_SV62`, `L01_SV01`, `L02_SV27`, `L03_SV01`: curly opening quotes and quote+word merges,
usually line-initial (`⏎⟦“This⟧ was an interesting`, `⟦“I⟧ agree`, `⟦“We⟧ should`, `⟦ ‘⟧you have
permission`); these five share events with each other (J up to 0.47) and are the one GSS cluster.
`L02_SV33`: ` &`:50/64 (`ideas,⟦ &⟧ feedback`). `L01_SV03`: line endings ending in typographic
clusters `.”\n`:34, ` ...\n`, ` […]\n`, `!!!\n`, ` :)\n`.

**Lexical word classes.** `L07_SV63` (lovely, solid, soul, fulfill, suit, beautiful, truly, bella),
`L04_SV27` (pertaining, relevant, substantial, adversely, adverse, prevalent, subsequently, abreast),
`L01_SV29` (innovative, relevant, specialised, integrated, industrialising, enriching),
`L09_SV59` (oyster, choking, trigger, lice, frosting, candidiasis, Terminator, EB-5).

**Slot-like GSS SVs (rare).** `L22_SV64` fires on function words *inside multiword proper names*
(`Airport BOS and landing⟦ at⟧ Atlanta Hartsfield-Jackson`, `School of Medicine⟦ in⟧ Baltimore`,
`Ministry⟦ of⟧ Information and Broadcasting`, `advance to⟦ the⟧ semi-finals`). `L22_SV59` is GSS's
only number-slot SV (`[Bug⟦ ⟧778111]`, `May⟦ ⟧2002`, plus mojibake `Ã`); its best cross-bank matches
are exactly UWS's number-slot SVs (`L22_SV05`, `L22_SV34`, `L21_SV16`).

## 6. Per-event statistics (selected polarity, top-60 selective per bank; per-SV medians)

| feature | UWS | GSS |
|---|---|---|
| share of events on non-word centre tokens (punct/newline/space/digit) | 0.35 (27 SVs > 0.5) | 0.02 (11 SVs > 0.5) |
| share with a digit immediately after the centre | 0.03 (13 SVs > 0.3) | 0.00 (1 SV > 0.3) |
| share on capitalised centre tokens | 0.04 | 0.29 (18 SVs > 0.5) |
| share whose centre token is preceded by a newline | 0.00 | 0.06 |
| largest single centre token share | 0.30 | 0.12 |
| pooled centre-token types: punct / newline / space | 17% / 16% / 11% | 7% / 2% / 1% |
| pooled: next char is a lowercase letter (word continues) | 33% | 48% |
| median token index in the 256-token crop | 15 | 74 |
| share of events at index <= 10 | 0.33 (21 SVs > 0.5) | 0.12 (1 SV > 0.5) |
| unique docs among 64 events | 55 | 59 |
| dominant next-char class (top-40 SVs) | cont_Upper 13, sp+lower 11, DIGIT 9, cont_lower 6 | cont_lower 21, sp+lower 14, cont_Upper 3, DIGIT 1 |

Both banks' tails are document-diverse (55-59 distinct URLs per SV), so neither is a single-source
artefact. UWS SVs are more concentrated on outlier positions hit by many SVs at once (median 0.10 of
events on positions hit by >= 10 SVs, q75 0.25, vs 0.06 for GSS); this is driven by the number-slot
cluster (0.44-0.72 for `L22_SV01/02/05/07/34`).

## 7. Cross-bank overlap

* Unique extreme positions: UWS 93,112, GSS 92,024, intersection 37,618 (J = 0.26 overall) but only
  J ~ 0.03-0.06 within any single layer: the banks mostly light up different tokens at the same depth.
* Best cross-bank match (Jaccard of 64-event sets) for UWS's top-40 selective SVs: 36 of 40 have
  J < 0.15. The four exceptions: `L01_SV01` (line-end `.”\n`) <-> GSS `L01_SV03` (J 0.25);
  the number-slot SVs `L22_SV05` / `L22_SV01` <-> GSS `L22_SV61` (0.20 / 0.17), a GSS direction whose
  own selectivity is only 0.42; and `L00_SV08` (` ‘`) <-> GSS `L12_SV23` / `L21_SV30` (0.16 / 0.15). For GSS's top-40: `L02_SV62` (`“I`/`“This`)
  <-> UWS `L01_SV17` (0.31) and `L00_SV13` (0.24); `L01_SV03` <-> UWS `L01_SV01` (0.25). Everything
  else, in particular every GSS word-onset SV and every UWS number/newline/function-word SV, is
  unmatched (J <= 0.12).
* Most-hit single positions differ in kind: UWS's are early-document tokens (`The Goon⟦ies⟧`,
  `you’re⟦ gonna⟧`, `a⟦ Roman⟧ triumphal`, `multif⟦aceted⟧`, `501⟦(c⟧)(4)` - 7 of the top 15 within
  13 tokens of `<|startoftext|>`); GSS's are quote-openers and line ends (`⏎⟦“This⟧` x3, `by⟦.’⏎⟧`,
  `with⟦ […]⏎⟧`, `⟦((((⟧1))))`).

## 8. Broadly-activated SVs (top by `mean_abs_cosine`) - similar across banks

Family counts over the top-20 broad SVs x 2 polarities: UWS CONTENT 20 / MIXED 11 / WORD_ONSET 5 /
FUNCTION 4; GSS CONTENT 20 / MIXED 8 / FUNCTION 8 / NEWLINE 4. Both banks' broad directions are
sign-constant offsets (`positive_rate` ~ 0 or 1, |mean|/std 2-4.5), persist across adjacent layers
with the same nearest-unembedding token, and contrast a tokenisation-regularity or word-class axis:

* UWS `L05-L08_SV03` (nearest unembed `_arr`/`almacen`): + tail = mojibake and mid-word garbage
  (`𝗖𝗵𝗲𝗳`, `Ã¢â‚¬`, Cyrillic, `Skane⟦at⟧eles`); - tail = Latinate adjectives/participles
  (potential, considering, monitoring, critical, historical, initially, allegedly, unparalleled).
* UWS `L09-L15_SV10` (`nos`/`336`): + = ` no`, ` person`, ` leave`, ` say`, ` way`; - = word-internal
  suffix pieces (Ther⟦most⟧at, Llew⟦elly⟧n, rock⟦abil⟧ly, Explor⟦atorium⟧).
* GSS `L03-L09_SV01/02` (`Bron`/`아래`): - = ` of` in 44-58 of 64 events (plus for/by/in/from);
  + = C-initial brand/proper nouns (Costco, cacao, chocolate, Casino, COPD, CNN, Chiropractic,
  Croatia, Congo, Consensus, Chateau, plus jewellery, Poker, DSLR, Beyoncé).
* GSS `L08-L17_SV02` (`Tales`): + = opening quotes/bullets (` “`, `“This`, `“As`, `•`, ` […]\n`);
  - = attributive nouns in compounds (legal profession, data centers, market capitalization,
  health crisis, job-related, clinical, physical).
* GSS `L07_SV04`: + = emphatic line endings (`.”\n`, ` ...\n`, `!!!\n`, `((((`); - = short
  modifiers in compounds (easy, high, new, light, close, front, end).

So GSS's broad SVs are somewhat more lexically peaked (a single token ` of` carries 70-90% of a tail)
while UWS's are more diffuse (largest share ~0.05), but the *content* - word vs fragment, function vs
content, quote vs noun - is the same kind of thing in both banks.

## 9. Caveats

* Family labels come from a crude character rule applied to 64 events per SV; MIXED is common
  (UWS 20/100, GSS 8/100). The exemplars, not the labels, are the primary evidence.
* Only the selected polarity and the 64 retained extreme events per tail were inspected; the
  bulk distribution is only seen through the CSV summary statistics.
* Documents are cropped to 256 tokens, so "early in document" means early in the crop; both banks
  see the same crops.
* Nothing about how the banks were constructed was used; the singular-value spectra differ a lot
  (UWS sigma1 falls from 1,596 at L0 to 33 at L22; GSS from 722 to 115) so the SV *rank* numbers are
  not comparable across banks and were not used for anything except naming.

## 10. One-paragraph summary

Both banks agree on the broad, always-on axes of the residual stream. They part ways in the
selective tails. UWS's selective directions are *positional/structural*: they fire on the token
before a number, at a line break, on a determiner or copula, or at an opening quote - places
defined by the category of the token that has not yet been produced, and they are redundant (several
SVs per slot at L22) and near-universal across documents. GSS's selective directions are
*lexical/orthographic*: they fire on the first piece of an unfinished word, organised by spelling
onset (Ch-, Ab-, Ar-, tre-/kn-/Kr-, trans-/Mon-/Poly-, i-), or on specific typographic symbols, and
each covers a small, distinct slice of the corpus. If one wanted a single discriminating measurement
it is the digit-slot family: 19 SVs in UWS (including its five most selective), one in GSS.
