# G-bank blinded hypotheses — committed before unblinding
# Written from review.md + bank_stats.txt only. Do not edit after candidates.csv arrives.
#
# INTEGRITY NOTE, stated up front: the random arm reused scan_rand_r0, so its blind IDs are
# identical to round 1's (same hash inputs). 23 IDs collide with my round-1 hypothesis sheet and
# are therefore KNOWN to be the random arm — not guessed. They are marked [KNOWN-RANDOM] and my
# re-reads of them double as a test-retest check on my own blind reading. The genuine blind
# comparison in this round is G-real vs G-rotated (134 unknown IDs).
#
# Format: id L pol | hypothesis | conf 0-3 | arm (real/rot/RANDOM)
# Shared-context clusters referenced below:
#   OUTLIER-C (new, early layers): "crania...self-loathing", "Mohanlal seven counters", "Jamaica 2009
#     plan", "/economists provided these analyses", "data...size and shape", "Kampfflugzeuge Im 2"
#   OUTLIER-A (round 1, mid layers): "served the army", "kittens of various ages", "plethora of
#     ideas", "gawking at a car accident", "hard to find a bad drink in Decatur", "rapprochement"
#   OUTLIER-B (round 1, late layers): "pages 427-446", "75% Nylon", "Bug 774842", spec-sheet docs

## Layer 0
0b2d20fb L00- | doc-initial open-quote ⟦ "⟧ + OUTLIER-C; quote-onset formatting | 1 | rot
1c30c6a7 L00+ | 'survey/search' lexical pair (round-1 re-read: same) | 2 | RANDOM [KNOWN]
25824990 L00- | lexical ' extensive' (round-1 re-read: same) | 2 | RANDOM [KNOWN]
3134adb7 L00- | plural nouns + 'adt' NAMECONT (round-1 re-read: same, weak) | 1 | RANDOM [KNOWN]
5a3d9c00 L00- | proper-noun continuation subwords (round-1 re-read: same) | 3 | RANDOM [KNOWN]
b973ccff L00+ | rare-word/place continuation subwords (round-1 re-read: same) | 3 | RANDOM [KNOWN]
bbe28469 L00- | capitalized institution nouns (Institute, League, Department, Index, Regional) | 2 | real

## Layer 1
009899b4 L01- | OUTLIER-C carrier; all marks lost to line boundaries; no feature | 0 | rot
27c9bd10 L01- | opening-quote variants at doc start ("We, ', '', ', «) + OUTLIER-C | 1 | rot
863eccfb L01- | doc-initial scare-quote ⟦ "⟧ ("best tasting", "selfie", "zero-tolerance") — quoted-term onset | 2 | real

## Layer 2
06c2f7af L02- | rare-word continuation fragments (round-1 re-read: same) | 2 | RANDOM [KNOWN]
18485283 L02- | lexical ' significant/distinct' (round-1 family e84758f0 was 'significant') | 3 | real
2d60ef9f L02+ | quote-onset + OUTLIER-C mixture | 1 | rot
447b493a L02- | doc-initial quoted first-person ⟦"I⟧/⟦"We⟧ — testimonial-quote opening (round-1 9875f821 family) | 2 | real
4c9c7757 L02+ | AMPERSAND ⟦ &⟧ in commercial listing prose | 3 | real
7d8ec113 L02- | OUTLIER-C carrier; single readable mark | 0 | rot
c33ba7cd L02- | .com/URL + misc (round-1 re-read: same, weak) | 1 | RANDOM [KNOWN]
c71470e8 L02+ | word-continuation fragments (round-1 re-read: same) | 2 | RANDOM [KNOWN]
d8ecfdd4 L02- | NAMECONT + misc (round-1 re-read: same) | 1 | RANDOM [KNOWN]
dde9c4e6 L02- | typographically unusual quote-mark variants at doc start (', ", "In, ") | 2 | real

## Layer 3
0bc2fff3 L03+ | 'information/Information' lexical; most marks line-lost | 1 | rot
37e91288 L03- | 'sub-/subject' + taxonomy register (subscales, species, subrogating) | 1 | rot
3bd280b6 L03- | lexical ' requirements/guidelines' (compliance nouns) | 2 | real
3d36c881 L03+ | BOS-adjacent misc; no coherent feature | 0 | rot
6201bd84 L03+ | capitalized facility names (Laboratory, Park, Nuclear Medicine); opp pole = ' may' x3 (!) | 1 | rot
6a473861 L03+ | capitalized Research/institutional names at BOS | 1 | rot
6aabac50 L03+ | misc web/commerce; no feature | 0 | rot
81ac3bcb L03+ | archaic/typographic quote variants (", ', '^ OCR) | 1 | rot
9359d2c1 L03- | formal-register adverbs/nouns (requisite, rigorous, adequately, executive) — erudite family | 2 | real
ae328150 L03+ | literate idiom phrasing (hallmarks, bar none, evident) | 1 | rot
b77cd46d L03+ | NONSTANDARD UNICODE hyphens/apostrophes (en-dash-as-hyphen, U+2010, split "n't", mojibake) | 2 | real
c053a450 L03+ | TRANSPORT/civic-infrastructure cluster (transportation x5, municipal) [family T] | 3 | real
ce5b3125 L03+ | biomedical-scientific morphemes (glutathione, phospho-, pharmac-, prophylactically) | 3 | real
de06f7db L03+ | entertainment-franchise titles (Greenlight, Croods, Star Trek, Čapek) | 1 | rot
f66f7e08 L03- | tech-spec plurals (QWERTY x2, workloads, keywords, targets) | 1 | rot

## Layer 4
0fa91e4b L04+ | news-report register (CNN, WMAR, Comcast); marks line-lost | 1 | rot
2a6b4822 L04- | CURLY-APOSTROPHE contractions ('d, 're, .') + encoding junk [family APOS] | 2 | real
697df60b L04+ | bureaucratic-Latinate participles (pertaining x3, adversely x2, subsequently, prevalent) | 3 | real
9f8765be L04- | formal Latinate com-/con- vocabulary (complementary, congregate, exclusively) | 2 | real

## Layer 5
4afb109b L05+ | -ing marketing nominals (ranking, rating, branding) + spec junk | 1 | rot
e3fe0b51 L05+ | formal participial connectives (regarding, pertaining, acknowledging, assenting to) | 2 | real
ef6d368e L05+ | lexical ' different/various' | 3 | real

## Layer 6
0b3d3f2a L06- | bodily-harm/medical-complaint topics (herniated, abuse, discharge, died, smell) | 1 | rot
0fbe840b L06- | -tion(s) nominalizations (miscalculation, determinations, reductions) | 1 | rot
28a10f96 L06- | no coherent feature; OUTLIER-C adjacent | 0 | rot
acabe9dc L06- | BOS first-content-token artifact (round-1 re-read: same artifact) | 0 | RANDOM [KNOWN]
c3f2b69e L06- | BOS + ' at' lexical; position-flavored | 1 | rot
e2fe49e3 L06- | professional-services misc | 0 | rot

## Layer 7
043fec0c L07- | promotional positive-evaluatives (lovely, solid, fulfill); opp pole em-dash parentheticals | 1 | rot
16c80b80 L07- | THRESHOLD/boundary nouns (transition x3, peak x3, intersection) | 2 | real
59a86ae7 L07+ | vivid-adjective continuation fragments (gargantuan, multifaceted, knotty, lackluster) | 1 | rot

## Layer 8
0185318e L08- | hospitality-brochure register (courteous, attentive, well-designed, comprehensive) | 2 | real
093a0d24 L08+ | hobbyist-technical misc (thyroxin, flycasting, Ch'i) | 0 | rot
1c668380 L08+ | intensifiers (extremely x2, incredibly, incredible, integral) | 2 | real
2065577f L08- | US pop-culture/patriotic topics (Shatner x2, Troops, Memorial Day, army) | 1 | rot
7bb5f4e4 L08- | winery/hospitality descriptions; marks mostly line-lost | 1 | rot
9df08abc L08+ | metro/meditation misc (round-1 re-read: same, none) | 0 | RANDOM [KNOWN]
a2f74672 L08- | medical-legal/reproductive-bioethics topics (cerebral palsy, Roe x2, abortion, malpractice) | 2 | real
a78fabbf L08+ | scattered-distribution participles (dotted x2, sprinkled, ranging, -shaped, kaleidoscopic) | 2 | real
a987de50 L08- | divorce/bars/band misc (round-1 re-read: same, none) | 0 | RANDOM [KNOWN]
b0aadb62 L08- | commerce-contact boilerplate | 0 | rot
c6139fc6 L08+ | Slicers/plants misc (round-1 re-read: same, none) | 0 | RANDOM [KNOWN]
cca98327 L08- | capitalized product-announcement headers at BOS (PRODUCT, Project, Post) | 1 | rot
ebf8e8a1 L08+ | misc proper nouns | 0 | rot

## Layer 9
0d9cc141 L09- | mojibake + numeric URL/phone fragments — junk-text [family MOJI] | 1 | rot
0f13e95f L09+ | concrete technical nouns misc | 1 | rot
11b0b8fe L09- | investors/inventories/institutions; weak | 1 | rot
56d8fca6 L09+ | CITATION/attribution verbs (quotes x3, refer, reference(d), share, report) | 3 | real
5cf1de17 L09- | all marks line-lost; commentary-journalism contexts; unreadable | 0 | rot
63bda4c2 L09+ | perception/sound family (hear, Sound, Noise, seen) — round-1 hearing-cluster analog | 2 | real
930d262c L09+ | lexical ' also' (additive connective) | 3 | real
a5659c36 L09- | formal-institutional proper nouns | 1 | rot
a9dd0930 L09+ | body/emotion misc | 0 | rot
af0a27db L09- | HAZARD/contamination topics (choking, lice, candidiasis, pesticide residues) | 2 | real
d5d28b1b L09+ | instructional-technical steps misc | 1 | rot

## Layer 10
019c469d L10- | tech-spec misc | 0 | rot
31ab1262 L10+ | OUTLIER-A function-word carrier (round-1 re-read: same) | 0 | RANDOM [KNOWN]
550309dc L10+ | web-brand/platform proper nouns at BOS (Kindle, Costco, GOP, Wikipedia) | 1 | rot
6d1ed5ff L10+ | numeric-statistic tokens (percentages, addresses) | 1 | rot
71014ff1 L10- | flycasting/hobby jargon | 0 | rot
7f7f64e4 L10+ | misc | 0 | rot
997f32aa L10- | technical fragments | 0 | rot
ecf5ee17 L10+ | OUTLIER-A carrier (round-1 re-read: same) | 0 | RANDOM [KNOWN]
effbdc9c L10- | British-drama/sports names | 1 | rot
f945d811 L10+ | outdoors/gaming place-proper-nouns | 1 | rot

## Layer 11
26f7e092 L11- | corporate-geopolitical institutions (ConocoPhillips, Emirates, Illinois Policy) | 1 | rot
2d57dcac L11+ | franchise/brand names | 1 | rot
2dbb5ac1 L11+ | misc | 0 | rot
3bb8720a L11+ | STATE-INSTITUTION cluster (Army x3, Dept of Justice, NIH, forces, economy) | 2 | real
fcf12476 L11- | beet-root/property misc lexical | 1 | rot

## Layer 12 (heavy OUTLIER-A contamination at this depth)
0788de42 L12- | OUTLIER-C/A mixture carrier | 0 | rot
14cec027 L12+ | OUTLIER-A carrier (round-1 re-read: same) | 0 | RANDOM [KNOWN]
2af8eca1 L12+ | LEXICAL ' global' (11/12 contexts) | 3 | real
33bd20c6 L12- | OUTLIER-A carrier (new id — contamination reaches G arms) | 0 | rot
4088f73e L12+ | space-before-digit + brake/penalty misc | 1 | rot
40c46542 L12+ | pop-ups/browser boilerplate | 0 | rot
40d6f33d L12- | BOS product misc | 0 | rot
4c074d6e L12+ | OUTLIER-A carrier | 0 | rot
68594b89 L12+ | OUTLIER-A carrier (round-1 re-read: same) | 0 | RANDOM [KNOWN]
80c9f00d L12- | OUTLIER-A carrier | 0 | rot
8985d859 L12+ | OUTLIER-A carrier | 0 | real
94367e9b L12+ | HEDGING adverbs (really x5, actually x2, almost x2, seeming(ly)) | 3 | real
9610beb7 L12- | OUTLIER-A + delivery boilerplate | 0 | rot
9d8b0cae L12+ | OUTLIER-A carrier (round-1 re-read: same) | 0 | RANDOM [KNOWN]
b2b12a49 L12+ | OUTLIER-A carrier | 0 | rot
bbe51b25 L12+ | misc | 0 | rot
bed2ab98 L12- | manual/bodily-handling contexts (bare hands, knees, saucepan) | 1 | rot
bffb0cb2 L12- | OUTLIER-A carrier (round-1 re-read: same) | 0 | RANDOM [KNOWN]
c0808f0a L12- | marks line-lost; unreadable | 0 | rot
c3444142 L12- | OUTLIER-A carrier | 0 | rot
c946034e L12+ | capitalized misc (Band, Dean, CI) | 0 | rot
d6c0806e L12- | OUTLIER-A carrier | 0 | rot
d7dc073d L12+ | legal-contractual verbs (denied allegations x2, constitute a waiver, forfeited, guarantee) | 2 | real
d90aef3d L12+ | casual downtoners (little x4, kinda x2, slightly x2) — round-1 'little' family + informal register | 3 | real
e8ac638e L12- | OUTLIER-A-adjacent carrier | 0 | rot
f905c340 L12+ | doc-initial bare punctuation/metadata delimiters (., ,, /, ://) | 1 | rot
fbe8c3a4 L12+ | TRANSPORT cluster (transportation x5, vehicle) [family T] | 3 | real
fc164ba0 L12- | OUTLIER-A carrier (round-1 re-read: same) | 0 | RANDOM [KNOWN]
feef0d1a L12- | One-Belt-One-Road + spam-policy misc | 1 | rot

## Layer 13
04a96d04 L13- | One Belt One Road + holistic-philosophy repeats | 1 | rot
304c7e6a L13- | institutional-event nouns (university x4, wedding x2) | 1 | rot
5a06ee47 L13+ | numeric-statistic tokens (percent, prices, addresses, wavelengths) | 2 | real
7bbf2a72 L13+ | product-spec misc | 0 | rot

## Layer 14
027139b2 L14+ | OUTLIER-A carrier (round-1 re-read: same) | 0 | RANDOM [KNOWN]
42d17e94 L14- | NUMERIC IDENTIFIER CODES (DOI digits, CI values, SKU/postal codes) | 2 | real
7bbfbf34 L14+ | OUTLIER-A carrier (round-1 re-read: same) | 0 | RANDOM [KNOWN]
a6b9f81b L14+ | dramatic-causation verbs (triggering/triggered, catalyze, skyrocketed, heralded) | 3 | real

## Layer 15
28722a34 L15- | REDUCTION verbs (minimizing x2, minimize x2, limiting, simplify, scaling) | 3 | real
66763cc2 L15- | emphatic adverbs (truly x2, quite) + spam-context repeats | 1 | rot
b291df35 L15+ | INFORMAL TYPOS/nonstandard spellings (theres, thats, lastest, buildling, verison, litterally, stripey) | 3 | real
c6377d12 L15+ | BOS headers misc | 1 | rot
d16c7977 L15+ | fragment misc | 0 | rot
e32f1149 L15+ | TRANSPORT cluster (same contexts as fbe8c3a4) [family T] | 3 | real

## Layers 16-18
15f7433d L16+ | TRANSPORT cluster, 4th appearance [family T] | 2 | real
32b995db L16+ | sports teams/athletes (Povetkin, Gators, Spurs, Saints, Pistorius) | 2 | real
5abf2cda L16+ | LEXICAL ' when' (temporal subordinator, 10/12) | 3 | real
3360bb4e L17+ | misc adverbs | 1 | rot
523983b8 L17- | AMPERSAND ⟦ &⟧ again (family with 4c9c7757 L02) | 2 | real
80377515 L17- | BOS function words; position-flavored carrier | 0 | rot
910a2c8d L17+ | MOJIBAKE/encoding-garbage + URL query fragments [family MOJI] | 3 | real
9d339cb9 L17+ | marks line-lost; unreadable | 0 | rot
351d897d L18+ | franchise/list misc | 1 | rot
3adc5b1d L18+ | doc-initial open-parenthesis ⟦ (⟧ (round-1 f54a4032 family) | 2 | real
54381361 L18+ | curly-apostrophe contractions ('t x4, 's) [family APOS] | 2 | real

## Layers 19-22
870383c7 L19+ | newline/paragraph-boundary token (round-1 re-read: same) | 2 | RANDOM [KNOWN]
9eb74c67 L19+ | curly-apostrophe possessives ('s x4, ') [family APOS] | 2 | real
fc965257 L19+ | 'pag/Pag' subword fragments | 1 | rot
12e4c4f6 L20+ | BOS fragments; position artifact | 0 | rot
b023876a L20- | mojibake apostrophes + OCR archaic quotes [family MOJI] | 2 | real
e0009e41 L20- | article ' a' lexical-carrier | 1 | rot
fbad34e9 L20- | capitalized consonant-cluster word-starts (Tw, Ph, Bl) | 1 | rot
1eecfb85 L21- | fragment misc | 0 | rot
a1bc3d58 L21+ | 'fr'-cluster word-start fragments | 1 | rot
b2429a1a L21+ | 'dec-' prefix fragments (decoupaged, decedent, Decaf, decamped) | 2 | real
6cbd93f3 L22+ | ZERO main-polarity contexts extracted — data glitch, no hypothesis | 0 | rot
d78540ce L22+ | numeric/craft misc | 0 | rot
ec9824b8 L22- | space-before-digit in dates/citations (round-1 OUTLIER-B numeric-formatting family) | 3 | real

# Summary counts: conf 3 = 20, conf 2 = 30, conf 1 = 45, conf 0 = 62 (incl. one zero-context glitch).
# 23 of 157 are KNOWN-RANDOM by ID collision; of the remaining 134, I guessed real for 47, rot for 87.
#
# Pre-registered checks for unblinding:
# 1. Test-retest: my 23 KNOWN-RANDOM re-reads should match round-1 hypotheses (they do by my
#    reading; verify none flipped in the written record).
# 2. G-real vs G-rotated blind interpretability: after round 1's dilution lesson I predict only a
#    SMALL edge for real at conf>=2 and a clearer edge at conf 3; if rates are equal, that
#    replicates the mimicry result on an independent bank.
# 3. Outlier clusters (OUTLIER-C early, A mid incl. ~10 new ids at L12, B late) should appear in
#    BOTH real and rotated arms — contamination is basis-agnostic. If confirmed, the winsorization
#    fix to the scanner is mandatory before any further rounds.
# 4. The typographic/quality tier (families APOS, MOJI, plus b291df35 typos, dde9c4e6/81ac3bcb
#    quote variants, b77cd46d unicode hyphens, 4c9c7757/523983b8 ampersand): predict predominantly
#    REAL G-axes — encoding/orthography-quality is exactly "read hard, context-varying" content.
# 5. Family T (transport, L03/L12/L15/L16): if >=3 of the four are real G-axes, check their
#    eigenvalue ranks at unblinding for cross-layer rank persistence (the SV39/SV50 signature).
# 6. Neither 'may'-modal nor 'most'-superlative surfaced in this set; SV39's G-side status comes
#    from bank_gatedness.csv, not from here.
