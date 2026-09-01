# Blinded hypotheses — committed before unblinding
# Written from review.md only. Do not edit after candidates.csv arrives.
#
# Format: blind_id  layer/polarity | hypothesis | confidence | condition guess
# Confidence: 3 = would bet on a predictive test; 2 = probable; 1 = tentative; 0 = no hypothesis (noise/artifact/carrier)
# Condition guesses are explicitly low-stakes.
# "NAMECONT" = continuation-subword-of-proper-noun/rare-word family. "OUTLIER-SET-A" = the shared
# extreme contexts recurring across many mid-layer candidates ("served the army", "plethora of ideas",
# "kittens of various ages", "rapprochement", "Beasley-Murray", "courseworks that rob you").
# "OUTLIER-SET-B" = the shared late-layer spec-sheet docs ("5s / 5c / 5 / SE iPad", phone numbers,
# "81.5 dB(A) ... 2.5 litres", "pages 427-446", trinidadexpress URL).

## Layer 0
1c30c6a7 L00+ | lexical pair 'survey/search' (inquiry nouns, case-insensitive) | 2 | svd
1e5c538b L00- | abstract nominalizations (-ance/-ment/-ity/-ice: injustice, enhancements, insecurity, efficiencies) | 2 | svd
25824990 L00- | lexical ' extensive' (+ stray NAMECONT) | 2 | svd
3134adb7 L00- | weak: plural common nouns (colors, bugs, charges, costs) + NAMECONT 'adt' | 1 | rot
3d41e05f L00+ | long Latinate abstract nouns (culmination, contemplation, plethora, determination) | 2 | svd
50eb9368 L00+ | lexical ' efficiency' (with -tion/-ity neighbors) | 3 | svd
5a3d9c00 L00- | NAMECONT: surname/rare-word continuation subwords (Carrey, Kunze, Grainger, Borden) | 3 | svd
78c704c0 L00- | closing-quote-after-period token ⟦."⟧; opposite pole is NAMECONT | 3 | svd
7c195116 L00+ | curly open single-quote ⟦ '⟧ | 3 | svd
96dc8fde L00+ | lexical 'regulatory/advisory' (-ory governance adjectives) | 2 | svd
b973ccff L00- | NAMECONT: place/technical-word continuation (Bozeman, Buryatia, Hyannis, Brachiosaurus) | 3 | svd

## Layer 1
20939736 L01+ | truncation/ellipsis markers (⟦...⟧, ⟦[…]⟧, ⟦."⟧) — web-text cutoff boilerplate | 2 | svd
2d8f268a L01+ | weak: capitalized header/section nouns (Tricks, Concepts, conventions) + NAMECONT | 1 | rot
4a03be23 L01- | lexical ' structure' (all senses) | 3 | svd
5401f830 L01+ | dispute/response nouns-verbs (backlash, argue, argued, advice) | 1 | rand
5a1d5b74 L01+ | generic entity nouns 'person/thing' | 2 | svd
6c7fca82 L01- | formal-erudite register (requisite, jurisprudence, purview, oeuvre, complaisance) — multi-lemma register feature | 2 | rand
997e1aaa L01+ | word-continuation fragments (mixed proper/technical: Keech, metathesis, Chordify) | 2 | rot
a3f9e338 L01- | corporate/tech buzzword register (leveraging, impactful, business-focused, tracking) | 1 | rand
b14f6891 L01- | weak: list/container nouns (boxes, list, Listings) | 1 | rot
d1d6779c L01- | long prefixed/compound words (interconnected, disadvantaged, discrepancies, countertops) | 1 | rot
e12aee82 L01+ | NAMECONT: proper-noun continuation (Seberg, Theresienstadt, Grinnell, Borden, Seeger) | 3 | svd
e84758f0 L01- | lexical ' significant' | 3 | svd
f7499ce0 L01+ | lexical 'advocacy/Advisory' (civil-society -acy/-ory nouns) | 2 | svd

## Layer 2
06c2f7af L02- | NAMECONT/rare-word continuation (Wu-Tang, Gargantuan, irresistible, chaetigers, renegade) | 2 | rot
0716fb17 L02- | academic-analytical register (methodology, geopolitical, quantitative, normative, socioeconomic) | 3 | rand
36a8c08d L02+ | alphanumeric code/acronym continuation fragments (NBXR463, YOHANN, GGNIMT) | 2 | svd
3a0d1f68 L02+ | stems 'showcase/signif-' (showcased, showcases, signifies, significance) | 2 | svd
7c2517a0 L02- | stem 'consisten-' (consistent, consistency) | 3 | svd
9875f821 L02- | quotation-onset ⟦"It⟧ after paragraph break (testimonial-quote opening) | 2 | svd
b7f17d5f L02+ | legal/numeric citation fragments: '501(c' parenthetical, date digits | 1 | rot
c33ba7cd L02- | weak: URL '.com' + misc | 1 | rot
c42c6103 L02- | BRITISH ORTHOGRAPHY: -ise/-isation/-our spellings (industrialising, rigours, jeopardise, vigour, Randomised) — distributed orthographic-dialect feature | 3 | rand
c71470e8 L02+ | word-continuation fragments (Opteron, Pinchot, Chrysostom, Tintin) — NAMECONT | 2 | rot
d89d0993 L02- | generic abstract plurals 'experiences/things/enterprises' | 1 | rot
d8ecfdd4 L02- | NAMECONT (Samuelson, Neuenhagen, Grossman, Schilling) + misc | 1 | rot
f54a4032 L02- | open-parenthesis token ⟦ (⟧ (parenthetical insertions) | 3 | svd

## Layers 3–4
8529df6f L03+ | lexical ' something' | 3 | svd
8cc5c583 L03+ | web-truncation boilerplate (⟦...⟧ line cutoffs, [email protected] artifact) | 2 | rot
e2125abd L03+ | lexical ' some' (often sentence/doc-initial) | 3 | svd
7083c960 L04- | lexical ' little' (+ much, sudden: small-quantity/suddenness colloquial) | 2 | svd
946075c0 L04+ | academic domain adjectives + 'knowledge' (biological, religious, political) | 2 | rand

## Layers 5–6
011da03c L05- | stem 'engag-' (engaged, engaging) + -ing growth verbs | 2 | svd
c0a5a8ed L05- | stem 'authentic-' (authentic, authenticate) | 2 | svd
a4578a46 L06- | NAMECONT: place-name continuation (Thanet, Skaneateles, Menomonie) | 2 | rot
acabe9dc L06- | POSITION ARTIFACT: first content token after BOS (marks all at position ~1) | 0 | rot
f5ad3e55 L06- | lexical ' new' (headline-flavored) | 3 | svd

## Layer 7
2442acd1 L07+ | medical/Greek-Latinate technical fragments (paramedic, pharmaceutical, pneumatic, protean, antibiotic); opposite pole = 'ensur-' stem | 2 | rand
4ac674dc L07+ | hedging degree adverb ' pretty' (+ relatively, internationally) | 2 | svd
69e26006 L07+ | weak: franchise/title fragments (Civil War ×2) | 1 | rot
74216a7d L07+ | weak: named technical objects (Titan X, silicon, hub, hull) | 1 | rot
75b7d9f6 L07- | weak: ' for' + misc | 0 | rot
c47bb5c8 L07+ | NAMECONT + 'arch' (Allende, Klamath, Warlock, Akiyoshi, Lucerne) | 1 | rot
caeffd7e L07+ | NAMECONT: place-name continuation (Kißlegg, Chessie, Chehalis, Hamrun, Nothofagus) | 2 | rot
d84a95ba L07+ | numeric/date continuation fragments (40,⟦000⟧km, Q3/Q⟦4⟧, Fall/W⟦inter⟧, 040⟦0⟧ GMT) | 2 | svd
f64e36e3 L07- | abstract drama/upheaval nouns (chaos, crisis, Cosmic, Journey, World); OPPOSITE POLE = ' may' ×3 | 2 | rand

## Layer 8
19c4b614 L08+ | lexical ' all' (universal quantifier) | 3 | svd
624956bc L08- | INDEFINITE family: some/any/somehow/something (multi-lemma indefiniteness) | 2 | svd
74675256 L08+ | intensity modifiers (slightly, intensified, intense, rapidly, highly) — degree-modifier class; some BOS flavor | 2 | rand
9df08abc L08+ | no coherent hypothesis (metro, meditation, Park, defence) | 0 | rot
a987de50 L08- | no coherent hypothesis (divorce, bars, band, Vampire, dental) | 0 | rot
aa8b2143 L08+ | crisis/chaos/journey cluster — same feature & contexts as f64e36e3; opposite pole again ' may' | 2 | rand
aad853ce L08+ | stem 'liv-' (living, live) + misc; weak | 1 | rot
c2e8429d L08+ | approximation adverbs 'almost' (+ equally, more) | 2 | svd
c6139fc6 L08+ | no coherent hypothesis (Slicers, plants, built) | 0 | rot
dc699448 L08+ | TEXT-ERROR DETECTOR: duplicated/disfluent words ("copy copy", "get be getting", "still is still", "in of itself") | 3 | rand
e27e3cd9 L08+ | compound-modifier prefixes (online, outdoor, oral, eco-, neuro-); opposite pole = 'signify' stem | 1 | rot

## Layer 9
2ac6d564 L09- | carrier/function words, BOS-heavy | 0 | rot
3e386a67 L09+ | -ally/-ky adjectival-adverbial suffix class (legally, intellectually, electrically, icky) | 1 | rot
82d2b97e L09+ | lexical ' little' (same contexts as 7083c960 — cross-layer duplicate family) | 2 | svd
da7374ba L09- | formal-erudite/French-derived register (complaisance, rapprochement, triumphal, uncloistered) — same family as 6c7fca82 | 2 | rand
fc627aac L09- | minimizer adverbs 'just/simply' | 3 | svd

## Layer 10
051f90f1 L10- | affect-laden informal evaluatives (icky, glorious, dirty, insanity); overlaps 3e386a67 contexts | 1 | rand
2a767913 L10- | stem 'approach-' (all inflections) | 3 | svd
31ab1262 L10+ | carrier: articles/function words on OUTLIER-SET-A | 0 | rot
867d1a09 L10- | carrier: function words on OUTLIER-SET-A | 0 | rot
989d5e45 L10- | junk mix (mailing, £, /lib) | 0 | rot
ea5665b3 L10+ | Indoor/Outdoor/Environment capitalized setting words (product-listing flavor) | 1 | rot
ecf5ee17 L10+ | carrier: function words on OUTLIER-SET-A | 0 | rot
f5d17605 L10+ | weak: nonstandard spellings + intensity (syncronization, aggresive, enormous, sensational) | 1 | rand

## Layer 11
2c9e56d1 L11+ | HEARING family: listen/hear/listening (multi-lemma perception verbs) | 3 | rand
34bd972c L11- | carrier: function words, BOS-heavy, OUTLIER-SET-A | 0 | rot
6bc90eec L11+ | CRISIS/CLIMATE cluster (crisis ×4, chaos, Climate, Ecology, trauma) — third appearance of the crisis feature | 3 | rand
ac23f359 L11- | clinical/engineering technical vocabulary (Clinical, torque, engineer, function) | 1 | rand
c6ebda9c L11+ | process/system nouns (planning, management, programming, mechanisms) | 1 | rand
f1bbab6e L11- | minimizer 'simply' (+ particularly) — same family as fc627aac | 3 | svd

## Layer 12
14cec027 L12+ | carrier: articles on OUTLIER-SET-A | 0 | rot
199b9c3c L12- | academic-analytical register, CROSS-LINGUAL (theoretical, hypothetical, эффектив-, Potential, Chemical) | 2 | rand
2f16ecd8 L12+ | BOS-adjacent word fragments (EXCALIBUR, Tuscaloosa, Seanan) — position + NAMECONT mix | 1 | rot
68594b89 L12+ | carrier on OUTLIER-SET-A | 0 | rot
9321390d L12- | carrier, BOS-heavy, OUTLIER-SET-A | 0 | rot
9d8b0cae L12+ | carrier on OUTLIER-SET-A | 0 | rot
bffb0cb2 L12- | carrier on OUTLIER-SET-A | 0 | rot
c4394437 L12+ | HEARING family again (Listen/hear/Heart) — same contexts as 2c9e56d1; cross-layer duplicate | 3 | rand
fc164ba0 L12- | carrier on OUTLIER-SET-A | 0 | rot

## Layer 13
247b3c62 L13- | infinitival 'to' in formal-obligation contexts (failure to enforce, right to subsequently enforce); opposite pole 'gaming' | 1 | rot
9f48c30a L13+ | carrier on OUTLIER-SET-A (incl. spam contexts) | 0 | rot
ac71697c L13- | weak: 'incredible' + marketing superlatives | 1 | rot

## Layer 14
019f7650 L14- | price/currency '$' in commerce boilerplate | 2 | svd
027139b2 L14+ | carrier on OUTLIER-SET-A | 0 | rot
596308e7 L14- | BOS-adjacent fragments + web metadata | 0 | rot
735bf43c L14- | POSITION ARTIFACT: first-content-token after BOS | 0 | rot
7bbfbf34 L14+ | carrier on OUTLIER-SET-A | 0 | rot
ba158d54 L14+ | SPUN-SPAM/GIBBERISH DETECTOR ("rob you of important many hours", "releasing weightechnology", word-salad SEO text) | 3 | rand
e1c1e752 L14+ | lifestyle-marketing register at BOS (Luxurious, Beautiful, stylish, entertaining) — position-mixed | 1 | rand
e72d8f20 L14- | growth/improvement verbs (grow, growing, improving, improved) | 2 | rand

## Layer 15
686107f7 L15+ | hedging/concessive discourse (seeming, though, perhaps) — BOS-mixed, weak | 1 | rot
c0fd4077 L15+ | coordinator 'and' + spam contexts; weak | 0 | rot
c87b3eb7 L15+ | BOS-adjacent incoherent (effect, Tractor, Term) | 0 | rot
f1a66bc9 L15- | stem 'use/used' | 3 | svd
fa132cbb L15- | 'little' family + BOS mix (weaker duplicate); opposite pole 'exceed-' stem | 1 | rot
ff4ed7fe L15+ | POSITION ARTIFACT + web-junk fragments at BOS | 0 | rot

## Layers 16–18
3f28aaaf L16- | futurity/enablement auxiliaries (will ×5, allow, allowing, are, have) | 2 | rand
801522e3 L16+ | positive-evaluation adjectives (perfect, possible, pretty, complete) — weak p-flavor | 1 | rand
5ec34ae6 L17+ | coordinator ' and' | 2 | svd
fa48df1c L17+ | complementizers (that, if, how, consider) on spam contexts — mixed | 1 | rot
b18e0f3d L18- | forum/datestamp boilerplate at BOS (07:40, 05/18/2014, em-dash) | 1 | rot
c16c18de L18- | coordinators or/and in list constructions | 1 | rot

## Layers 19–22 (dominated by OUTLIER-SET-B spec-sheet docs)
089cacfb L19- | open-parenthesis/quote in metadata listings (BOS-heavy) — same lexical class as f54a4032 | 2 | svd
5f7d90f9 L19- | spam/gibberish-adjacent + coordinators (sneer dissembles, noiseware keygen) | 1 | rand
8219baca L19- | NON-ENGLISH (Romance-language) text: Spanish/Italian function words (a la adopción, ai piedi, en mayor escala) | 3 | rand
870383c7 L19+ | newline/paragraph-boundary token ⟦.\n⟧ | 2 | svd
8ff51116 L19- | POSITION ARTIFACT: BOS-adjacent name/URL-slug fragments | 0 | rot
02cb535c L20- | BOS product-listing/metadata boilerplate (OUTLIER-SET-B) | 0 | rot
e6f80676 L20- | POSITION ARTIFACT: BOS fragments (OUTLIER-SET-B overlap) | 0 | rot
f64b6759 L20- | numeric spec-sheet formatting: space-before-digit, phone hyphens (OUTLIER-SET-B) | 1 | rot
24484c53 L21- | numeric measurement listings: space-before-digit token in stats/specs ( ⟦ ⟧348, ⟦ ⟧2.5 litres) | 2 | svd
7afe987d L21- | same numeric-measurement family, shared contexts with 24484c53 | 2 | rot
9ca29807 L21+ | OUTLIER-SET-B boilerplate at BOS | 0 | rot
05337639 L22+ | structured-data numerics (percent lists, phone, model codes) — OUTLIER-SET-B | 1 | rot
35996919 L22- | OUTLIER-SET-B numerics (only 7 contexts) | 1 | rot
3dede074 L22+ | structured-data DELIMITERS: '/', '-', '(', '=' in dates, coordinates, phone numbers, URLs | 2 | svd
57dff45a L22+ | same delimiter/measurement family as 3dede074 | 2 | rot
92fe8570 L22+ | newline/segment-boundary token ⟦.\n⟧ (product lines, paragraph ends) — same class as 870383c7 | 2 | svd

# Summary counts (my scoring): confidence 3 = 22, confidence 2 = 41, confidence 1 = 32, confidence 0 = 39.
# Pre-registered checks for unblinding:
# 1. Interpretability rate by condition (conf>=2 fraction): predict svd > random > rotated.
# 2. The ~15 OUTLIER-SET-A carrier candidates and the BOS-artifact candidates: predict mostly rotated.
# 3. The register/semantic/anomaly cluster (crisis x3, hearing x2, erudite x2, academic x2, spam, typo,
#    British spelling, Romance-language): predict mostly random.
# 4. Sharp single-token/stem detectors: predict mostly svd, concentrated at low sv_rank in early layers.
# 5. Where did the user's 'may' and 'most' directions land? 'may' appeared only as the OPPOSITE pole of
#    the crisis-cluster candidates (f64e36e3, aa8b2143) — check whether their original may-direction is
#    one of these axes sign-flipped, or below bar entirely.
