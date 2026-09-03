# A semantic reading of the top contexts: `unrealized_words_selectivity` vs `g_scan_sparse`

Second pass, companion to `top_contexts_uws_vs_gss.md`. That file characterised the two banks
structurally (UWS: between-word prediction slots; GSS: unfinished-word onsets). This one sets the
fragments and punctuation aside and reads the tails that consist of real words and phrases, both
poles of each direction, for what they are *about*: register, voice, stance, genre, and the kind of
thing being referred to. It is an interpretive essay with quoted evidence, not a measurement; the
few numbers are there to keep the reading honest. UWS = `unrealized_words_selectivity`, GSS =
`g_scan_sparse`; ⟦ ⟧ marks the centre token; ⏎ is a newline; word lists are the centre tokens of a
tail with their counts out of 64.

What was read: for each bank, the 50 most selective SVs whose selected tail is at least 60% whole
words, the 15 broadest SVs, and 15 mid-selectivity word-tails drawn at random - both polarities each,
about 160 tails per bank - plus a lexicon probe over every tail in both banks and a look at which
documents each bank's selective SVs are drawn to.

## 1. The claim in one paragraph

Both banks find the great register axis of English - Latinate officialdom against the vernacular -
and they find it many times over. Past that shared backbone they carve meaning along different
grains. UWS's word-level directions are organised around **stance**: how the utterance stands toward
what it says - permission and obligation (`may`, `shall`), wonder (`fantastic`, `surreal`),
grandeur (`crisis`, `chaos`, `world`, `journey`, `cosmic`), appraisal and its deflation (`lovely`;
`just`, `simply`, `pretty`), belief and hope (`hear`, `trust`, `believe`; `hope`, `doubt`, `await`),
assertion (`say`, `signifies`, `certainly`), editorial framing (`It is surprising that`, `This could
be interpreted as`), the marketing second person (`you've got`, `Click here`) - and, tellingly, they
are drawn to text that has no stance at all because it has no meaning: machine-spun word salad.
GSS's word-level directions are organised around **reference**: what kind of thing is being talked
about - the elemental and bodily (`water`, `sun`, `fish`, `ice`, `rain`, `smoke`; `mouth`,
`stomach`, `tongue`; `Ground`, `Head`, `Born`, `Body`), the everyday object (`dogs`, `cars`, `cups`,
`cheese`, `chicken`, `salmon`, `box`, `game`), the apparatus of the world (`transport`, `station`,
`port`, `satellite`; `Army`, `NIH`, `Justice`, `University`; `law`, `requirements`; `death`,
`problems`, `prison`, `cure`) - set against the institutional abstraction that names it from above
(`strategic`, `innovative`, `integrated`; `Project`, `Product`, `Official`; `Service`, `Relations`,
`Leadership`; `matters`, `affairs`, `aspects`, `things`). Put crudely: UWS's directions read like
*tones of voice*; GSS's read like *nouns*. The broad, always-on directions say the same thing in
miniature: UWS's broad axis is a vocabulary of *assessment* (`potential`, `alleged`, `hypothetical`,
`preliminary`, `optimal`), GSS's a vocabulary of *sectors* (`legal`, `clinical`, `market`, `tax`,
`health`).

## 2. The shared backbone: officialdom against the vernacular

Every layer of both banks contains at least one direction whose two poles are the learned Latinate
register and ordinary speech. The lexicon probe finds 33 UWS tails and 24 GSS tails carrying 12+
words from a Latinate-official list, and 41 vs 43 carrying 12+ stance/hedge adverbs. Examples:

UWS
* `L01_SV16`: `think`:33 `around`:9 `says`:6 `Maybe`:5 `really`:5 `like` `big` `sure`
  ("I'm a pretty big fan of warm salads", "the same girl you ran around the grass with naked")
  against `judicial` `unparalleled` `requisite` `discourse` `efficacy` `arduous` `profound`
  `plethora` `adherence` `jurisprudence` `oeuvre` `maturation`.
* `L02_SV18`: `got`:22 `out`:13 `up`:7 `!”` `around` `'em` `we're` ("If you've got a sound casting
  stroke", "You've got a huge pick, from fur-lined moccasins") against `significant` `biological`
  `normative` `methodology` `quantitative` `comprehensive` `confidentiality` `cognitive` `geopolitical`.
* `L13_SV11`: `expertise` `research` `evidence` `study` `literature` `policy` `jurisprudence` against
  `can't` `gonna` `wanna` `gotta` `soooo` `friggin` ("That's all you gotta do", "Wanna-bes stay home").
* `L00_SV14`: `clinical` `legislative` `novel` `normative` `requisite` `advocacy` `statutory`
  `substantive` against `guy`:33 `biggest`:17 `Maybe`:8 ("The main bad guy happens to be a look alike
  major", "Maybe, for his birthday, I'll get him a raincoat").

GSS
* `L01_SV29`: `dogs`:8 `cars`:6 `mouth`:5 `City` `cups` `room` `girls` `toilet` `stomach` `tongue`
  `cheese` ("hot dogs", "your mouth feeling fresh", "the mouth, pharynx (throat), esophagus, stomach")
  against `innovative`:9 `relevant` `specialised` `integrated` `integration` `subsequently` `enriching`
  `adversely` `personalised` `sophisticated` `industrialising` `privatised`.
* `L18_SV09`: `send`:10 `water`:10 `buy` `money` `do` `bought` `cook` `walk` `put` `wagon` `sugar` `work`
  against `critical`:7 `Critical` `particularly` `Prior` `Typical` `exceptional` `Specifically`
  `significant` `preliminary` `Alternative` `Essential`.
* `L06_SV16`: `game`:11 `dog`:7 `word` `movie` `file` `card` `girl` `video` `job` `song` against
  `pervasive` `pertaining` `adversely` `sustaining` `unparalleled` `prevalent` `prosperous` `favorable`.
* `L03_SV36`: `Information`:23 `information`:15 `instruction` `provision` `Evidence` `Definition`
  `Document` against `past` `TN` `lol` `you're` `i'm` `we're` `hottest` `crazy` `weekend` `:)`.
* `L15_SV10` / `L16_SV10` / `L09_SV11` / `L12_SV11` / `L14_SV11` (one direction persisting across
  L9-L16): `transport` `transportation` `resource` `station` `vehicle` `space` `port` `stock`
  `instrument` `cell` `Plant` `satellite` `atom` `material` against `Why` `Yeah` `?!` `lol` `Yep`
  `huh` `Nah` `uh` `ahaha` `anymore` `ever` `again` `yet` ("Yep, they call it 'Johns.'", "Useless? Yep.
  Accurate? Nah.", "How about that eclipse, folks, huh?").

The axis is the same; the way the two banks reach the vernacular is not. UWS arrives through the
*speaking subject* - verbs and adverbs of opinion and hedging (`think`, `says`, `Maybe`, `really`,
`sure`, `got`, `gonna`). GSS arrives either through the *phatic noise of talk* (`Yeah`, `huh`, `Nah`,
`Yep`, `lol`, `?!`) or, more often, through the *nouns of the everyday world* (`dogs`, `cars`,
`mouth`, `cheese`, `chicken`, `box`, `game`). That difference in how the low pole is populated is the
first sign of the larger split.

## 3. UWS: directions of voice and attitude

**Permission and obligation.** `L07_SV39` is the word `may` in all 64 events, in the register of
terms-of-service: "You⟦ may⟧ use Our Site for legal purposes only", "Additional charges⟦ may⟧ be
applied to gumpaste flowers, bows, figures", "Parking (charges⟦ may⟧ apply)", "technologists⟦ may⟧
not require continuous access". Its other pole is `world`:13 `complex`:10 `chaos` `crisis` `journey`
`Cosmic` `comprehend` `ocean` `Odyssey` `ecstatic` `profound` - "The Secret⟦ World⟧ of Pharmaceutical
Trial Subjects", "the information management⟦ chaos⟧". `L08_SV39` is the same contrast reversed
(`complex` `crisis` `chaos` `curious` `globe` `Journey` `Cosmic` `Crunch` `Legacy` `Odyssey` against
`may`:34 `payment`:13 `might` `paying` `payments` `need` `maybe` `want`). `L07_SV07` sets the archaic
legal-biblical voice - `shall` `whilst` `forth` `furnished` `upon` `sought` `bade` `scarcely` `duly`
("who dw⟦elt⟧ in Sodom", "God might bring to⟦ pass⟧ in Montreal", "and⟦ bade⟧ adieu early") - against
the internet brand (`.com` `Network` `Wallet` `chat` `/maps`: "Cartoon⟦ Network⟧", "Nerd⟦Wallet⟧",
"Snap⟦chat⟧", "goo.gl⟦/maps⟧"). The probe finds 17 UWS tails with 12+ legal-modal words vs 10 in GSS.

**Grandeur.** A single direction persists as `SV30` from L10 to L13 and as the negative pole of
`L10_SV31`: `crisis`:10 `chaos`:5 `climate` `conflict` `global` `paradigm` `trauma` `culture`
`ecology` `chronic` `clarity` `community` `harmony`; `world`:25 `-Man` `glamour` `fascination`
`mesmer`. Its counter-pole is narrative possession and taking - `have` `had` `were` `would` `took`
`take` `taken` `taking` `got` ("And they⟦ took⟧ Lot, Abram's brother's son, who dwelt in Sodom", "I
have⟦ got⟧ to make a considered decision", "The trains were then⟦ taken⟧ forward") - or the bureaucratic
`equal` `order` `per` `priorities` `≤` `guidelines` `assigned`. The probe finds 12 UWS tails with 12+
such grand-systemic words; GSS has one.

**Wonder.** `SV29` at L11-L13: `fantastic`:21 `spectacular`:8 `imagine` `experiment` `fascinated`
`fascinating` `surreal` `marvellous` `fanciful` `monster` `comet` `crystal` - "His creations are open
doors to⟦ fantastic⟧ and dreamy horizons", "the fanc⟦iful⟧ imagery of surrealist photomontages",
"give comets their⟦ spectacular⟧ tails". Its other pole is the price tag: `$`:29 `@` `|` `grade`
`priorities` `Duty` `payday` - "Regular price⟦ $⟧22.70", "Members free, non members⟦ $⟧20, at the
door $30 (cash only)", "90 Minute Session -⟦ $⟧110". The marvellous against the transactional. The
probe finds 7 UWS tails with 12+ imagination words; GSS has none.

**Eloquence.** At L0, `L00_SV22` is grandiloquence itself: `determination`:6 `comprehend`
`culmination` `captivating` `unparalleled` `facilitate` `imperative` `plethora` `affirmation`
`fascination` `profound` `deterioration` `consideration` `contemplation` - "make it⟦ imperative⟧ that
we are able to provide support", "It was the⟦ culmination⟧ of a journey", "the order of reading and
⟦ contemplation⟧". `L00_SV28` is its administrative cousin: `infrastructure` `sustainability`
`regulatory` `engineering` `administrative` `advisory` `emergency` `rationale` `authorization`
`discretionary` `mitigation` ("statutory and⟦ regulatory⟧ funding conditions", "compliance and
⟦ regulatory⟧ risk functions").

**Appraisal and its deflation.** `L09_SV58`: `just`:37 `simply`:9 `whatever` `reportedly` `certainly`
`definitely` - "was⟦ simply⟧ a case of bad diarreah", "her dancing⟦ just⟧ made me feel so unfit", "was
⟦ just⟧ straight up disrespectful" - against technical staging (`order` `high` `phase` `lane` `layer`
`clock` `failure` `Channel`: "In a second⟦ phase⟧", "the CPU is over⟦clock⟧ed"). `L07_SV58`: the
hedged mild (`pretty`:32 `relatively` `approx` `otherwise` `comparatively`: "it is actually⟦ pretty⟧
good", "Glad it turned out⟦ pretty⟧ well!") against killing and kingship (`killed` `kill` `death`
`kills` `killing` `King` `Battle`: "The iPad has already⟦ killed⟧ netbooks", "jurors should go home
tonight and⟦ kill⟧ themselves", "Jim will always be the⟦ king⟧"). `L11_SV60`: institutional-educational
nouns against colloquial measure (`whopping` `tsp` `quite` `roughly` `nifty`: "document this nifty
⟦ little⟧ thing I did today", "Bottled at a⟦ whopping⟧ 75%").

**Belief, hope, perception, assertion.** `L12_SV50`: `heard`:10 `seen` `hear` `Listen` `waiting`
`hearing` `listen` `felt` `see` `trust` `believe` `warning` `read` `saw` - "I keep⟦ hearing⟧ about how
kids who drink mostly bottled water", "Don't take my word for it,⟦ listen⟧ to Celente". `L12_SV64`
(negative pole): `hope`:13 `before` `would` `maybe` `doubt` `expected` `believe` `hopeful` `wait`
`await` `hoping` `perhaps` `sure` `trusted` - "about to⟦ doubt⟧ this, but it seems pretty true", "stay
home and⟦ await⟧ the knives" - against `Commerce` `Carnegie` `telecommunications` `civil` `acres`
`capital`. `L08_SV54` (negative pole): `say`:17 `'s` `certainly` `shall` `signifies` `reaffirm`
`signify` `asserted` `steadfast` `said` `undoubtedly` - "I'm proud to⟦ say⟧", "The change in our
brand identity⟦ signifies⟧ our intent", "has⟦ certainly⟧ ruffled feathers" - against the domain prefix
(`online` `outdoor` `neuro` `oral` `indoor` `eco` `optical`).

**The editorial opener and the blogger's deixis.** `L21_SV21`: `It`:25 `This` `which` `What` - "(1969)⏎
⟦It⟧ was originally titled Mo Getta Mo", "then I'm a terrorist.'"⟦ This⟧ could be interpreted as
Hirschberg trying to frame", "⟦It⟧ is surprising that Snow still has this job", "⟦It⟧ would be
impossible to discuss Sanction without", "⟦It⟧'s funny because it's true" - against the venue notice
(`at` `home` `elegance` `town`: "a pre-theatre party from 3 to 6:30 PM⟦ at⟧ the Times Square Madame
Tussaud's", "Job Opening⟦ at⟧ T & R Creative Pub"). `L21_SV54`: `This`:26 `this`:12 `your` `'s` `my`
`also` - "⟦ this⟧ week's podcast guest", "⟦ this⟧ phone almost eliminates my thirst for a tablet",
"⟦ this⟧ woman was so worth it" - against numerals. `L21_SV51`: judgmental description ("New York
⟦ is⟧ just an overcrowded, over-stimulated, heartless community", "the bare buildings look dull⟦ and⟧
boring") against the structured spec (`Step` `Overall` `Workout` `Size` `Processor`).

**The institutional voice against lived narrative.** `L21_SV19`: `of` `a` `the` `The` `in` `for` in
programme-speak - "Compendium⟦ of⟧ knowledge of economic sciences", "⟦The⟧ Country Support Program
(CSP) is a GEF-funded", "you may consider the⟦ following⟧ situations" - against the sensory and
spoken (`"` `—` `[` `sun` `needle` `look` `wind`: "watching the⟦ sun⟧ set over Lake Buckatabon", "Your
cock will get so much⟦ sun⟧ and salt", "Tracy Morgan, he wanna do it⟦ [⟧too]"). `L19_SV55`: care and
kinship in the first person (`family` `mother` `your` `Child` `caregivers` `warm` `Nurse` `prayer`
`my`: "my relationship to the object of⟦ my⟧ belief", "allows⟦ caregivers⟧ to secure a child") against
the product chronicle ("re-re⟦leased⟧ for the Wii in 2009", "The car had clock⟦ed⟧ around 40,000km",
"debuted in the Official New Zealand Top 40 Albums⟦ on⟧ 10 January 2011"). `L17_SV19`: formal
expository openings against the raw existential noun (`sex` `death` `service` `suicide` `emotional`
`television` `voice` `electricity` `ultrasound` `election`).

**The marketing second person.** `L19_SV17` (positive): `Click`:22 `Book` `Christmas` `Comment`
`Print` - "⟦Click⟧ HERE to see how our Multipurpose Facial Oil", "⟦ Click⟧ here to watch their video".
`L02_SV18` above ("you've got"). `L06_SV52`: the vocabulary of growth (`emerging`:9 `More` `modern`
`greater` `richer` `enhanced`) against short bodily action verbs (`do` `tie` `tied` `drive` `bail`
`bite` `bind` `dial`: "⟦ tie⟧ the knot myself", "⟦ dial⟧ the specified").

**The garbled.** Several of UWS's most selective word-tails are populated by machine-spun essay-mill
prose: `L21_SV24` ("consider creating⟦ a⟧ lot of courseworks that rob you of important many hours",
"This is⟦ the⟧ nipple from a judgment i am hardly possible", "You know Scribd goes always bear!⟦ The⟧
third world had while the Web subscription figured"), `L22_SV58`, the `SV01` family at L9-L17 ("inda
finally managed⟦ to⟧ head home and Lindsey took Tony", "effectation of making use⟦ of⟧ different
recruitment methods"), and `L08_SV02`, which fires on outright disfluency ("how well⟦ well⟧-organized",
"a hard copy⟦ copy⟧", "we seem to get be⟦ getting⟧ back", "the Guwop-featured tune still is⟦ still⟧ banging"). Five
such documents (coastbridal.com, iyor2018.org, schroeder-alsleben.de, flashsblogs.tk,
location-menuires.eu) account for 2.0% of the selected-polarity events of UWS's top-100 selective SVs
(16 SVs put 3+ events there) against 0.6% for GSS (3 SVs); the corpus base rate is 0.7% vs 0.3%.

## 4. GSS: directions of reference and matter

**The elemental against the managerial.** `L00_SV35`: `water`:15 `sun`:11 `cooking` `gas` `cook`
`fish` `ice` `glass` `brown` `horse` `cake` `boat` `snow` `cow` `wine` `wool` `rain` `smoke` - against
`strategic`:12 `-specific` `Key` `targeted` `Additional` `Target` `Specific` `detailed` `-Based`
`-Term` `standardized` ("president of⟦ Strategic⟧ Energy", "a⟦ strategic⟧ sponsor", "at⟦ strategic⟧
places on the front and back sides of the shirt"). `L17_SV09`: `water`:24 `gas`:22 `gun` `dog` `cook`
`drink` `car` `wagon` `food` `dust` against the connectives of exposition (`However`:19 `Further`
`Finally` `Ultimately` `Alternatively` `Additionally` `Indeed`). The probe finds 9 GSS tails with 12+
elemental words vs 3 in UWS.

**The particular against the placeholder.** `L09_SV59` and `L12_SV57` (one direction, L9-L12): the
generic nouns of administration - `matters` `affairs` `whole` `operations` `hand` `-being` `moment`
`lives` `things` `moves` `present` `general` `needs`; `activities` `management` `benefits` `means`
`changes` `aspects` `multifaceted` ("all⟦ matters⟧ relevant to visas", "an uncomfortable state of
⟦ affairs⟧", "business⟦ operations⟧") - against the stubbornly specific: `oyster` `choking` `trigger`
`lice` `frosting` `candidiasis` `pesticide` `tattoo` `mustard` `patent` `rinse` `egg`; `lobster` `tiger`
`cereal` `cactus` `nitrate` ("an amazing⟦ oyster⟧ appetizer", "a⟦ choking⟧ hazard", "a⟦ trigger⟧ lock",
"1-day⟦ lice⟧ removal treatment", "oral⟦ candid⟧iasis"). `L01_SV10`: `Dragon` `hydraulic` `Cuban`
`Ronaldo` `cocaine` `Tomato` `chicken` `Egyptian` `Liverpool` `Sony` `USB` `turbo` `drum` `BMW`
`Dolphins` against `perspective` `bit` `detail` `considerations` `outlined` `aspects` `context`
`manner` `basis` `view` `respects`. `L01_SV18`: the brochure adjective (`innovative` `diverse`
`strategic` `Clinical` `Indigenous` `National` `Competitive` `Advanced` `Regional` `Professional`)
against the household object (`box` `boxes` `boomb⟦ox⟧` `extinguis⟦her⟧` `sau⟦cers⟧` `mail⟦boxes⟧`
`scall⟦ops⟧`). `L07_SV08`: the evidential hedge (`'t` `indicated` `suggest` `meant` `seems` `noted`
`considered` `relied`: "This shouldn⟦'t⟧ surprise us") against food and hardware (`chicken`:8 `car`
`solar` `hotel` `salmon` `restaurant` `laptop` `apple` `oven`: "Brush both sides of the⟦ salmon⟧ with
the olive oil", "the $5 precooked Costco rotissery⟦ chicken⟧"). The probe finds 11 GSS tails with 12+
concrete-everyday nouns vs 3 in UWS.

**The felt against the labelled.** `L08_SV28`: `so`:20 `feeling`:13 `but` `seem` `feel` `trembling`
`shivering` `swimming` `swirling` `seemingly` `saddened` `floating` - "work out our salvation in fear
and⟦ trembling⟧", "I feel you shivering and qu⟦ivering⟧", "yet feels⟦ so⟧ familiar at the same time"
- against `Project`:8 `Product`:8 `Main` `API` `PRODUCT` `PROJECT` `Custom` `Official` `Chief` `VIP`
`Feature` `Submit` ("⟦Product⟧ was successfully added to your shopping cart", "NEW⟦ PRODUCT⟧ NEWS").
`L00_SV33`: `felt`:33 `feel` `owed` `wore` `affords` `sow` `vow` `vowed` `owe` against `specific`:21
`traditional` `Different` `Custom` `complex` `Interactive` `Innovative` `Strategic` `Unique`
`Collaborative`. `L03_SV47`: bodies in space (`self` `Spider` `floating` `carbon` `neuro` `spatial`
`bone` `transported` `carving` `sexually`: "naked body was found by a fisherman⟦ floating⟧ in the Red
River", "walk through non-venomous⟦ spider⟧webs") against the norm (`requirements`:39 `Requirements`
`guidelines` `practices` `criteria` `recommended` `approved` `proposals`: "System⟦ Requirements⟧ and
Technical Details", "⟦ Requirements⟧ vary by state"). `L21_SV38`: origin and body (`Here` `Ground`
`Head` `Born` `Core` `Gene` `Bird` `Barn` `Father's` `Generation` `Dry` `Body`: "⟦Ground⟧ Beef Shredded
Chicken", "2013.B⟦orn⟧ December 18, 1955 in Honduras") against enhancement (`positive`:19 `promote`
`boost` `positively` `improve` `plus` `supplement` `tribute` `benefit`: "must have their skills and
qualifications⟦ positively⟧ assessed", "End the season on a⟦ positive⟧"). `L09_SV36`: the unprocessed
(`raw` `full` `pure` `clear` `max` `huge` `tiny` `clean` `captured`: "⟦ pure⟧, 100% nutritious envy",
"2PB (⟦raw⟧)") against the programme (`Service` `Relations` `involvement` `Cooperation` `Attorney`
`Instruction` `Orbit` `Leadership` `Affairs` `Intervention`: "the Distinguished⟦ Service⟧ Award", "The
Curriculum and⟦ Instruction⟧ Department"). `L06_SV38`: `kids`:37 `&` `veggies` `here` `hidden` `edgy`
`-friendly` `exciting` ("free⟦ kids⟧ craft table, zip line, rock wall") against `order` `refrain`
`pursuit` `pursuant` `reaffirm` `ensure` `provision` `assure` `reversal` `jurisprudence` `inquire`
("Fasting is not only⟦ refrain⟧ from eating and drinking").

**The apparatus of the world.** Physical infrastructure, one direction across L9-L16 (`transport`
`station` `port` `vehicle` `satellite` `cell` `atom` `Plant` `material` `surface` `platform`
`distribution`), 17 GSS tails vs 2 in UWS. Civic infrastructure, `L03_SV62`: `transportation`:16
`transport`:13 `municipal` `residential` `institutional` `municipality` `network` `coordinate`
`housing`. The state and the firm: `L11_SV56` (`Army`:9 `economy`:7 `beef` `office` `NIH` `military`
`California` `Health` `Justice` `Ottawa` `forces` `Agriculture` `Michigan`: "a stint in the U.S.
⟦ Army⟧", "The National Institutes of⟦ Health⟧", "the U.S. Department of⟦ Justice⟧") against the
apparatus of description (`match` `tag` `examples` `Label` `Describe` `characterized` `Definitions`
`descriptive`); `L13_SV56` (`U.S` `Office` `Inc` `Army` `economy` `Co` `Medicine` `Bank` `Engineers`:
"E.P. Dutton &⟦ Co⟧.", "the Dade County Medical Examiner's⟦ Office⟧"); `L09_SV55` (`investors`:10 `USA`
`Army` `university` `Indiana` `universities` `businesses` `inventories` `urban` `network` `managers`);
`L03_SV34` (`University`:41 `California`:18); `L19_SV23` (`project` `Project` `Business` `Request`
`Team` `Task` `Report` `Service`). 19 GSS tails vs 2 in UWS. The law: `L04_SV52` is `law`/`Law` in 59
of 64 events ("the Federal⟦ Law⟧ on concession agreements", "New Jersey State⟦ law⟧ requires us").
Rank and standing: `L05_SV54` (`rating` `status` `balance` `identity` `recognition` `ranking`
`branding` `relationship` `adherence` `validity`) against the vague quantity (`few`:52 `some`: "a
⟦ few⟧ odds and ends", "over the next⟦ few⟧ weeks"). The calendar and the map: `L18_SV52` (`June`:21
`Ohio` `Women` `Miami` `Midwest` `Wellington` `Greece`), 7 GSS tails vs 4.

**Death, trouble, remedy.** `L16_SV28`: `death`:27 `States` `failure` `birth` `health` `waste` `WWII`
`war` `disease` ("file a wrongful⟦ death⟧ claim", "electronic⟦ death⟧ certificates") against
`planning` `plan` `figuring` `simply` `picking` `figure` `guiding` `Consulting` `Adjust` `factoring`
("⟦ figuring⟧ out logistics", "start⟦ planning⟧ for the tight coupling") - the fact against the intent.
`L09_SV61`: `liver` `wall` `solve` `problem` `difficult` `hospital` `FBI` `lawyer` `prison` `Honduras`
`cure` `solution` `mouth` `Court` `vulnerable` `hurt` ("the teeth, tongue, salivary glands,⟦ liver⟧,
pancreas", "buyers are finding it⟦ difficult⟧ to push prices higher", "answer the⟦ problem⟧ to submit
your comment"). `L11_SV59`: `trust` `find` `understand` `tell` `confronted` `clear` `prove` `court`
`liar` `cough` `cure` `cracked` `argue` `catch` `confront` `hunting` `solve` ("Early detection is the
key to⟦ cure⟧ cancer", "Can't⟦ argue⟧ with you on that Mike", "she will never ever⟦ trust⟧ me anymore",
"chances after chances to '⟦prove⟧' themselves"). `L05_SV41`: `problems`:22 `fun`:13 `confused`:9
`confusion` `confuse` `confusing` `surprising` `complexities` `funny` `curious` `feared` `monsters`
`skepticism` ("Many people are⟦ confused⟧ about decaffeination", "a general⟦ confusion⟧⏎about how to get
the system to work"). The probe: 7 GSS tails with 12+ problem/remedy words vs 2 in UWS; 5 vs 2 for
violence/death.

GSS is not without discourse directions - it has `also` (`L09_SV28`, 63 of 64), `between`
(`L07_SV56`, 38: "any conflict⟦ between⟧ this Agreement and any Program Policy", "the distance
⟦ between⟧ these two areas"), `through` (`L15_SV29`, 31: "a fire tore⟦ through⟧ a nightclub", "they
cycle⟦ through⟧ ecosystems"), `because` (`L19_SV23` negative, 30: "Is it because you don't want to
eat at home?⏎Or is it⟦ because⟧ you don't want to cook?"), `just` (`L19_SV36` 51, `L07_SV54` 55), the
tourist's `boasts` (`L11_SV41`: "York⟦ boasts⟧ an exciting mix"), the em-dash aside (`L07_SV63`:
"nothing⟦—not⟧ one thing! — came into being", "Miles⟦—I⟧ believe— is just along for the ride"), and
rhetorical degree (`L19_SV32`/`L18_SV32`: `more` `not` `such` `much` `—not` `elegantly` `dramatically`
`magnificent` `meticulous`). But notice what these are: relations and connectives - between, through,
because, also - the joints of reference rather than the tone of the speaker. And notice what they are
set against: `Education` `Lady` `FDA` `Insurance` `Academy`; `Area` `Sea` `Box` `Place` `Country`
`Exchange`; `project` `Business` `Team` `Report` - labels of things, again.

## 5. Where the banks meet, and how they differ even there

* **Stance adverbs** are equally present (41 vs 43 tails): UWS `L09_SV58` (`just`/`simply`),
  `L07_SV58` (`pretty`); GSS `L19_SV36` and `L07_SV54` (`just`), `L11_SV41` (`boasts`, `absolutely`).
* **Perception** is in both, but not the same perception. UWS `L12_SV50` is the receptive mind -
  `heard` `seen` `listen` `felt` `trust` `believe` `read`. GSS `L09_SV51` pairs `see` `heard` `hear`
  `seen` with the stimulus itself - `Sound` `Noise` `Bell` `waves` `Notes`; GSS `L00_SV33` pairs
  `felt` `feel` with `wore` `owe` `vow` `sow`, against `specific` `traditional` `custom`.
* **Children** appear in both banks as the same joke told twice: UWS `L01_SV18` sets `kids`:40
  against `significant`:37 `comprehensive` `substantial`; GSS `L06_SV38` sets `kids`:37 `veggies`
  against `pursuant` `refrain` `provision` `jurisprudence`.
* **Warm appraisal**: the probe gives UWS more tails (10 vs 5), but the single most selective warm
  tail in either bank is GSS `L07_SV63` (rank 22): `lovely`:12 `solid`:7 `soul` `fulfill` `suit`
  `beautiful` `truly` `flawless` `fabulous` `worthwhile` `worth` ("a⟦ lovely⟧ 3-course Shabbat dinner",
  "⟦ solid⟧ evaluative evidence", "a change agent in the⟦ soul⟧") - and even there, GSS's opposite pole
  is the dash-aside, a structural thing, where UWS's wonder-tail is opposed by the price tag.
* **Broad, always-on directions**: UWS `L05-L08_SV03` carries the vocabulary of assessment -
  `potential`:10 `considering` `monitoring` `critical` `historical` `underlying` `hypothetical`
  `preliminary` `alleged` `initially` `optimal` `conventional` `revolutionary` ("analyze the
  ⟦ potential⟧ environmental consequences", "holistically⟦ considering⟧ the major economic and
  environmental factors", "The hackers⟦ allegedly⟧ stole"). GSS `L08-L17_SV02` carries the vocabulary
  of sectors - `legal` `data` `market` `health` `job` `clinical` `physical` `treatment` `tax`
  `global` `equity` `credit` `customer` `business` `product` ("the⟦ legal⟧ profession", "on-premises
  ⟦ data⟧ centers", "⟦ market⟧ capitalization", "a mental⟦ health⟧ crisis"). One bank's bulk direction
  says how the speaker holds a claim; the other's says which world the claim belongs to.
* UWS `L09_SV06` restates the register axis morphologically: `old` `under` `inner` `bare` `wild`
  `early` against `compensation` `convenience` `Communications` `contribution` `configuration`
  `accommodation` `Accountability`. UWS `L11_SV08`/`L10_SV08` sets the interjection ("Oh my⟦.⟧",
  "Well, well⟦,⟧ well", "Oh yass⟦!⟧") against British `-ised`/`-isation` and `nourishing` `enriching`
  `synthesis`.

## 6. Genre: what each bank's selective directions are drawn to

Looking at the documents that receive the most selected-polarity events from each bank's top-100
selective SVs: UWS's list is a pesticide-residue article with tables, a taxonomic description of an
annelid worm, five machine-spun essays, an Italian local history, a phone-accessory listing,
horse-racing results with times, a college statistics table, a bug tracker, a card-making supply
list, phone-number contact pages, camera specifications. GSS's list is a cosplay forum, a neuroscience
essay, a job listing, an ad-copywriting article, an oil-drilling report, a Department of Justice
press release, a pharmacology grant, 3D-software marketing, a development-bank investment framework,
a criminology programme, a World of Warcraft warlock blog, a Linux wiki, a construction newsletter, a
condo listing. Both banks spread over 1,600-1,750 of the 2,000 documents, so this is a matter of
emphasis, but the emphasis is legible: UWS gathers where meaning is thin or absent - numbers, lists,
salad - and GSS gathers where words are rare - jargon, names, specialist prose.

## 7. Reading the difference

A direction in the residual stream can carry meaning in two ways. It can carry *reference*: what
the words point at - substances, animals, institutions, dates, laws, diseases. Or it can carry
*stance*: how the utterance stands to its own content - permitting, marvelling, hedging, praising,
asserting, hoping, deflating. GSS's word-level directions fall largely on the referential side;
UWS's on the stance side. Their shared axis, Latinate against vernacular, is where the two meet,
because that axis is at once referential (abstractions against things) and attitudinal (officialdom
against talk); each bank reads it from its own side, GSS through the things, UWS through the talk.

The two banks' antitheses have different characters too. UWS's are rhetorical and almost literary:
imagination against the price tag; the cosmic against the permissive; `pretty` against `kill`; hope
and doubt against Commerce and acreage; care and prayer against the product chronicle. GSS's are
categorial: thing against abstraction; body against institution; substance against strategy;
feeling against Product; death against planning; the raw against the Service. If one had to name the
genre of each bank's tails, UWS's are epigrams and GSS's are taxonomies.

A bridge to the structural finding, offered as interpretation rather than proof. If a UWS direction
is defined by what it *anticipates* (the earlier result: it fires on the token before a number, a
line break, a determiner's noun), then what stays constant across its contexts is the posture toward
the not-yet-said - `may` anticipates a permitted act, `fantastic` anticipates a marvel, `It is`
anticipates a judgment, `hope` anticipates an outcome. Stance is the semantics of expectation. If a
GSS direction is defined by what has *already been read* (it fires on the first piece of a word,
holding its spelling so far), then what stays constant is the referent in hand - water, transport,
Army, law. Reference is the semantics of the given. The same asymmetry would then show at the level
of meaning as it did at the level of tokens. The word-salad attraction fits this: a direction tuned
to "what comes next?" is most excited precisely where nothing can be predicted, and spun text is the
limiting case of unrealized expectation; a direction tuned to word-form is unmoved by salad, whose
words are ordinary, and excited instead by jargon, whose words are rare.

## 8. Caveats

* These are readings. The theme lexicons were written after the reading, from the words that had
  been noticed, so the probe confirms the reading's consistency across all tails rather than
  testing it independently; counts of "tails with 12+ theme words" are inflated by directions that
  persist across several layers (UWS `SV29`, `SV30`; GSS `SV11`, `SV02`).
* Several poles are entangled with spelling, particularly in early layers: UWS's grand-systemic
  words are mostly c-initial (`crisis` `chaos` `climate` `culture` `community` `consensus` `cosmic`),
  `L01_SV23`'s consumer nouns are co-/ca- words (`costume` `coupon` `casino` `cocaine` `Costco`
  `coconut` `cosplay` `cookie`), GSS `L11_SV45`'s positive pole is a rhyme class (`Dean` `Brian`
  `brain` `Band` `Hand` `Khan` `grain`), `L01_SV09`'s ceremonial verbs share `em-`/`cel-` onsets, and
  `L18_SV61`'s playful adjectives share `ch-`/`cl-`/`j-`. Semantic coherence and orthographic
  coherence coincide often enough that neither reading excludes the other.
* About 160 tails per bank were read closely; everything else enters only through the lexicon probe
  and the document tally. The tails are the 64 most extreme events per pole, not the distribution.
