"""Hand-written knowledge of English, typed from memory (no corpus was consulted).

Everything numeric here is my own estimate:
  * WORDS: common English words in my estimated frequency order for 19th-century
    fiction/dialogue; frequency assigned by a Zipf-Mandelbrot law f(r) ~ 1/(r + 2.7).
  * NAMES: proper names and places I know occur in Pride and Prejudice (world
    knowledge, not counts). Kept separate so their contribution can be measured.
  * PHRASES: multi-word units whose internal word-to-word dependency matters.
  * SUFFIXES: inflection rules used to generate words outside the list.
  * Separator / punctuation probabilities.

`synth_text(n_chars, seed, names=True)` samples text from this generative model.
Its character statistics are what get written into the weights.
"""
import numpy as np

WORDS = """
the to of and a her i in was she that not it he you be his had as with for but is
have at him so my all on could would been by they no which very said from this
what were there me any such much an mr. do mrs. more if must will them their than
one only should we or who am your when miss own did how little know think never
can before now soon well nothing might being sister may has time see lady some
good every though again man great make made say other into upon should most dear
even herself himself shall too young without whom after first two way thought
long day family house own its our ever us same while always room father mother
quite cannot really believe give us here then just yet out up down over these
those where why both each few many many off once still under through between
about against during hope wish feel felt heard saw went came come go going told
tell took take seen known given done find found left gave seemed seem look looked
began set put kept let thing things friend friends happy happiness pleasure
manner manners letter answer replied cried added continued returned observed
said says say saying speak spoke spoken talk talked word words reason subject
nothing something anything everything nobody somebody everybody one another
however therefore indeed certainly perhaps rather almost already enough else
less least last next whole half because since until till unless whether
though although either neither nor yes oh ah sir madam ladies gentleman gentlemen
woman women men people world life heart mind eyes face hand hands home town
country place moment hour hours week weeks morning evening night year years
months month part side kind sort sense idea matter means name party ball dinner
visit walk dance dancing marriage married marry engaged daughter daughters
son wife husband aunt uncle cousin brother brothers sisters child children
girl girls young younger youngest elder eldest old new other others short whole
best better worse worst less more most greater greatest high low large small
real true certain sure full free able proud pride prejudice attention attentions
affection regard respect opinion character conduct behaviour situation
acquaintance consequence fortune estate income society civility pleasure
pleased pleasant agreeable amiable handsome beautiful charming fine poor
rich young happy unhappy glad sorry angry afraid ashamed surprised astonished
anxious uneasy impossible possible likely unlike like likely necessary
received receive wrote write written read reading heard hear hearing knew
know knowing understood understand meant mean meaning wished wanted want
loved love liked like thinking believed supposed suppose expected expect
obliged oblige determined determine assured assure assure intended intend
spirits temper feelings feeling sentiment sentiments interest power hope
doubt fear surprise wonder truth fact case instance account change chance
door window table carriage horse horses road way journey stay staying
immediately directly instantly presently scarcely hardly nearly entirely
perfectly exactly particularly especially generally really truly greatly
highly entirely soon often sometimes always never ever again away together
home abroad behind beyond within along among towards toward across around
""".split()

# v3: additional words typed from memory, continuing my estimated frequency order
# (ranks ~600 onwards; everyday vocabulary first, then the register of
# 19th-century novels of manners).
EXTRA_WORDS = """
also around another anyone thus whose whatever whenever wherever whichever
myself yourself itself ourselves themselves yours hers ours theirs mine
sat stood lay ran walked turned stopped passed asked called answered
brought bought taught caught thought fought sought held felt meant sent spent
became become becoming fell fallen rose risen grew grown drew drawn threw
showed shown shew shewn knew known wrote written spoke broken chose chosen
forgot forgotten forgave forgiven understood stood withstood undertook
lived living died dying moved moving opened opening closed closing
followed following allowed allow allowing began beginning begun
waited waiting watched watching listened listening laughed laughing smiled
smiling wondered wondering hoped hoping feared fearing tried trying
used using served serving reached reaching entered entering joined joining
carried carrying married marrying hurried hurrying worried worrying
remained remaining appeared appearing seemed seeming happened happening
considered considering desired desiring declared declaring repeated
mentioned mentioning explained explaining described describing
acknowledged acknowledge acknowledging admitted admit admitting
confessed confess confessing promised promise promising refused refuse
accepted accept accepting offered offer offering returned return returning
remember remembered remembering forget forgetting recollect recollected
imagine imagined imagining fancy fancied conceive conceived perceive
perceived perceiving observe observing discover discovered discovering
learn learned learnt learning teach teaching mind minded minding
concern concerned concerning regret regretted regretting rejoice rejoiced
deserve deserved deserving excuse excused apologise apologised
recommend recommended introduce introduced introduction invite invited
invitation visit visited visiting visitor visitors call calling caller
leave leaving depart departed departure arrive arrived arrival
stay stayed remain spend spending pass passing travel travelled
dine dined dining breakfast supper tea coffee meal meals cake wine
walk walking ride riding drive driving drove driven carriage chaise coach
house houses home homes park garden gardens grounds lawn wood woods
gate gates lane field fields hill hills river lake village town towns
street streets church parish parsonage rectory living cottage lodge inn
room rooms parlour drawing library hall staircase stairs chamber window
windows door doors fire chair chairs table sofa floor wall walls book
books paper papers pen ink letter letters note notes post express
music piano instrument song songs sing singing sang play playing played
cards game games ball balls assembly assemblies dance dances danced partner
partners couple couples set sets figure figures supper evening evenings
morning mornings day days afternoon noon night nights week weeks month
spring summer autumn winter season seasons weather rain sun wind cold
warm heat fine wet dry snow frost road roads miles mile distance near far
family families relation relations connexion connexions connection
parents parent child daughter niece nephew cousins nieces nephews
grandfather grandmother widow widower bride bridegroom lover lovers
friend friendship acquaintance acquaintances neighbour neighbours
neighbourhood stranger strangers guest guests company companion
servant servants housekeeper butler footman maid master mistress
gentleman lady lord duke earl baronet knight captain officer officers
militia regiment army navy soldier soldiers clergyman clergy rector
curate bishop lawyer attorney physician apothecary merchant trade
tradesman business affairs property estate estates fortune fortunes
income money pounds shillings thousand thousands hundred hundreds
year a-year settlement settlements will entail entailed heir heiress
marriage marriages wedding engagement proposal proposals addresses
attachment attached affection affectionate affectionately love lovely
loving beloved admire admired admiring admiration esteem esteemed
regard regarded respect respected respectable respectful contempt
dislike disliked hate hated hatred resentment resentful anger angry
temper tempered vexed vexation provoked provoking offended offence
insult insulted insolence insolent impertinent impertinence rude
rudeness civil civility polite politeness courteous courtesy
manner manners behaviour conduct deportment address air figure
countenance face features eyes looks look expression smile smiles voice
tone tones words speech conversation conversations discourse
subject subjects topic opinion opinions sentiment judgement judgment
understanding sense sensible senseless reason reasonable unreasonable
knowledge wisdom wise foolish folly fool fools silly silliness absurd
absurdity ridiculous nonsense clever cleverness talent talents
accomplished accomplishment accomplishments elegant elegance taste
style fashion fashionable genteel vulgar vulgarity low inferior
superior superiority rank station birth family consequence importance
important dignity pride proud vanity vain conceit conceited humble
humility modest modesty shy shyness reserve reserved open openness
frank frankness candid candour sincere sincerity honest honesty
truth true truly false falsehood deceit deceive deceived deception
character characters disposition dispositions nature natural naturally
heart hearts soul feelings feeling sensibility tender tenderness
gentle gentleness kind kindness kindly goodness generous generosity
liberal benevolent benevolence charity charitable cruel cruelty
selfish selfishness unkind unfeeling indifferent indifference
cold coldness warmth warm eager eagerness anxious anxiety uneasiness
distress distressed misery miserable wretched wretchedness unhappy
unhappiness sorrow grief tears cry crying wept weep weeping comfort
comforted comfortable uncomfortable relief relieved consolation
joy joyful delight delighted delightful pleasure pleasures pleased
satisfied satisfaction gratified gratitude grateful thankful thanks
hope hopes hopeful hopeless despair doubt doubts doubtful certain
certainty uncertain uncertainty suspense surprise surprised surprising
astonishment astonished amazement amazed wonder wonderful shock shocked
alarm alarmed fear fears fearful afraid terror dread danger dangerous
safe safety secure security risk chance chances fate fortune lucky
unlucky misfortune misfortunes evil evils good bad worse worst ill
health healthy illness sick sickness cold fever recover recovered
better well strong weak weakness tired fatigue rest sleep slept awake
dress dressed dresses gown gowns bonnet hat cap gloves lace ribbon
silk muslin coat shoes clothes fine finery appearance beauty pretty
plain handsome handsomer handsomest beautiful ugly tall short young
youth old age elderly older oldest twenty thirty forty fifty
one two three four five six seven eight nine ten eleven twelve
first second third fourth fifth last next former latter whole half
several various numerous many few fewer every each either both
sudden suddenly quick quickly slow slowly hasty hastily soon sooner
early earlier late later lately recently formerly hitherto henceforth
meanwhile afterwards before beforehand already yet still always
ever forever seldom rarely frequently often usually commonly ordinarily
certainly surely undoubtedly doubtless probably possibly perhaps
evidently obviously plainly clearly openly secretly privately publicly
seriously gravely earnestly warmly coldly calmly quietly silently
loudly eagerly anxiously gladly happily sadly unhappily heartily
sincerely honestly fairly justly properly decently tolerably extremely
exceedingly excessively infinitely remarkably uncommonly particularly
chiefly mostly mainly merely simply only even just hardly scarcely
barely nearly almost quite rather somewhat little much far
whatever however whoever wherever therefore thereby wherefore hence
thence whence accordingly consequently nevertheless notwithstanding
moreover besides otherwise else instead indeed truly really actually
except excepting unless until whilst amongst amidst towards upon
beneath beside besides beyond throughout within without inside outside
above below near nearer nearest opposite round about across
agree agreed agreeable agreeably disagree disagreeable consent consented
approve approved approval disapprove disapproved disapprobation
blame blamed praise praised commend commended censure condemn condemned
judge judged judging decide decided decision determine resolution
resolved resolve intention intentions design designs scheme schemes
plan plans purpose purposes object objects wish wishes wishing desire
desires want wants need needs necessity necessary require required
ought shall should must may might can could will would dare
able unable ability power powers authority influence persuade
persuaded persuasion convince convinced conviction prove proved proof
evidence fact facts account accounts report reports news intelligence
rumour rumours story stories tale history circumstance circumstances
situation event events occasion occasions incident affair matter matters
business concern concerns interest interests advantage advantages
benefit profit loss gain object view views prospect prospects
expectation expectations hope design future past present time times
moment moments instant minute minutes hour hours period while
beginning end ending close conclusion finish finished complete
completed whole entire entirely wholly partly part parts portion
share piece kind kinds sort sorts manner degree degrees measure
point points line lines side sides place places spot quarter
country countries county shire kingdom world earth nature heaven
god lord christian church sunday prayer duty duties obligation
obliged obliging honour honours honourable dishonour shame shameful
ashamed disgrace disgraceful guilt guilty innocent innocence crime
fault faults error errors mistake mistaken wrong wrongs right rights
justice injustice injury injured injure harm hurt pain painful
suffering suffer suffered bear bore borne endure endured patience
patient impatience impatient calm composure composed agitation
agitated emotion emotions passion passions violent violence spirit
spirits lively liveliness cheerful cheerfulness gay gaiety merry
dull dulness grave gravity serious solemn thoughtful thought thoughts
reflection reflections reflect reflected consider consideration
attention attentive attentions notice noticed regard heed care cared
careful careless carelessness caution cautious prudent prudence
imprudent imprudence rash rashness folly wild wildness thoughtless
foolishness weakness strength firm firmness steady steadiness
constant constancy fickle inconstant faithful faithless trust trusted
confidence confide confided secret secrets mystery suspicion suspect
suspected jealous jealousy envy envious rival rivals compete
head heads arm arms foot feet leg legs back shoulder shoulders neck hair
lips mouth cheek cheeks brow forehead ear ears nose finger fingers body
blood breath breathe breathing step steps pace walk path way ways track
glance glanced gaze gazed stare stared view viewed sight sights seen
hear heard sound sounds noise silence silent quiet still stillness
speak speaking spoken utter uttered whisper whispered exclaim exclaimed
exclamation reply replies answer answered respond rejoined interrupted
interrupt continued continue proceed proceeded resume resumed pause
paused hesitate hesitated hesitation stammer silence ceased cease
begin begins began end ended ends finish start started starting
give gives giving gave take takes taking took get gets getting got gotten
put puts putting set sets setting keep keeps keeping kept hold holds
holding bring brings bringing send sends sending carry carries lead led
leads leading follow follows show shows showing tell tells telling ask
asks asking seek seeking find finds finding lose losing lost win won
winning meet meets meeting met part parted parting join joined unite
united separate separated separation divide divided open opened shut
rise rising raise raised lift lifted fall falling drop dropped
sit sitting stand standing lie lying lay laid run running hurry rush
rushed fly flew flight escape escaped return come comes coming came go
goes going gone went arrive enter leave leaves quit wait await awaited
look looks looking looked see sees seeing saw watch notice observe
think thinks thinking believe believes believing know knows knowing
understand understands feel feels feeling seem seems appear appears
become becomes grow grows growing turn turns turning change changes
changed changing alter altered alteration improve improved improvement
increase increased decrease lessen lessened add adds adding join
make makes making made do does doing done did say says saying said
write writes writing read reads speak speaks talk talks talking tell
call calls name named names word words letter sentence page volume
chapter story novel poem poetry verse reading writing hand handwriting
person persons people man men woman women boy boys girl girls child
children baby infant mother father brother sister wife husband son
daughter uncle aunt cousin nephew niece friend enemy enemies lover
mama papa sir madam ma'am miss mister master mistress lady ladies
gentlemen fellow fellows creature creatures being beings soul souls
everybody everyone somebody someone anybody nobody none nothing
anything something everything all whole each every any some no
own same such other another else either neither both few many more most
much less least enough plenty several certain various different
alike like unlike similar equal equally same different difference
differ differed distinguish distinguished distinction notice
remarkable remark remarked remarks comment observation observed
true real reality truth actual actually indeed sure surely certain
wrong right correct exact exactly precise precisely proper improper
fit fitted suitable unsuitable becoming unbecoming decent indecent
natural unnatural common uncommon usual unusual ordinary extraordinary
strange stranger strangely odd oddly singular singularly curious
curiosity interesting interested uninteresting tedious tiresome
amusing amusement amuse amused entertain entertained entertainment
diversion diversions pleasure leisure idle idleness busy employment
employed occupation occupied engage engaged engagement engagements
work worked working labour toil effort efforts attempt attempted try
succeed succeeded success successful fail failed failure lose
afford afforded cost costs price expense expenses expensive cheap
poor poverty rich riches wealth wealthy comfortable luxury luxuries
want wanted wanting lack lacking need needed needful useful useless
use used usage purpose service services serve served help helped
assist assistance aid support supported supporting depend depended
dependent independent independence free freedom liberty slave
command commanded order ordered orders direct directed direction
directions guide guided rule rules ruled govern governed government
law laws legal claim claims claimed demand demanded request requested
beg begged beseech entreat entreated entreaty implore pray prayed
thank thanked bless blessed blessing curse cursed swear swore oath
promise promised vow vowed pledge assure assured assurance assurances
believe belief faith trust hope charity
nice nicer pleasant unpleasant agreeable charming delightful lovely
sweet sweetness bitter bitterness sharp sharply soft softly hard
hardly harsh harshly rough smooth light lighter dark darker darkness
bright brightness clear clearer dim pale paler colour coloured red
white black blue green brown grey golden gold silver
large larger largest small smaller smallest little less big bigger
great greater greatest huge vast immense tiny wide narrow long longer
longest short shorter high higher highest low lower lowest deep
deeper shallow thick thin full empty heavy light
new newer old older young younger fresh recent ancient modern
early late quick slow fast hasty sudden gradual gradually
easy easily difficult difficulty difficulties hard simple plain
possible impossible probable improbable likely unlikely sure unsure
glad sorry happy unhappy pleased displeased content discontent
contented satisfied dissatisfied angry calm quiet noisy loud
""".split()

NAMES = """
elizabeth darcy jane bennet bingley wickham collins lydia kitty mary catherine
lucas charlotte gardiner longbourn netherfield pemberley meryton rosings hunsford
lizzy fitzwilliam georgiana bourgh phillips forster brighton derbyshire
hertfordshire london lambton hurst denny william de lady sir colonel
""".split()
NAME_RANK_START = 40          # names behave like frequent mid-rank words

PHRASES = [  # (phrase, relative weight vs. a word of rank 20)
    ("of the", 3.0), ("in the", 2.0), ("to the", 1.5), ("to be", 1.5), ("it was", 1.2),
    ("i am", 1.0), ("she had", 0.8), ("he had", 0.8), ("i have", 0.8), ("she was", 1.0),
    ("he was", 0.9), ("could not", 0.9), ("would not", 0.6), ("do not", 0.6),
    ("did not", 0.6), ("had been", 0.8), ("have been", 0.6), ("as she", 0.4),
    ("and the", 0.8), ("on the", 0.7), ("at the", 0.6), ("for the", 0.6), ("by the", 0.5),
    ("with the", 0.6), ("from the", 0.5), ("that he", 0.5), ("that she", 0.5),
    ("of her", 1.0), ("to her", 0.8), ("of his", 0.6), ("in a", 0.6), ("as to", 0.4),
    ("such a", 0.4), ("so much", 0.4), ("i do not", 0.4), ("i cannot", 0.3),
    ("there was", 0.5), ("there is", 0.3), ("he is", 0.4), ("she is", 0.4),
    ("young man", 0.3), ("young lady", 0.2), ("your sister", 0.2), ("my dear", 0.6),
    ("of course", 0.2), ("at least", 0.2), ("as soon as", 0.3), ("in short", 0.1),
]
NAME_PHRASES = [
    ("mr. darcy", 2.5), ("mrs. bennet", 1.5), ("mr. bennet", 1.2), ("mr. bingley", 1.5),
    ("miss bingley", 0.8), ("mr. collins", 1.2), ("mr. wickham", 0.8), ("lady catherine", 0.8),
    ("sir william", 0.4), ("miss bennet", 0.4), ("colonel fitzwilliam", 0.3),
    ("mrs. gardiner", 0.4), ("miss darcy", 0.3), ("de bourgh", 0.3), ("mrs. phillips", 0.2),
]

SUFFIX_RULES = [  # applied to a list word to make an out-of-list word: (suffix, weight)
    ("s", 4), ("ed", 3), ("ing", 3), ("ly", 2), ("er", 1), ("est", 0.5), ("ness", 0.6),
    ("ment", 0.5), ("ion", 0.5), ("able", 0.3), ("ful", 0.3), ("less", 0.2), ("ity", 0.2),
]

# what follows a word (before the next word); probabilities sum to 1
SEPARATORS = [
    (" ", 0.835), (", ", 0.075), (". ", 0.040), ("; ", 0.012), ("! ", 0.004), ("? ", 0.005),
    (": ", 0.002), (",\" ", 0.006), (".\" ", 0.007), ("?\" ", 0.002), ("!\" ", 0.002),
    ("-", 0.003), ("--", 0.003),
]
P_OPEN_QUOTE = 0.012      # a word preceded by an opening quote
P_ITALIC = 0.002          # Gutenberg convention: _word_ for italics
P_PHRASE = 0.12           # a phrase unit instead of a single word
P_DERIVED = 0.10          # a suffixed out-of-list word instead of a list word
P_OOV = 0.0               # v1 = 0.0; v2 sets 0.25 (a novel word from the syllable model)

# v2: syllable model for words outside the list (English phonotactics, from memory)
ONSETS = [("", 3), ("t", 2), ("s", 2), ("r", 1.5), ("l", 1.5), ("m", 1.5), ("n", 1.5), ("d", 1.5),
          ("p", 1.2), ("c", 1.2), ("b", 1), ("f", 1), ("h", 1), ("w", 0.8), ("g", 0.7), ("v", 0.5),
          ("k", 0.3), ("j", 0.2), ("y", 0.2), ("st", 0.5), ("th", 0.8), ("sh", 0.4), ("ch", 0.4),
          ("pr", 0.5), ("tr", 0.4), ("gr", 0.3), ("br", 0.3), ("cr", 0.3), ("fr", 0.3), ("pl", 0.3),
          ("cl", 0.3), ("bl", 0.2), ("fl", 0.2), ("sp", 0.2), ("wh", 0.3), ("qu", 0.15), ("str", 0.1),
          ("sc", 0.1)]
NUCLEI = [("e", 3), ("a", 2.5), ("i", 2.5), ("o", 2), ("u", 1), ("ea", 0.5), ("ou", 0.5), ("ai", 0.3),
          ("ee", 0.3), ("oo", 0.2), ("ie", 0.2), ("y", 0.3), ("io", 0.3)]
CODAS = [("", 3), ("n", 2), ("r", 1.5), ("s", 1.5), ("t", 1.5), ("l", 1.2), ("d", 1), ("m", 0.6),
         ("nt", 0.6), ("nd", 0.6), ("st", 0.4), ("ng", 0.4), ("ss", 0.3), ("ll", 0.3), ("ct", 0.2),
         ("ck", 0.2), ("rt", 0.2), ("rs", 0.2), ("ns", 0.2), ("sh", 0.1), ("ch", 0.1), ("th", 0.2),
         ("x", 0.05)]
ENDINGS = [("", 3), ("e", 0.6), ("s", 0.8), ("ed", 0.5), ("ing", 0.5), ("ly", 0.3), ("tion", 0.3),
           ("er", 0.4), ("y", 0.3), ("al", 0.2), ("ment", 0.1), ("ness", 0.1), ("ity", 0.1),
           ("ous", 0.1), ("ance", 0.05), ("ence", 0.05)]
N_SYLL = [(1, 0.3), (2, 0.45), (3, 0.2), (4, 0.05)]


def _pick(rng, table):
    p = np.array([w for _, w in table], float)
    return table[rng.choice(len(table), p=p / p.sum())][0]


def novel_word(rng):
    w = "".join(_pick(rng, ONSETS) + _pick(rng, NUCLEI) + _pick(rng, CODAS)
                for _ in range(_pick(rng, N_SYLL)))
    return w + _pick(rng, ENDINGS)


def _zipf(n, start=0):
    r = np.arange(start, start + n)
    return 1.0 / (r + 2.7)


def _dedupe(ws):
    seen, out = set(), []
    for w in ws:
        if w not in seen:
            seen.add(w); out.append(w)
    return out


def synth_text(n_chars, seed=0, names=True, p_oov=None, extra=False, p_derived=None):
    """extra=True adds EXTRA_WORDS (prior v3)."""
    p_oov = P_OOV if p_oov is None else p_oov
    p_derived = P_DERIVED if p_derived is None else p_derived
    rng = np.random.default_rng(seed)
    words = _dedupe(WORDS + (EXTRA_WORDS if extra else []))
    wf = _zipf(len(words))
    n_general = len(words)          # suffix rules only apply to content words (rank >= 100), never names
    if names:
        nm = [n for n in _dedupe(NAMES) if n not in words]
        words = words + nm
        wf = np.concatenate([wf, _zipf(len(nm), NAME_RANK_START) * 0.5])
    wp = wf / wf.sum()
    phrases = PHRASES + (NAME_PHRASES if names else [])
    pp = np.array([w for _, w in phrases]); pp = pp / pp.sum()
    sp = np.array([p for _, p in SEPARATORS]); sp = sp / sp.sum()
    sfx = np.array([w for _, w in SUFFIX_RULES], float); sfx = sfx / sfx.sum()
    out, n = [], 0
    while n < n_chars:
        u = rng.random()
        if u < P_PHRASE:
            unit = phrases[rng.choice(len(phrases), p=pp)][0]
        elif u > 1 - p_oov:
            unit = novel_word(rng)
        else:
            wi = rng.choice(len(words), p=wp)
            w = words[wi]
            if u < P_PHRASE + p_derived and 100 <= wi < n_general and len(w) >= 4 and w.isalpha():
                s = SUFFIX_RULES[rng.choice(len(SUFFIX_RULES), p=sfx)][0]
                if s in ("ed", "er", "est", "able") and w.endswith("e"):
                    w = w[:-1]
                w = w + s
            unit = w
        if rng.random() < P_ITALIC:
            unit = "_" + unit + "_"
        if rng.random() < P_OPEN_QUOTE:
            unit = '"' + unit
        sep = SEPARATORS[rng.choice(len(SEPARATORS), p=sp)][0]
        if unit.endswith(".") and sep.startswith("."):
            sep = sep[1:] or " "
        piece = unit + sep
        out.append(piece)
        n += len(piece)
    return "".join(out)[:n_chars]


if __name__ == "__main__":
    print(synth_text(800, seed=1))
    print()
    print(synth_text(800, seed=1, p_oov=0.25))
    print()
    print(synth_text(800, seed=1, p_oov=0.12, extra=True, p_derived=0.05))
