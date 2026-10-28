"""
substances.py - the lab's substance library and the lookups Muse uses before writing an experiment.

Two things live here:

1. LIBRARY: substances beyond the seven classic drugs, each stated as a MECHANISM on Maxime's handles
   (brain.params), with the evidence it rests on and an honest confidence label. The entries that have
   handles are merged into pharmacology.DRUGS, so "dose deet 0.8" works like any other drug.
   Entries with no handle (sugar, salt) stay in the library so Muse can say *why* the lab cannot test them yet.

2. lookup(name): what the lab knows about a substance, plus two public sources fetched live when the
   machine is online - PubChem (formula, IUPAC name, description) and Wikipedia (first paragraph).
   These are returned as text for the experimenter to read; they never change the simulation.
   Nothing here is written for people. Maxime is a simulated fly; every mechanism is stated for a fly brain.

Confidence labels
   "established in Drosophila"   the mechanism was shown in fruit flies (citation given)
   "insect evidence"             shown in another insect (mosquito, cockroach), assumed to carry over
   "inferred"                    mammalian pharmacology mapped onto the fly's counterpart molecules
   "not testable yet"            the lab has no handle for it (e.g. taste) - say so, do not pretend
"""
import json, os, re, time, urllib.parse, urllib.request
from pharmacology import hill

LIBRARY = {
    # ---- repellents -------------------------------------------------------------------------------
    'deet': dict(
        aliases=['DEET', 'N,N-diethyl-meta-toluamide', 'diethyltoluamide', 'OFF! Deep Woods', 'Repel 100', 'Cutter Backwoods'],
        kind='repellent (active ingredient)',
        mechanism="inhibits odorant-receptor (Orco-dependent) responses of olfactory receptor neurons to attractive odours; also detected by bitter-taste neurons on contact",
        handles=[('I_sens', 'mult', lambda c: 1.0 - 0.85 * hill(c, ec50=0.3, n=2))],
        half_life=1500,
        confidence="established in Drosophila",
        evidence=["Ditzen, Pellegrino & Vosshall (2008) Science 319:1838 - DEET inhibits Orco-dependent odorant receptor responses in Drosophila ORNs",
                  "Lee, Kim & Montell (2010) Curr Biol 20:1672 - DEET is sensed as bitter by Gr66a gustatory neurons in Drosophila"],
        not_modelled="the contact (taste) route: Maxime has no gustatory neurons wired yet, so only the smell-masking route is tested",
    ),
    'picaridin': dict(
        aliases=['icaridin', 'KBR 3023', 'Sawyer Picaridin', 'Natrapel', 'OFF! FamilyCare'],
        kind='repellent (active ingredient)',
        mechanism="reduces odorant-receptor responses of olfactory receptor neurons to host/food odours",
        handles=[('I_sens', 'mult', lambda c: 1.0 - 0.6 * hill(c, ec50=0.4, n=2))],
        half_life=1500,
        confidence="insect evidence",
        evidence=["Bohbot & Dickens (2010) PLoS ONE 5:e12138 - picaridin and DEET modulate Aedes aegypti odorant receptors in oocytes"],
        not_modelled="any taste route",
    ),
    'citronellal': dict(
        aliases=['citronella', 'citronella oil', 'lemongrass oil', 'Cutter Citro Guard'],
        kind='repellent (essential-oil component)',
        mechanism="activates TRPA1 channels and odorant receptors in olfactory receptor neurons regardless of the odour present (odour-independent receptor drive)",
        handles=[('I_sens_bg', 'add', lambda c: 6.0 * hill(c, ec50=0.4, n=2))],
        half_life=800,
        confidence="established in Drosophila",
        evidence=["Kwon et al. (2010) Curr Biol 20:1672 - Drosophila TRPA1 in ORNs is required to avoid citronellal; citronellal excites ORNs via Or and TRPA1 pathways"],
        not_modelled="the aversion pathway itself: Maxime's readout has no 'walk away' circuit, so the lab tests whether the banana signal is drowned out",
    ),
    'pmd': dict(
        aliases=['p-menthane-3,8-diol', 'oil of lemon eucalyptus', 'OLE', 'Repel Lemon Eucalyptus', 'Cutter Lemon Eucalyptus'],
        kind='repellent (active ingredient)',
        mechanism="excites olfactory receptor neurons independently of food odour (odorant-receptor and TRP-channel activation)",
        handles=[('I_sens_bg', 'add', lambda c: 4.0 * hill(c, ec50=0.5, n=2))],
        half_life=800,
        confidence="insect evidence",
        evidence=["Drapeau et al. (2009) and mosquito electrophysiology: PMD activates ORNs and is avoided; direct Drosophila receptor work is thin"],
        not_modelled="the aversion pathway (see citronellal)",
    ),
    # ---- insecticides -----------------------------------------------------------------------------
    'permethrin': dict(
        aliases=['pyrethroid', 'pyrethrin', 'cypermethrin', 'prallethrin', 'deltamethrin', 'Raid', 'Ortho Home Defense', 'Sawyer Permethrin', 'Nix'],
        kind='insecticide (pyrethroid class)',
        mechanism="holds voltage-gated sodium channels (para) open: prolonged depolarising current and repetitive firing; at high concentration a depolarising conduction block silences neurons",
        handles=[('I_bg', 'mult', lambda c: 1.0 + 1.5 * hill(c, ec50=0.3, n=2)),
                 ('noise', 'mult', lambda c: 1.0 + 1.0 * hill(c, ec50=0.3, n=2)),
                 ('ttx_frac', 'set', lambda c: hill(c, ec50=1.1, n=5))],
        half_life=1200,
        confidence="established in Drosophila",
        evidence=["Soderlund et al. (2002) Toxicology 171:3 - pyrethroid action on insect sodium channels",
                  "Pittendrigh et al. (1997) Mol Gen Genet 256:602 - para sodium-channel (kdr) mutations confer pyrethroid resistance in Drosophila"],
        not_modelled="muscle and nerve-cord effects: Maxime's body is driven only by descending neurons",
    ),
    # ---- stimulants / transporter drugs -------------------------------------------------------------
    'amphetamine': dict(
        aliases=['Adderall', 'dextroamphetamine', 'mixed amphetamine salts', 'amphetamine sulfate'],
        kind='stimulant (monoamine releaser)',
        mechanism="substrate of the dopamine transporter (dDAT): reverses transport and empties dopamine vesicles, raising dopamine release; weaker effect on the octopamine/tyramine transporter",
        handles=[('g_DA', 'mult', lambda c: 1.0 + 2.0 * hill(c, ec50=0.4, n=2)),
                 ('g_OCT', 'mult', lambda c: 1.0 + 0.5 * hill(c, ec50=0.5, n=2))],
        half_life=800,
        confidence="established in Drosophila (dopamine); inferred (octopamine)",
        evidence=["Freyberg et al. (2016) Nat Commun 7:10652 - amphetamine action watched on dopamine synaptic vesicles in the Drosophila brain",
                  "Pizzo et al. (2013) Mol Psychiatry 18:824 - dDAT is the site of amphetamine action in Drosophila"],
        not_modelled="dopamine as a slow modulator: here dopamine neurons' synapses simply release more (sign per Shiu et al. 2024 convention)",
    ),
    'bupropion': dict(
        aliases=['Wellbutrin', 'Zyban', 'Aplenzin'],
        kind='transporter inhibitor',
        mechanism="inhibits the dopamine transporter (dDAT) and the noradrenaline-type transporter (octopamine in the fly), so released dopamine and octopamine act longer; also a non-competitive antagonist at nicotinic acetylcholine receptors",
        handles=[('g_DA', 'mult', lambda c: 1.0 + 0.8 * hill(c, ec50=0.5, n=2)),
                 ('g_OCT', 'mult', lambda c: 1.0 + 0.4 * hill(c, ec50=0.5, n=2)),
                 ('g_ACH', 'mult', lambda c: 1.0 - 0.25 * hill(c, ec50=0.6, n=2))],
        half_life=1000,
        confidence="inferred",
        evidence=["Stahl et al. (2004) Prim Care Companion J Clin Psychiatry 6:159 - DAT/NET inhibition (mammalian)",
                  "Slemmer, Martin & Damaj (2000) J Pharmacol Exp Ther 295:321 - nicotinic receptor antagonism (mammalian)",
                  "Porzgen et al. (2001) Mol Pharmacol 59:83 - dDAT, the Drosophila dopamine transporter, is the fly counterpart"],
        not_modelled="no fly experiment with bupropion is cited: this entry maps mammalian mechanisms onto the fly's transporters",
    ),
    'cocaine': dict(
        aliases=[],
        kind='transporter inhibitor',
        mechanism="blocks the dopamine transporter (dDAT), so released dopamine stays in the synapse longer",
        handles=[('g_DA', 'mult', lambda c: 1.0 + 1.5 * hill(c, ec50=0.4, n=2))],
        half_life=600,
        confidence="established in Drosophila",
        evidence=["Bainton et al. (2000) Curr Biol 10:187 - dopamine modulates cocaine responses in Drosophila",
                  "Porzgen et al. (2001) Mol Pharmacol 59:83 - cocaine-sensitive dDAT"],
        not_modelled="dopamine as a slow modulator (see amphetamine)",
    ),
    # ---- things the lab cannot test as a dose (yet) ----------------------------------------------------
    'sucrose': dict(
        aliases=['sugar', 'table sugar', 'glucose', 'fructose', 'honey', 'soda', 'maple syrup'],
        kind='food',
        mechanism="tasted by sweet gustatory receptor neurons (Gr5a, Gr64 family) on the legs and proboscis; feeds the fly",
        handles=[], half_life=0,
        confidence="not testable yet",
        evidence=["Dahanukar et al. (2007) Neuron 56:503 - Gr5a/Gr64a sweet taste receptors in Drosophila"],
        not_modelled="taste and feeding: Maxime has no gustatory neurons or energy state. Sugar can only be used as the banana (the smell source), not as a dose",
    ),
    'sodium chloride': dict(
        aliases=['salt', 'table salt', 'NaCl', 'sea salt'],
        kind='food/mineral',
        mechanism="low salt is attractive and high salt aversive through gustatory neurons (IR76b, Gr66a, ppk23 classes)",
        handles=[], half_life=0,
        confidence="not testable yet",
        evidence=["Zhang, Ni & Montell (2013) Science 340:1334 - IR76b salt taste in Drosophila", "Jaeger et al. (2018) eLife 7:e37167 - high-salt aversion circuits"],
        not_modelled="taste: Maxime has no gustatory neurons. The lab cannot test salt as a dose",
    ),
}

# aliases for the seven classic drugs in pharmacology.DRUGS, so lookup('coffee') lands on caffeine
CLASSIC_ALIASES = {
    'caffeine': ['coffee', 'espresso', 'energy drink', 'Red Bull', 'Monster', 'tea', 'NoDoz'],
    'ethanol': ['alcohol', 'beer', 'wine', 'vodka', 'whisky', 'whiskey', 'rum'],
    'nicotine': ['tobacco', 'cigarette', 'vape', 'nicotine patch'],
    'picrotoxin': ['cocculus'],
    'isoflurane': ['anaesthetic', 'anesthetic', 'general anaesthesia', 'general anesthesia'],
    'ttx': ['tetrodotoxin', 'pufferfish toxin', 'fugu'],
    'tetanus_toxin': ['tetanus', 'tetanospasmin'],
}


def drug_entries():
    """LIBRARY entries with handles, in the shape pharmacology.DRUGS expects."""
    out = {}
    for name, e in LIBRARY.items():
        if e['handles']:
            out[name] = dict(mechanism=e['mechanism'], half_life=e['half_life'], effects=list(e['handles']),
                             confidence=e['confidence'], evidence=list(e['evidence']), kind=e['kind'])
    return out


def _norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()


def find(name, drugs=None):
    """Match a free-text name against the library and the classic drugs. Returns (key, how) or (None, None)."""
    q = _norm(name)
    if not q:
        return None, None
    cands = []
    for key, e in LIBRARY.items():
        for alias in [key] + e['aliases']:
            a = _norm(alias)
            if a == q: return key, 'exact'
            if a and (a in q or (len(q) > 3 and q in a)): cands.append((len(a), key))
    for key, aliases in CLASSIC_ALIASES.items():
        for alias in [key.replace('_', ' '), key] + aliases:
            a = _norm(alias)
            if a == q: return key, 'exact'
            if a and (a in q or (len(q) > 3 and q in a)): cands.append((len(a), key))
    if drugs:
        for key in drugs:
            if _norm(key) == q: return key, 'exact'
    if cands:
        cands.sort(reverse=True)
        return cands[0][1], 'partial'
    return None, None


# ---- live sources -------------------------------------------------------------------------------
_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebook', 'lookups.json')
_UA = {'User-Agent': 'maxime-neuro-lab/0.1 (hackathon demo; contact via GitHub)'}


def _get_json(url, timeout=12):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout)
    return json.loads(r.read().decode('utf-8'))


def pubchem(name):
    """Formula, weight, IUPAC name and a one-paragraph description from PubChem (PUG REST)."""
    q = urllib.parse.quote(name)
    out = {'source': 'PubChem', 'url': f'https://pubchem.ncbi.nlm.nih.gov/#query={q}'}
    props = _get_json(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/property/MolecularFormula,MolecularWeight,IUPACName,Title/JSON')
    p = (props.get('PropertyTable') or {}).get('Properties') or [{}]
    p = p[0]
    out.update(cid=p.get('CID'), title=p.get('Title'), formula=p.get('MolecularFormula'), weight=p.get('MolecularWeight'), iupac=p.get('IUPACName'))
    if out.get('cid'):
        out['url'] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{out['cid']}"
        try:
            d = _get_json(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{out['cid']}/description/JSON")
            infos = (d.get('InformationList') or {}).get('Information') or []
            descs = [i for i in infos if i.get('Description')]
            if descs:
                out['description'] = descs[0]['Description'][:700]
                out['description_source'] = descs[0].get('DescriptionSourceName')
        except Exception as e:
            out['description_error'] = str(e)[:80]
    return out


def wikipedia(name):
    """Title, first paragraph and page URL from Wikipedia's REST summary endpoint."""
    q = urllib.parse.quote(name.strip().replace(' ', '_'))
    w = _get_json(f'https://en.wikipedia.org/api/rest_v1/page/summary/{q}')
    if w.get('type') == 'disambiguation':
        return {'source': 'Wikipedia', 'title': w.get('title'), 'note': 'disambiguation page - be more specific', 'url': ((w.get('content_urls') or {}).get('desktop') or {}).get('page')}
    return {'source': 'Wikipedia', 'title': w.get('title'), 'extract': (w.get('extract') or '')[:900],
            'url': ((w.get('content_urls') or {}).get('desktop') or {}).get('page')}


def _cache_load():
    try:
        return json.load(open(_CACHE_FILE))
    except Exception:
        return {}


def _cache_save(c):
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        json.dump(c, open(_CACHE_FILE, 'w'), indent=1)
    except Exception:
        pass


def lookup(name, drugs=None, online=True):
    """Everything the lab can say about a substance: library match (mechanism on Maxime's handles, confidence,
    evidence, what is not modelled) + PubChem + Wikipedia text. Online parts are cached in notebook/lookups.json."""
    name = str(name).strip()
    key, how = find(name, drugs)
    out = {'query': name, 'match': None, 'sources': []}
    if key in LIBRARY:
        e = LIBRARY[key]
        out['match'] = dict(name=key, how=how, kind=e['kind'], mechanism=e['mechanism'], confidence=e['confidence'],
                            evidence=e['evidence'], not_modelled=e['not_modelled'],
                            testable=bool(e['handles']), dose_as=key if e['handles'] else None,
                            handles=[f"{h[0]} ({h[1]})" for h in e['handles']])
    elif key:
        out['match'] = dict(name=key, how=how, kind='classic drug', testable=True, dose_as=key, confidence='established (see pharmacology.py)',
                            mechanism=(drugs or {}).get(key, {}).get('mechanism') if isinstance(drugs, dict) else None)
    else:
        out['match'] = None
        out['note'] = ("not in the lab's library. If this is a product, give the active ingredients from its label "
                       "(e.g. 'DEET 25%'); the lab can only dose substances it has a mechanism for.")
    if online:
        cache = _cache_load()
        for fn, label in ((pubchem, 'pubchem'), (wikipedia, 'wikipedia')):
            ck = f'{label}:{_norm(name)}'
            if ck in cache:
                out['sources'].append(cache[ck]); continue
            try:
                r = fn(name); r['fetched'] = time.strftime('%Y-%m-%d')
                cache[ck] = r; out['sources'].append(r)
            except Exception as ex:
                out['sources'].append({'source': label, 'error': f'{type(ex).__name__}: {str(ex)[:80]}'})
        _cache_save(cache)
    return out


def describe(key):
    """One readable paragraph for the notebook / README."""
    e = LIBRARY[key]
    hs = ', '.join(f"{h[0]}" for h in e['handles']) or 'none (not testable as a dose)'
    return (f"{key} [{e['kind']}] - {e['mechanism']}. Handles: {hs}. Confidence: {e['confidence']}. "
            f"Not modelled: {e['not_modelled']}. Evidence: " + '; '.join(e['evidence']))


if __name__ == '__main__':
    import sys
    for q in (sys.argv[1:] or ['DEET', 'Adderall', 'coffee', 'sugar', 'Raid']):
        r = lookup(q, online='--offline' not in sys.argv)
        print(f"\n== {q} ==")
        print(json.dumps(r, indent=1)[:1600])
