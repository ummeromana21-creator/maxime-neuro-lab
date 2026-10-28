"""
experiments.py  -  what an experiment IS in this lab: the card, the steps it generates,
the verdict rule, and the notebook that remembers every experiment ever run on Maxime.

A card (JSON):
{
  "title":       "Does caffeine make Maxime walk more?",
  "hypothesis":  "Caffeine increases Maxime's walking activity.",
  "description": "Why we think so and what we will do, in two or three sentences.",
  "control":     {"name": "sober",        "setup": []},
  "treatments": [{"name": "caffeine 0.5", "setup": [{"cmd": "dose", "drug": "caffeine", "amount": 0.5}]}],
  "measure":     "activity",        # the primary measurement, see MEASURES
  "prediction":  "higher",          # treatment vs control: "higher" | "lower" | "different"
  "repeats":     5,                 # how many independent runs per condition (minimum 3)
  "steps":       400,               # length of each run
  "variables":   {"changed": "...", "measured": [...], "held": [...]},
  "constraints": ["..."],
  "author":      "Muse" | "human" | "script"
}
Only the lab's generic verbs may appear in a setup: restore, dose, washout, set_param, scramble.
The lab itself adds "restore" at the start of EVERY run, so no run inherits anything from the
previous one. The "steps" shown on the card are generated from the card, so what is written is
exactly what runs.

The verdict rule (one rule, plain words, no statistics vocabulary):
  * a run's value is the primary measurement of one run; the control and each treatment get
    `repeats` runs each
  * gap    = treatment average - control average
  * wobble = the typical run-to-run spread (root of the mean of the two variances)
  * hits   = how many treatment runs fall on the predicted side of the control average
  SUPPORTED      if hits >= 80% of runs AND |gap| >= wobble AND the gap's sign matches the prediction
  NOT SUPPORTED  if hits <= half the runs, OR the gap is at least one wobble in the WRONG direction
  CAN'T TELL YET otherwise (the honest answer is: run more repeats)
"""
import json, math, os, re, time

MEASURES = {                      # key in TrialMeter.result() -> plain English
    'activity':        'distance walked per step',
    'bananas':         'bananas found',
    'firing_all':      'share of all neurons firing per step',
    'firing_peak':     'peak share of neurons firing in one step',
    'synchrony':       'fraction of steps with mass firing (>15% at once)',
    'tortuosity':      'path length divided by displacement (wandering)',
    'mean_abs_turn':   'average turning strength',
    'firing.sens_L':   'firing in the left nose', 'firing.sens_R': 'firing in the right nose',
    'firing.central':  'firing in the central brain',
    'firing.motor_L':  'firing in the left leg command neurons', 'firing.motor_R': 'firing in the right leg command neurons',
}
ALLOWED_SETUP = ('restore', 'dose', 'washout', 'set_param', 'scramble')
NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebook')


def get_measure(metrics, key):
    if key.startswith('firing.'):
        return float(metrics['firing'][key.split('.', 1)[1]])
    v = metrics.get(key)
    return None if v is None else float(v)


def validate(card, drugs, params):
    """Return (clean_card, error). Fills defaults; rejects anything the lab cannot run."""
    c = dict(card or {})
    err = None
    c['title'] = str(c.get('title', '')).strip()[:120] or 'Untitled experiment'
    c['hypothesis'] = str(c.get('hypothesis', '')).strip()[:300]
    c['description'] = str(c.get('description', '')).strip()[:1000]
    if not c['hypothesis']:
        return None, "a hypothesis is required (one sentence you can be wrong about)"
    ctrl = c.get('control') or {'name': 'control', 'setup': []}
    treats = c.get('treatments') or []
    if not isinstance(treats, list) or not treats:
        return None, "at least one treatment condition is required"
    def clean_cond(cond, default_name):
        name = str(cond.get('name', default_name))[:60]
        setup = []
        for a in cond.get('setup', []) or []:
            cmd = a.get('cmd')
            if cmd not in ALLOWED_SETUP:
                raise ValueError(f"setup verb '{cmd}' is not allowed; allowed: {ALLOWED_SETUP}")
            if cmd == 'dose':
                d = a.get('drug')
                if d not in drugs:                                   # accept 'DEET', 'Adderall', 'tetanus toxin'...
                    try:
                        import substances as _S
                        key, _how = _S.find(str(d), drugs)
                    except ImportError:
                        key = None
                    if key is not None and key not in drugs:
                        why = getattr(_S, 'LIBRARY', {}).get(key, {}).get('not_modelled', '')
                        raise ValueError(f"'{d}' is {key}, which the lab cannot dose yet: {why}")
                    if key is None:
                        raise ValueError(f"unknown drug '{d}'; known: {sorted(drugs)}")
                    d = key
                amt = float(a.get('amount', 0.5))
                if not (0.0 <= amt <= 2.0):
                    raise ValueError(f"dose amount {amt} out of range 0-2")
                a = {'cmd': 'dose', 'drug': d, 'amount': amt}
                if not 0 <= a['amount'] <= 2: raise ValueError("dose amount must be 0..2")
            elif cmd == 'set_param':
                if a.get('key') not in params:
                    raise ValueError(f"unknown parameter '{a.get('key')}'; known: {sorted(params)}")
                a = {'cmd': 'set_param', 'key': a['key'], 'value': float(a['value'])}
            else:
                a = {'cmd': cmd}
            setup.append(a)
        return {'name': name, 'setup': setup}
    try:
        c['control'] = clean_cond(ctrl, 'control')
        c['treatments'] = [clean_cond(t, f'treatment {i+1}') for i, t in enumerate(treats)][:6]
    except ValueError as e:
        return None, str(e)
    if c.get('measure') not in MEASURES:
        return None, f"measure must be one of {list(MEASURES)}"
    if c.get('prediction') not in ('higher', 'lower', 'different'):
        return None, "prediction must be 'higher', 'lower' or 'different'"
    c['repeats'] = int(max(3, min(10, int(c.get('repeats', 5)))))
    c['steps'] = int(max(100, min(2000, int(c.get('steps', 400)))))
    v = c.get('variables') or {}
    c['variables'] = {'changed': str(v.get('changed', '')) or ', '.join(t['name'] for t in c['treatments']) + ' vs ' + c['control']['name'],
                      'measured': list(v.get('measured') or [c['measure'], 'bananas', 'firing_all']),
                      'held': list(v.get('held') or ['same arena', f"same run length ({c['steps']} steps)", 'Maxime reset before every run'])}
    c['constraints'] = list(c.get('constraints') or ['reset before every run', 'no human intervention during a run',
                                                       f"{c['repeats']} independent runs per condition"])
    c['author'] = str(c.get('author', 'human'))[:40]
    return c, None


def steps_text(card):
    """The procedure, generated from the card so it is exactly what runs."""
    lines = []
    conds = [card['control']] + card['treatments']
    for cond in conds:
        setup = ', '.join(describe_action(a) for a in cond['setup']) or 'nothing'
        lines.append(f"Condition \"{cond['name']}\": reset Maxime (restore), then {setup}; let her settle for 30 steps; "
                     f"run {card['steps']} steps and record {MEASURES[card['measure']]} (and the other measurements). "
                     f"Repeat {card['repeats']} times.")
    lines.append(f"Compare each treatment with \"{card['control']['name']}\" using the lab's verdict rule "
                 f"(predicted: {card['prediction']}).")
    return lines


def describe_action(a):
    if a['cmd'] == 'dose': return f"give {a['drug'].replace('_', ' ')} {a['amount']:g}"
    if a['cmd'] == 'set_param': return f"set {a['key']} = {a['value']:g}"
    if a['cmd'] == 'scramble': return "scramble her synapses"
    if a['cmd'] == 'washout': return "wash out all drugs"
    return a['cmd']


def analyse(card, results):
    """results: {condition name: [metrics, ...]} -> (analysis_text, verdict, per-treatment details)."""
    m, pred = card['measure'], card['prediction']
    label = MEASURES[m]
    ctrl = card['control']['name']
    cvals = [get_measure(r, m) for r in results[ctrl]]
    n = len(cvals)
    cmean = sum(cvals) / n
    cvar = sum((x - cmean) ** 2 for x in cvals) / max(1, n - 1)
    out, lines, verdicts = [], [], []
    lines.append(f"{ctrl.capitalize()}: {label} averaged {fmt(cmean)} over {n} runs (from {fmt(min(cvals))} to {fmt(max(cvals))}).")
    for t in card['treatments']:
        tvals = [get_measure(r, m) for r in results[t['name']]]
        tmean = sum(tvals) / len(tvals)
        tvar = sum((x - tmean) ** 2 for x in tvals) / max(1, len(tvals) - 1)
        gap = tmean - cmean
        wobble = math.sqrt(0.5 * (cvar + tvar)) + 1e-12
        if pred == 'higher':   hits = sum(x > cmean for x in tvals)
        elif pred == 'lower':  hits = sum(x < cmean for x in tvals)
        else:                  hits = max(sum(x > cmean for x in tvals), sum(x < cmean for x in tvals))
        want = {'higher': 1, 'lower': -1, 'different': (1 if gap >= 0 else -1)}[pred]
        sign_ok = (gap * want) > 0
        need = math.ceil(0.8 * len(tvals))
        if hits >= need and abs(gap) >= wobble and sign_ok:
            v = 'SUPPORTED'
        elif hits <= len(tvals) / 2 or (abs(gap) >= wobble and not sign_ok):
            v = 'NOT SUPPORTED'
        else:
            v = "CAN'T TELL YET"
        pct = (100 * gap / cmean) if cmean else float('nan')
        direction = 'more' if gap > 0 else 'less'
        ratio = abs(gap) / wobble
        lines.append(f"{t['name'].capitalize()}: {label} averaged {fmt(tmean)} over {len(tvals)} runs "
                     f"(from {fmt(min(tvals))} to {fmt(max(tvals))}) — {abs(pct):.0f}% {direction} than {ctrl}. "
                     f"{hits} of {len(tvals)} runs were on the predicted side of the {ctrl} average; "
                     f"the gap ({fmt(abs(gap))}) is {ratio:.1f}× the run-to-run wobble ({fmt(wobble)}).")
        why = {'SUPPORTED': "the direction held in almost every run and the gap is bigger than the wobble",
               'NOT SUPPORTED': ("the gap points the wrong way" if (abs(gap) >= wobble and not sign_ok) else "the runs do not agree on a direction"),
               "CAN'T TELL YET": "the gap is smaller than the run-to-run wobble, or too few runs agreed — more repeats would settle it"}[v]
        lines.append(f"Verdict for {t['name']}: {v} — {why}.")
        verdicts.append(v)
        out.append({'treatment': t['name'], 'control_mean': cmean, 'treatment_mean': tmean, 'gap': gap, 'wobble': wobble,
                    'hits': hits, 'of': len(tvals), 'verdict': v, 'values': tvals, 'control_values': cvals})
    overall = ('SUPPORTED' if all(v == 'SUPPORTED' for v in verdicts) else
               'NOT SUPPORTED' if all(v == 'NOT SUPPORTED' for v in verdicts) else
               'MIXED' if len(set(verdicts)) > 1 and "CAN'T TELL YET" not in verdicts else "CAN'T TELL YET")
    return "\n".join(lines), overall, out


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return 'n/a'
    if abs(x) >= 100: return f"{x:.0f}"
    if abs(x) >= 1: return f"{x:.2f}"
    return f"{x:.3f}"


# ------------------------------------------------------------------ notebook
class Notebook:
    def __init__(self, path=NOTEBOOK_DIR):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def new_id(self, title):
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:40] or 'experiment'
        return time.strftime('%Y%m%d_%H%M%S') + '_' + slug

    def save(self, record):
        with open(os.path.join(self.path, record['id'] + '.json'), 'w') as f:
            json.dump(record, f, indent=1)
        self.write_index()

    def load(self, exp_id):
        p = os.path.join(self.path, exp_id + '.json')
        if not os.path.exists(p): return None
        with open(p) as f: return json.load(f)

    def list(self):
        out = []
        for fn in sorted(os.listdir(self.path)):
            if fn.endswith('.json'):
                try:
                    with open(os.path.join(self.path, fn)) as f: r = json.load(f)
                    out.append({'id': r['id'], 'title': r['card']['title'], 'hypothesis': r['card']['hypothesis'],
                                'verdict': r.get('verdict'), 'date': r.get('date'), 'subject': r.get('subject'),
                                'author': r['card'].get('author'), 'fly_age': r.get('fly_age'), 'rerun_of': r.get('rerun_of')})
                except Exception:
                    pass
        return out

    def write_index(self):
        rows = self.list()
        with open(os.path.join(self.path, 'notebook.md'), 'w') as f:
            f.write("# Maxime's lab notebook\n\nEvery experiment ever run on her, newest last. Each has its own JSON file beside this index.\n\n")
            f.write("| date | fly age (steps) | title | hypothesis | verdict | by | subject |\n|---|---|---|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r['date']} | {r.get('fly_age','')} | {r['title']} | {r['hypothesis']} | **{r['verdict']}** | {r['author']} | {r['subject']} |\n")

    def markdown(self, record):
        c = record['card']
        s = [f"# {c['title']}", "", f"*Hypothesis:* {c['hypothesis']}", "", f"*Description:* {c['description']}", "",
             f"*Subject:* {record['subject']}  ·  *Date:* {record['date']}  ·  *Fly age:* {record.get('fly_age')} steps  ·  *By:* {c['author']}", "",
             "## Steps", ""] + [f"{i+1}. {l}" for i, l in enumerate(steps_text(c))] + ["",
             "## Variables", "", f"- Changed: {c['variables']['changed']}", f"- Measured: {', '.join(MEASURES.get(k, k) for k in c['variables']['measured'])}",
             f"- Held constant: {', '.join(c['variables']['held'])}", "", "## Constraints", ""] + [f"- {x}" for x in c['constraints']] + ["",
             "## Measurements", "", f"Primary: {MEASURES[c['measure']]}", "", "| condition | " + " | ".join(f"run {i+1}" for i in range(c['repeats'])) + " | average |", "|---|" + "---|" * (c['repeats'] + 1)]
        for name, runs in record['results'].items():
            vals = [get_measure(r, c['measure']) for r in runs]
            s.append(f"| {name} | " + " | ".join(fmt(v) for v in vals) + f" | **{fmt(sum(vals)/len(vals))}** |")
        s += ["", "## Analysis", "", record['analysis'], "", f"## Verdict: {record['verdict']}", ""]
        return "\n".join(s)
