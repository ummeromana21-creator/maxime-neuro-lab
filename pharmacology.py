"""
pharmacology.py  -  drugs as hypotheses about mechanism, applied to the fly brain.

How it works
    A drug is a concentration (0..2, in "dose units") that decays with a half-life.
    Each drug changes one or more entries of brain.params through a Hill curve,
    E(c) = Emax * c^n / (EC50^n + c^n), which is the shape every real dose-response has.
    Nothing else is touched: not position, not heading, not spikes.

Honesty rules (check_drugs.py enforces them)
    1. The DRUGS table may only state MECHANISMS: which synapses, channels or drives change.
       Behavioural words (sedate, hyperactive, slower, turn...) are forbidden in the table;
       behaviour must EMERGE from the perturbed network or the model is wrong.
    2. Dose 0 must be indistinguishable from no drug.
    3. washout() returns every parameter to its baseline.

Scope
    This is a phenomenological model: it encodes what each drug is known to do to a network,
    not molecular binding. The fly is a simulated insect. Nothing here is about people.
"""
import numpy as np


def hill(c, ec50, n=2.0, emax=1.0):
    """Hill equation: fraction of maximal effect at concentration c."""
    c = max(0.0, float(c))
    if c == 0.0:
        return 0.0
    return emax * c ** n / (ec50 ** n + c ** n)


# ---------------------------------------------------------------------------------------
# The drug table.  Each entry: mechanism text, half-life (steps), and a list of effects.
# An effect is (param, kind, function_of_concentration):
#     kind 'mult' -> param = baseline * f(c)      (several drugs multiply together)
#     kind 'set'  -> param = max(current, f(c))   (used for ttx_frac)
#     kind 'add'  -> param = current + f(c)         (used for odour-independent sensory drive)
# ---------------------------------------------------------------------------------------
DRUGS = {
    'picrotoxin': dict(
        mechanism="GABA-A receptor antagonist: blocks the chloride channel, removing inhibitory transmission",
        half_life=400,
        effects=[('g_inh', 'mult', lambda c: 1.0 - hill(c, ec50=0.35, n=2))],
    ),
    'ttx': dict(
        mechanism="tetrodotoxin: blocks voltage-gated sodium channels, so affected neurons cannot fire",
        half_life=800,
        effects=[('ttx_frac', 'set', lambda c: hill(c, ec50=0.3, n=3))],
    ),
    'tetanus_toxin': dict(
        mechanism="cleaves synaptobrevin at motor terminals: motor neurons fire but release nothing onto muscle",
        half_life=2000,
        effects=[('motor_out', 'mult', lambda c: 1.0 - hill(c, ec50=0.3, n=3))],
    ),
    'caffeine': dict(
        mechanism="adenosine receptor antagonist: removes a tonic brake on background excitatory drive",
        half_life=600,
        effects=[('I_bg', 'mult', lambda c: 1.0 + 0.25 * hill(c, ec50=0.4, n=1.5))],
    ),
    'isoflurane': dict(
        mechanism="volatile anaesthetic: potentiates GABA-A inhibition and reduces background excitatory drive",
        half_life=300,      # was 120; raised so a 400-step trial samples a meaningful exposure (see validation.md)
        effects=[('g_inh', 'mult', lambda c: 1.0 + 1.5 * hill(c, ec50=0.4, n=2)),
                 ('I_bg', 'mult', lambda c: 1.0 - 0.35 * hill(c, ec50=0.4, n=2))],
    ),
    'ethanol': dict(
        # single-mechanism hypothesis on purpose: see validation.md - real flies show a biphasic
        # response whose low-dose phase runs through dopamine/octopamine, which this brain lacks.
        mechanism="potentiates GABA-A inhibition, depresses excitatory (NMDA-like) transmission, adds membrane noise",
        half_life=600,
        effects=[('g_inh', 'mult', lambda c: 1.0 + 0.8 * hill(c, ec50=0.5, n=2)),
                 ('g_exc', 'mult', lambda c: 1.0 - 0.3 * hill(c, ec50=0.5, n=2)),
                 ('noise', 'mult', lambda c: 1.0 + 0.5 * hill(c, ec50=0.5, n=2))],
    ),
    'nicotine': dict(
        mechanism="nicotinic acetylcholine receptor agonist: activates excitatory synapses, then desensitises them at higher concentration",
        half_life=300,
        effects=[('g_exc', 'mult', lambda c: 1.0 + 0.4 * hill(c, ec50=0.2, n=2) - 0.7 * hill(c, ec50=0.7, n=4))],
    ),
}


class Pharmacology:
    """Holds the drug concentrations in the fly and applies their combined effect each step."""

    def __init__(self, brain):
        self.brain = brain
        self.base = dict(brain._defaults)        # baseline params (set_param edits this layer)
        self.conc = {}                           # drug name -> concentration
        self.apply()

    # ---- experimenter's verbs ----------------------------------------------------------
    def dose(self, name, amount):
        if name not in DRUGS:
            raise KeyError(f"unknown drug '{name}'; known: {sorted(DRUGS)}")
        self.conc[name] = min(2.0, self.conc.get(name, 0.0) + float(amount))
        self.apply()
        return self.conc[name]

    def washout(self):
        self.conc.clear()
        self.apply()

    def set_param(self, key, value):
        """Change a baseline parameter directly (a lesion-style manipulation, not a drug)."""
        if key not in self.base:
            raise KeyError(f"unknown param '{key}'; known: {sorted(self.base)}")
        self.base[key] = float(value)
        self.apply()

    def reset(self):
        """Baseline parameters back to the brain's defaults and no drugs on board."""
        self.base = dict(self.brain._defaults)
        self.conc.clear()
        self.apply()

    # ---- per-step dynamics --------------------------------------------------------------
    def step(self):
        """Pharmacokinetics: every concentration decays with its half-life."""
        if not self.conc:
            return
        gone = []
        for name, c in self.conc.items():
            c *= 0.5 ** (1.0 / DRUGS[name]['half_life'])
            if c < 1e-3:
                gone.append(name)
            else:
                self.conc[name] = c
        for name in gone:
            del self.conc[name]
        self.apply()

    def apply(self):
        """Recompute brain.params from the baseline and every drug on board."""
        p = dict(self.base)
        for name, c in self.conc.items():
            for key, kind, f in DRUGS[name]['effects']:
                val = f(c)
                if kind == 'mult':
                    p[key] = p[key] * val
                elif kind == 'set':
                    p[key] = max(p[key], val)
                elif kind == 'add':
                    p[key] = p[key] + val
        self.brain.params.update(p)

    def state(self):
        return {k: round(v, 4) for k, v in self.conc.items()}


# ---------------------------------------------------------------------------------------
# Generic measurements an experiment can ask for. No drug-specific code lives here.
# ---------------------------------------------------------------------------------------
class TrialMeter:
    """Accumulates the generic metrics of a trial, one step at a time.
    Used by run_trial() below and by server.py, so both report identical numbers."""

    def __init__(self, brain, body):
        self.R = brain.regions
        self.rates = {k: 0.0 for k in self.R}
        self.total_rate = self.path = self.turn_abs = 0.0
        self.sync_hits = self.caught = self.n = 0
        self.x0, self.y0 = body.x, body.y
        self.peak_rate = 0.0

    def update(self, spiked, motor, body, prev_xy, caught_now):
        self.n += 1
        self.caught += int(caught_now)
        self.path += float(np.hypot(body.x - prev_xy[0], body.y - prev_xy[1]))
        self.turn_abs += abs(float(motor[1]))
        frac = float(spiked.mean())
        self.total_rate += frac
        self.peak_rate = max(self.peak_rate, frac)
        self.sync_hits += int(frac > 0.15)
        for k, (a, b) in self.R.items():
            self.rates[k] += float(spiked[a:b].mean())
        self.last_xy = (body.x, body.y)

    def result(self, brain, pharm=None):
        n = float(max(1, self.n))
        net = float(np.hypot(self.last_xy[0] - self.x0, self.last_xy[1] - self.y0)) if self.n else 0.0
        return {
            'steps': self.n,
            'bananas': self.caught,
            'activity': self.path / n,                          # distance per step
            'tortuosity': (self.path / net) if net > 0.5 else None,
            'mean_abs_turn': self.turn_abs / n,
            'firing_all': self.total_rate / n,                  # fraction of neurons firing per step
            'firing_peak': self.peak_rate,
            'firing': {k: v / n for k, v in self.rates.items()},
            'synchrony': self.sync_hits / n,                    # fraction of steps with >15% firing at once
            'drugs': pharm.state() if pharm is not None else {},
            'scrambled': bool(getattr(brain, 'scrambled', False)),
        }


def run_trial(brain, body, world, pharm=None, steps=400, catch_radius=0.9, rng=None):
    """Run `steps` steps and return generic metrics. The experiment is whatever state
    (drugs, params, scramble) the caller put the fly in before calling this."""
    meter = TrialMeter(brain, body)
    for _ in range(steps):
        sensory, dist = world.get_sensory(body.x, body.y, body.angle)
        caught_now = dist < catch_radius
        if caught_now:
            world.randomize_sugar()
        motor, spiked = brain.step(sensory)
        if pharm is not None:
            pharm.step()
        prev = (body.x, body.y)
        body.move(motor)
        meter.update(spiked, motor, body, prev, caught_now)
    return meter.result(brain, pharm)


# ---------------------------------------------------------------------------------------
# The substance library (repellents, insecticides, stimulants...) lives in substances.py with its
# evidence and confidence labels; entries that have a mechanism on Maxime's handles are drugs too.
# ---------------------------------------------------------------------------------------
try:
    import substances as _subst
    DRUGS.update(_subst.drug_entries())
except ImportError:
    pass
