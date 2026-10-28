"""
check_drugs.py - referee for pharmacology.py. Run: python3 check_drugs.py
Each test prints PASS or FAIL. Nothing here knows what a drug is "supposed" to do to
behaviour; it checks the honesty rules and the mechanisms, then prints what emerged.
"""
import re, sys, inspect
import numpy as np
import fly, pharmacology as ph

results = {}
def verdict(name, ok, note=""):
    results[name] = ok
    print(f"   -> {'PASS' if ok else 'FAIL'}  {note}")

print("Building brain ...")
brain = fly.FlyBrain(seed=1); body = fly.FlyBody(); world = fly.World(seed=1)
pharm = ph.Pharmacology(brain)
rng = np.random.default_rng(1)

def fresh():
    """Same fly, drugs off, back at the origin with the banana ahead."""
    pharm.reset(); brain.restore()
    brain.v[:] = brain.params['v_rest']; brain.refrac[:] = 0
    body.x, body.y, body.angle = 0.0, 0.0, 0.0
    world.sugar_x, world.sugar_y = 5.0, 0.0

def trial(steps=300):
    for _ in range(40):                       # settle
        s, _ = world.get_sensory(body.x, body.y, body.angle); brain.step(s); pharm.step()
    return ph.run_trial(brain, body, world, pharm, steps=steps, rng=rng)

# ---- TEST 1: the table states mechanisms only -------------------------------------------
print("\nTEST 1  honesty: no behavioural words in the drug table")
src = inspect.getsource(ph)
table = src[src.index('DRUGS = {'):src.index('class Pharmacology')]
banned = ['sedat', 'hyperactiv', 'locomot', 'walk', 'turn', 'speed', 'slow', 'faster', 'banana', 'behav']
hits = [w for w in banned if re.search(w, table, re.I)]
verdict('mechanism_only', not hits, f"banned words found: {hits}" if hits else "")

# ---- TEST 2: dose 0 == no drug; washout restores baseline exactly --------------------------
print("\nTEST 2  dose 0 is nothing; washout restores every parameter")
fresh()
before = dict(brain.params)
pharm.dose('picrotoxin', 0.0); pharm.dose('ethanol', 0.0)
same0 = all(abs(brain.params[k] - before[k]) < 1e-12 for k in before)
pharm.dose('picrotoxin', 1.0); pharm.dose('isoflurane', 1.0); pharm.dose('ttx', 0.5)
changed = any(abs(brain.params[k] - before[k]) > 1e-9 for k in before)
pharm.washout()
back = all(abs(brain.params[k] - before[k]) < 1e-12 for k in before)
verdict('dose_zero_washout', same0 and changed and back,
        f"dose0 unchanged={same0}, drugs changed params={changed}, washout restored={back}")

# ---- TEST 3: baseline -----------------------------------------------------------------------
print("\nTEST 3  baseline trial (no drug)")
fresh(); base = trial()
print(f"   firing {100*base['firing_all']:.2f}%  activity {base['activity']:.3f}  synchrony {base['synchrony']:.2f}  bananas {base['bananas']}")
verdict('baseline', 0.01 < base['firing_all'] < 0.15 and base['activity'] > 0.03)

# ---- TEST 4: TTX silences the brain AND the body ---------------------------------------------
print("\nTEST 4  ttx 1.0: sodium channels blocked -> firing should collapse, fly should stop")
fresh(); pharm.dose('ttx', 1.0); r = trial()
print(f"   firing {100*r['firing_all']:.2f}% (baseline {100*base['firing_all']:.2f}%)  activity {r['activity']:.3f} (baseline {base['activity']:.3f})")
verdict('ttx', r['firing_all'] < 0.05 * base['firing_all'] and r['activity'] < 0.6 * base['activity'],
        "firing must drop >95%; body must slow (only forward_base is left)")

# ---- TEST 5: tetanus toxin: brain lit, body frozen ----------------------------------------------
print("\nTEST 5  tetanus_toxin 1.0: motor terminals silenced -> brain keeps firing, body stops")
fresh(); pharm.dose('tetanus_toxin', 1.0); r = trial()
print(f"   firing {100*r['firing_all']:.2f}% (baseline {100*base['firing_all']:.2f}%)  activity {r['activity']:.3f} (baseline {base['activity']:.3f})")
verdict('tetanus', r['firing_all'] > 0.5 * base['firing_all'] and r['activity'] < 0.1 * base['activity'],
        "firing within 50% of baseline AND activity <10% of baseline")

# ---- TEST 6: picrotoxin: inhibition removed -> what emerges? ----------------------------------------
print("\nTEST 6  picrotoxin 1.0: inhibition removed -> is the network still balanced?")
fresh(); pharm.dose('picrotoxin', 1.0); r = trial()
print(f"   firing {100*r['firing_all']:.2f}% (baseline {100*base['firing_all']:.2f}%)  synchrony {r['synchrony']:.2f} (baseline {base['synchrony']:.2f})  bananas {r['bananas']} (baseline {base['bananas']})")
verdict('picrotoxin', r['firing_all'] > 1.1 * base['firing_all'],
        "mechanism check only: firing must rise when inhibition is removed. Burst synchrony is REPORTED, not required: "
        "see validation.md (this model is drive-dominated and does not seize - a documented negative result)")

# ---- TEST 7: anaesthesia is reversible ------------------------------------------------------------
print("\nTEST 7  isoflurane 1.0 then washout by decay: does the fly come back?")
fresh(); pharm.dose('isoflurane', 1.0); under = trial(200)
for _ in range(900):                       # ~7 half-lives of decay, fly keeps living
    s, _ = world.get_sensory(body.x, body.y, body.angle); m, _ = brain.step(s); pharm.step(); body.move(m)
after = trial(200)
print(f"   under: firing {100*under['firing_all']:.2f}%  activity {under['activity']:.3f}   after decay: firing {100*after['firing_all']:.2f}%  activity {after['activity']:.3f}   conc now {pharm.state()}")
verdict('anaesthesia_reversible', under['firing_all'] < 0.7 * base['firing_all'] and after['firing_all'] > 0.7 * base['firing_all'],
        "firing must fall under the drug and recover after decay")

print("\n==== SUMMARY ====")
for k, v in results.items():
    print(f"   {k:22s} {'PASS' if v else 'FAIL'}")
print("ALL PASS" if all(results.values()) else "NOT YET")
sys.exit(0 if all(results.values()) else 1)
