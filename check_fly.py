"""
check_fly.py - independent referee for the fly simulation.

Run:  python check_fly.py
It imports fly.py from the same folder and runs four tests. Each prints PASS or FAIL.
Nothing here draws anything; it is pure numbers.

fly.py must expose:
  FlyBrain(n_neurons=139255, seed=None)
      .n                      int
      .W                      scipy.sparse matrix, shape (n, n), W[src, tgt] = weight
      .regions                dict with keys 'sens_L','sens_R','central','motor_L','motor_R'
                              -> (start, stop) index ranges
      .step(sensory)          sensory: np.ndarray, sensory[0]=left smell, sensory[1]=right smell (0..1)
                              returns (motor, spiked); motor = np.array([forward, turn]); spiked = bool array (n,)
      .scramble()             shuffle synaptic targets, keep everything else the same
      .restore()              (optional) reload the original W after a scramble
  FlyBody()
      .x .y .angle            floats; angle in radians, 0 = facing +x, positive turn = counter-clockwise (left)
      .move(motor)            returns (x, y, angle); must keep the fly inside the arena (|x|,|y| <= 10) - walls
  World(seed=None)
      .sugar_x .sugar_y       floats (banana position)
      .get_sensory(x, y, angle) returns (sensory, dist)
"""
import os, sys, time
import numpy as np

try:
    import fly
except ImportError as e:
    print("Could not import fly.py:", e); sys.exit(1)

N_TRIALS, MAX_STEPS, SETTLE = 8, 500, 40
results = {}

def region_rate(brain, spiked_hist, key):
    a, b = brain.regions[key]
    return float(np.mean([s[a:b].mean() for s in spiked_hist]))

def steer_test(brain, world, body, banana_side, steps=100):
    """Hold the body still, banana to one side, measure mean turn after settling."""
    body.x, body.y, body.angle = 0.0, 0.0, 0.0
    world.sugar_x, world.sugar_y = 0.0, 5.0 * banana_side      # +5 = left of a fly facing +x
    turns = []
    for i in range(SETTLE + steps):
        sensory, _ = world.get_sensory(body.x, body.y, body.angle)
        motor, _ = brain.step(sensory)
        if i >= SETTLE:
            turns.append(float(motor[1]))
    return float(np.mean(turns)), float(np.std(turns))

def closed_loop(brain, world_seed_base, label):
    succ, steps_used = 0, []
    for t in range(N_TRIALS):
        rng = np.random.default_rng(world_seed_base + t)
        world = fly.World(seed=world_seed_base + t)
        world.sugar_x, world.sugar_y = rng.uniform(-6, 6), rng.uniform(-6, 6)
        body = fly.FlyBody()
        body.x, body.y, body.angle = 0.0, 0.0, rng.uniform(0, 2 * np.pi)
        found = False
        for s in range(MAX_STEPS):
            sensory, dist = world.get_sensory(body.x, body.y, body.angle)
            if dist < 0.9:
                found = True; steps_used.append(s); break
            motor, _ = brain.step(sensory)
            motor = np.nan_to_num(motor)
            body.move(motor)
        succ += found
        print(f"   [{label}] trial {t+1}/{N_TRIALS}: {'found in %d steps' % steps_used[-1] if found else 'not found'}")
    return succ / N_TRIALS, (float(np.mean(steps_used)) if steps_used else float('nan'))

BRAIN = os.environ.get('BRAIN', 'random')          # BRAIN=flywire python3 check_fly.py  -> real connectome
if BRAIN == 'flywire':
    import flywire_brain
    make_brain = lambda seed: flywire_brain.FlyWireBrain(seed=seed)
else:
    make_brain = lambda seed: fly.FlyBrain(seed=seed)
print(f"Building brain ({BRAIN}) ...")
t0 = time.time()
brain = make_brain(0)
print(f"   built in {time.time()-t0:.1f}s, n={brain.n}, synapses={brain.W.nnz}")
world, body = fly.World(seed=0), fly.FlyBody()

# ---- TEST 1: all regions fire, nothing is silent, nothing is saturated -------------
print("\nTEST 1  firing rates (fly at origin facing +x, banana straight ahead at (5,0))")
body.x, body.y, body.angle = 0.0, 0.0, 0.0
world.sugar_x, world.sugar_y = 5.0, 0.0
hist = []
for i in range(SETTLE + 150):
    sensory, _ = world.get_sensory(body.x, body.y, body.angle)
    _, spiked = brain.step(sensory)
    if i >= SETTLE: hist.append(spiked)
rates = {k: region_rate(brain, hist, k) for k in brain.regions}
for k, v in rates.items(): print(f"   {k:8s} {100*v:6.2f}% of neurons firing per step")
ok1 = 0.003 <= rates['central'] <= 0.20 and rates['motor_L'] > 0.002 and rates['motor_R'] > 0.002
print("   ->", "PASS" if ok1 else "FAIL  (central must be 0.3%-20%, each motor pool > 0.2%)")
results['firing'] = ok1

# ---- TEST 2: banana on the left -> turn left; on the right -> turn right -------------
print("\nTEST 2  steering sign")
mL, sL = steer_test(brain, world, body, +1)
mR, sR = steer_test(brain, world, body, -1)
print(f"   banana LEFT : mean turn {mL:+.3f} (std {sL:.3f})")
print(f"   banana RIGHT: mean turn {mR:+.3f} (std {sR:.3f})")
margin = 0.25 * max(sL, sR, 1e-6)
ok2 = mL > margin and mR < -margin
print("   ->", "PASS" if ok2 else "FAIL  (mean turn must be clearly positive for LEFT, clearly negative for RIGHT)")
results['steering'] = ok2

# ---- TEST 3: lesion - with all synapses removed, steering must disappear ------------
print("\nTEST 3  lesion (W = 0): if the fly still steers, something bypasses the neurons")
W_backup = brain.W.copy()
brain.W = brain.W * 0.0
mL0, sL0 = steer_test(brain, world, body, +1)
mR0, sR0 = steer_test(brain, world, body, -1)
brain.W = W_backup
print(f"   lesioned LEFT : mean turn {mL0:+.3f}   lesioned RIGHT: mean turn {mR0:+.3f}")
ok3 = abs(mL0 - mR0) < 0.5 * abs(mL - mR) if ok2 else False
print("   ->", "PASS" if ok3 else "FAIL  (a lesioned brain steered almost as well as the intact one: a sensory->motor shortcut exists)")
results['lesion'] = ok3

# ---- TEST 4: closed loop, intact vs scrambled connectome -------------------------------
print(f"\nTEST 4  closed loop: {N_TRIALS} bananas, max {MAX_STEPS} steps each")
rate_i, steps_i = closed_loop(brain, 100, "intact")
brain.scramble()
rate_s, steps_s = closed_loop(brain, 100, "scrambled")
print(f"   intact   : {100*rate_i:.0f}% found, mean {steps_i:.0f} steps")
print(f"   scrambled: {100*rate_s:.0f}% found, mean {steps_s:.0f} steps")
ok4 = rate_i >= 0.6 and (rate_i - rate_s) >= 0.2
print("   ->", "PASS" if ok4 else "FAIL  (intact must find >=60% and beat scrambled by >=20 points)")
results['closed_loop'] = ok4

# ---- TEST 5: free run - stays inside the arena, keeps catching bananas ------------
print("\nTEST 5  free run: 1500 steps, banana re-placed on each catch (radius 0.9)")
brain.restore() if hasattr(brain, "restore") else None
if not hasattr(brain, "restore"):
    brain.W = W_backup            # undo the scramble from TEST 4
rng = np.random.default_rng(7)
world, body = fly.World(seed=7), fly.FlyBody()
body.x, body.y, body.angle = 0.0, 0.0, rng.uniform(0, 2 * np.pi)
world.sugar_x, world.sugar_y = rng.uniform(-6, 6), rng.uniform(-6, 6)
caught, max_abs, last_catch = 0, 0.0, 0
for s in range(1500):
    sensory, dist = world.get_sensory(body.x, body.y, body.angle)
    if dist < 0.9:
        caught += 1; last_catch = s
        world.sugar_x, world.sugar_y = rng.uniform(-6, 6), rng.uniform(-6, 6)
    motor, _ = brain.step(sensory)
    body.move(np.nan_to_num(motor))
    max_abs = max(max_abs, abs(body.x), abs(body.y))
print(f"   bananas caught: {caught}   furthest from centre: {max_abs:.1f}   steps since last catch: {1500-last_catch}")
ok5 = max_abs <= 10.0 and caught >= 8 and (1500 - last_catch) < 500
print("   ->", "PASS" if ok5 else "FAIL  (must stay within +-10 [walls], catch >=8, and not stall/orbit for 500 steps)")
results['free_run'] = ok5

print("\n==== SUMMARY ====")
for k, v in results.items(): print(f"   {k:12s} {'PASS' if v else 'FAIL'}")
print("ALL PASS" if all(results.values()) else "NOT YET")
