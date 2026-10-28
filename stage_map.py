"""
stage_map.py - where does the left/right smell signal die in the real FlyWire brain?

Maxime (FlyWire v783 wiring) is held still with the banana on her LEFT, then on her RIGHT.
For each stage of the known olfactory-to-motor path we measure, per side, the mean firing rate
of the neurons of that stage, and report a lateralisation index

    LI = (rate on the banana side - rate on the far side) / (rate on the banana side + rate on the far side)

averaged over the two banana positions. LI ~ 1 means the stage "knows" which side the banana is on;
LI ~ 0 means the signal is gone. This is the diagnostic behind validation.md P7.

    python3 stage_map.py            # ~2 minutes
"""
import re as _re, sys, time
import numpy as np
import flywire_brain as F
import fly

STAGES = [                                   # (label, test on primary_type / super_class)
    ('banana ORNs',          lambda pt, sc: np.array([any(t.startswith(g) for g in F.BANANA_GLOMERULI) for t in pt])),
    ('banana-glomerulus PNs (DM1/DM2/DM3/DM4/VM2/VA2)', lambda pt, sc: np.array([any(t.startswith(g.replace('ORN_', '') + '_') and t.endswith('PN') for g in F.BANANA_GLOMERULI) for t in pt])),
    ('all uniglomerular olfactory PNs', lambda pt, sc: np.array([bool(_re.match(r'^(D|V|VA|VC|VL|VM|DA|DC|DL|DM|DP)[A-Z]*\d+[a-z]?_(ad|l|il|v|lv)PN$', t)) for t in pt])),
    ('lateral horn (LH*)',   lambda pt, sc: np.array([t.startswith('LH') for t in pt])),
    ('mushroom body KCs',    lambda pt, sc: np.array([t.startswith('KC') for t in pt])),
    ('MBONs',                lambda pt, sc: np.array([t.startswith('MBON') for t in pt])),
    ('LAL neurons',          lambda pt, sc: np.array([t.startswith('LAL') for t in pt])),
    ('central complex (EPG/PEN/PFL/FC/hDelta/vDelta/ER)', lambda pt, sc: np.array([t.startswith(('EPG', 'PEN', 'PFL', 'FC', 'hDelta', 'vDelta', 'ER', 'Delta7', 'FR')) for t in pt])),
    ('descending neurons',   lambda pt, sc: sc == 'descending'),
]


def rates_for(brain, world, body, steps=300, settle=60):
    acc = np.zeros(brain.n, dtype=np.float64)
    for i in range(settle + steps):
        s, _ = world.get_sensory(body.x, body.y, body.angle)
        _, spiked = brain.step(s)
        if i >= settle:
            acc += spiked
    return acc / steps


def main(seed=1, steps=300):
    t0 = time.time()
    brain = F.FlyWireBrain(seed=seed)
    pt = np.asarray(brain.meta['primary_type']).astype(str)
    sc = np.asarray(brain.meta['super_class']).astype(str)
    side = np.asarray(brain.meta['side']).astype(str)
    body = fly.FlyBody(); body.x = body.y = 0.0; body.angle = 0.0
    world = fly.World(seed=seed)
    out = {}
    for where, (sx, sy) in (('left', (0.0, 3.0)), ('right', (0.0, -3.0))):     # heading +x; +y is her left
        world.sugar_x, world.sugar_y = sx, sy
        brain.restore(); brain.v[:] = brain.params['v_rest']; brain.refrac[:] = 0
        out[where] = rates_for(brain, world, body, steps=steps)
        print(f"banana on the {where}: done ({time.time()-t0:.0f}s)")
    print(f"\n{'stage':52s} {'n L/R':>11s} {'rate L':>8s} {'rate R':>8s}  (banana left)   LI")
    for label, test in STAGES:
        m = test(pt, sc)
        mL, mR = m & (side == 'left'), m & (side == 'right')
        if mL.sum() == 0 or mR.sum() == 0:
            print(f"{label:52s} {mL.sum():5d}/{mR.sum():<5d}  (no neurons on one side)"); continue
        lis = []
        for where, near, far in (('left', mL, mR), ('right', mR, mL)):
            rn, rf = out[where][near].mean(), out[where][far].mean()
            lis.append((rn - rf) / (rn + rf) if rn + rf > 0 else 0.0)
        rl, rr = out['left'][mL].mean(), out['left'][mR].mean()
        print(f"{label:52s} {mL.sum():5d}/{mR.sum():<5d} {100*rl:7.2f}% {100*rr:7.2f}%               {np.mean(lis):+.3f}")
    print("\nLI: +1 = the stage fires only on the banana's side, 0 = no left/right information left.")


if __name__ == '__main__':
    main(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 300)
