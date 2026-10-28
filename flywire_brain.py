"""
flywire_brain.py  -  Maxime's brain built from the REAL FlyWire v783 connectome.

Data: FlyWire, female adult fly brain (FAFB) v783, downloaded from https://codex.flywire.ai
(Dorkenwald et al. 2024 Nature; Schlegel et al. 2024 Nature; transmitter predictions Eckstein et al. 2024).
Files used (put them in ./flywire/):
    neurons.csv.gz                 root_id, nt_type (predicted transmitter), ...
    classification.csv.gz          root_id, super_class, class, side, ...
    consolidated_cell_types.csv.gz root_id, primary_type (e.g. ORN_DM2, DNa02)
    connections_princeton.csv.gz   pre_root_id, post_root_id, neuropil, syn_count, nt_type   (>=5 synapses)

What is real here: every neuron, every connection and its synapse count, which neuron is on which
side, which neurons are olfactory receptor neurons of which glomerulus, which are descending neurons.
What is an assumption (and says so): transmitter -> sign (ACH excites; GABA and glutamate inhibit;
DA/5-HT/OCT and unknown treated as excitatory, following Shiu et al. 2024), the per-synapse strength,
membrane dynamics, background drive, which glomeruli "banana" means, and that descending-neuron
firing by side is what the legs read. The nerve cord, muscles and sensory transduction are not modelled.

Usage:
    python3 flywire_brain.py             # builds and caches flywire/maxime_v783.npz, prints a summary
    brain = flywire_brain.FlyWireBrain() # in code; same API as fly.FlyBrain
"""
import os, sys, time
import numpy as np
from scipy import sparse
import fly

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flywire')
CACHE = os.path.join(DATA_DIR, 'maxime_v783.npz')

# Glomeruli whose receptor neurons respond to fruit esters (banana-type odours): Hallem & Carlson 2006.
BANANA_GLOMERULI = ['ORN_DM2', 'ORN_DM1', 'ORN_DM4', 'ORN_VM2', 'ORN_DM3', 'ORN_VA2']
# Transmitter -> sign of every synapse that neuron makes. Convention of Shiu et al. 2024.
NT_SIGN = {'ACH': 1.0, 'GABA': -1.0, 'GLUT': -1.0, 'DA': 1.0, 'SER': 1.0, 'OCT': 1.0, '': 1.0}
# Same-side vs other-side release strength of olfactory receptor neuron synapses (Gaudry et al. 2013).
# 1.0 = no asymmetry (raw connectome). 1.3 means same-side x1.3, other-side x0.7 (approximate).
ORN_IPSI_BIAS = 1.3


def build(data_dir=DATA_DIR, cache=CACHE, verbose=True):
    import pandas as pd
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    say("Reading FlyWire v783 tables ...")
    neu = pd.read_csv(os.path.join(data_dir, 'neurons.csv.gz'), usecols=['root_id', 'nt_type', 'nt_type_score'], dtype={'root_id': np.int64})
    cls = pd.read_csv(os.path.join(data_dir, 'classification.csv.gz'), usecols=['root_id', 'super_class', 'class', 'side'], dtype={'root_id': np.int64})
    typ = pd.read_csv(os.path.join(data_dir, 'consolidated_cell_types.csv.gz'), usecols=['root_id', 'primary_type'], dtype={'root_id': np.int64})
    df = neu.merge(cls, on='root_id', how='left').merge(typ, on='root_id', how='left')
    for c in ('nt_type', 'super_class', 'class', 'side', 'primary_type'):
        df[c] = df[c].fillna('')
    n = len(df)
    say(f"  {n:,} neurons")

    # ---- regions: nose L/R, central, legs L/R -------------------------------------------
    orn = (df['class'] == 'olfactory') & df['primary_type'].isin(BANANA_GLOMERULI)
    dn = df['super_class'] == 'descending'
    sens_L = df.index[orn & (df['side'] == 'left')].to_numpy()
    sens_R = df.index[orn & (df['side'] == 'right')].to_numpy()
    motor_L = df.index[dn & (df['side'] == 'left')].to_numpy()
    motor_R = df.index[dn & (df['side'] == 'right')].to_numpy()
    used = np.zeros(n, bool); used[np.concatenate([sens_L, sens_R, motor_L, motor_R])] = True
    central = np.flatnonzero(~used)
    order = np.concatenate([sens_L, sens_R, central, motor_L, motor_R])
    assert order.size == n
    bounds, regions, k = {}, {}, 0
    for name, idx in (('sens_L', sens_L), ('sens_R', sens_R), ('central', central), ('motor_L', motor_L), ('motor_R', motor_R)):
        regions[name] = (k, k + idx.size); k += idx.size
    say("  regions: " + ", ".join(f"{k}={b-a:,}" for k, (a, b) in regions.items()))

    new_index = np.empty(n, dtype=np.int64); new_index[order] = np.arange(n)   # old row -> new position
    root_ids = df['root_id'].to_numpy()[order]
    id_to_new = pd.Series(np.arange(n), index=root_ids)
    sign_by_new = np.array([NT_SIGN.get(t, 1.0) for t in df['nt_type'].to_numpy()[order]], dtype=np.float32)

    # ---- connections ---------------------------------------------------------------------
    say("  reading connections (5.3M rows) ...")
    con = pd.read_csv(os.path.join(data_dir, 'connections_princeton.csv.gz'), usecols=['pre_root_id', 'post_root_id', 'neuropil', 'syn_count'], dtype={'pre_root_id': np.int64, 'post_root_id': np.int64, 'syn_count': np.int32})
    # ASSUMPTION (Gaudry et al. 2013 Nature): an olfactory receptor neuron releases more transmitter in
    # its own antennal lobe than in the opposite one. The connectome records the synapses on both sides
    # but not their strength; scale same-side synapses up and other-side synapses down. ORN_IPSI_BIAS=1
    # switches this off. The neuropil column tells which side each synapse is on (e.g. AL_L / AL_R).
    orn_side = dict(zip(df.loc[df['class'] == 'olfactory', 'root_id'], df.loc[df['class'] == 'olfactory', 'side']))
    is_orn = con['pre_root_id'].map(orn_side)
    syn_side = con['neuropil'].str.extract(r'_(L|R)$')[0].map({'L': 'left', 'R': 'right'})
    ipsi = (is_orn == syn_side) & is_orn.notna() & syn_side.notna()
    contra = is_orn.notna() & syn_side.notna() & (is_orn != syn_side) & is_orn.isin(['left', 'right'])
    scale = np.ones(len(con), dtype=np.float32)
    scale[ipsi.to_numpy()] = ORN_IPSI_BIAS
    scale[contra.to_numpy()] = 2.0 - ORN_IPSI_BIAS
    con['w'] = con['syn_count'].to_numpy().astype(np.float32) * scale
    say(f"  ORN synapses: {int(ipsi.sum()):,} same-side x{ORN_IPSI_BIAS}, {int(contra.sum()):,} other-side x{2.0-ORN_IPSI_BIAS:.2f}")
    con = con.groupby(['pre_root_id', 'post_root_id'], as_index=False).agg(syn_count=('syn_count', 'sum'), w=('w', 'sum'))
    pre = id_to_new.reindex(con['pre_root_id'].to_numpy()).to_numpy()
    post = id_to_new.reindex(con['post_root_id'].to_numpy()).to_numpy()
    ok = ~(np.isnan(pre) | np.isnan(post))
    pre, post = pre[ok].astype(np.int64), post[ok].astype(np.int64)
    cnt = con['syn_count'].to_numpy()[ok].astype(np.float32)
    w = con['w'].to_numpy()[ok].astype(np.float32) * sign_by_new[pre]     # weight = (biased) synapse count, signed by the presynaptic neuron
    W = sparse.csr_matrix((w, (pre, post)), shape=(n, n), dtype=np.float32)
    say(f"  {W.nnz:,} connections, {int(cnt.sum()):,} synapses; {100*np.mean(w<0):.1f}% inhibitory connections; {time.time()-t0:.0f}s")

    # ---- real 3D positions (for the lab's hologram) ----------------------------------------
    xyz = np.full((n, 3), np.nan, dtype=np.float32)
    cpath = os.path.join(data_dir, 'coordinates.csv.gz')
    if os.path.exists(cpath):
        co = pd.read_csv(cpath, usecols=['root_id', 'position'], dtype={'root_id': np.int64}).drop_duplicates('root_id')
        pos = np.array([[float(v) for v in p.strip('[]').split()] for p in co['position']], dtype=np.float32)
        new = id_to_new.reindex(co['root_id'].to_numpy()).to_numpy()
        okc = ~np.isnan(new)
        xyz[new[okc].astype(int)] = pos[okc]
        good = ~np.isnan(xyz[:, 0])
        centre = np.nanmedian(xyz[good], axis=0); span = np.nanpercentile(np.abs(xyz[good] - centre), 99, axis=0).max()
        xyz = (xyz - centre) / span                                   # roughly -1..1, aspect preserved
        say(f"  positions for {int(good.sum()):,} neurons")
    meta = dict(root_id=root_ids, primary_type=df['primary_type'].to_numpy()[order].astype('U32'),
                side=df['side'].to_numpy()[order].astype('U8'), nt_type=df['nt_type'].to_numpy()[order].astype('U6'),
                super_class=df['super_class'].to_numpy()[order].astype('U24'), xyz=xyz)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, data=W.data, indices=W.indices, indptr=W.indptr, n=n,
                        region_names=np.array(list(regions)), region_bounds=np.array([regions[k] for k in regions]), **meta)
    say(f"  cached -> {cache} ({os.path.getsize(cache)/1e6:.0f} MB)")
    return W, regions, meta


def load(cache=CACHE):
    z = np.load(cache, allow_pickle=False)
    n = int(z['n'])
    W = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=(n, n))
    regions = {str(k): (int(a), int(b)) for k, (a, b) in zip(z['region_names'], z['region_bounds'])}
    meta = {k: z[k] for k in ('root_id', 'primary_type', 'side', 'nt_type', 'super_class', 'xyz') if k in z}
    return W, regions, meta


class FlyWireBrain(fly.FlyBrain):
    """Same API as fly.FlyBrain, but W and regions come from the FlyWire v783 connectome.

    Stated assumptions on top of the wiring (each one is a line in validation.md):
      * synapse strength = synapse count x g (mV), compressed where a neuron's total input
        exceeds `input_budget` synapses (gain control standing in for short-term depression
        and lateral inhibition; Kazama & Wilson 2008, Olsen & Wilson 2008)
      * olfactory receptor neurons release more on their own side (ORN_IPSI_BIAS; Gaudry 2013)
      * the legs read a specific set of steering descending neurons, not all 1,300
        (Rayshubskiy 2020, Braun 2024); see steering_neurons / select_steering_neurons()
      * the descending command is integrated over `motor_tau` steps before it moves the body
        (the nerve cord is not in this dataset)
    """
    NAME = 'FlyWire v783'

    def __init__(self, seed=None, cache=CACHE, params=None, input_budget=300.0, steering=None):
        if not os.path.exists(cache):
            build(cache=cache)
        self.rng = np.random.default_rng(seed)
        self.W, self.regions, self.meta = load(cache)
        self.n = self.W.shape[0]
        self.input_budget = input_budget
        if input_budget:
            tot = np.asarray(abs(self.W).sum(axis=0)).ravel()          # total |input| (synapses) per neuron
            scale = np.minimum(1.0, input_budget / np.maximum(tot, 1e-9)).astype(np.float32)   # compress only above budget
            self.W = (self.W @ sparse.diags(scale)).tocsr()
        c0, c1 = self.regions['central']
        cm = (c0 + c1) // 2
        self.central_L, self.central_R = (c0, cm), (cm, c1)   # not meaningful anatomically; kept for API
        self.params = {
            'v_rest': -65.0, 'v_thresh': -50.0, 'v_reset': -65.0, 'tau': 8.0, 'refrac': 2,
            'adapt_inc': 0.0, 'adapt_tau': 60.0,
            'I_bg': 1.3,         # background drive for spontaneous activity (Shiu et al. use none; tuned: ~2% firing)
            'noise': 0.5,
            'I_sens': 8.0,       # smell drive into the banana receptor neurons
            'I_sens_bg': 0.0,    # odour-independent drive into those receptor neurons
            'g_exc': 0.3,        # mV per synapse (Shiu et al.: 0.275 mV)
            'g_inh': 0.3,
            'ttx_frac': 0.0,
            'motor_tau': 8.0,    # steps over which the descending command is integrated (nerve-cord stand-in)
            'forward_base': 0.3, 'forward_gain': 15.0, 'turn_gain': 15.0, 'motor_out': 1.0,   # descending rates are ~2%: higher body gain
            # release gain per transmitter class (from FlyWire's nt_type): dopamine/octopamine/acetylcholine drugs act here
            'g_ACH': 1.0, 'g_GABA': 1.0, 'g_GLUT': 1.0, 'g_DA': 1.0, 'g_SER': 1.0, 'g_OCT': 1.0,
        }
        if params:
            self.params.update(params)
        self._defaults = dict(self.params)
        self.wiring = 'FlyWire v783 connectome'
        nt = np.asarray(self.meta['nt_type']).astype(str)
        self.nt_groups = {k: np.flatnonzero(nt == k) for k in self.NT_NAMES if (nt == k).any()}   # for the g_<NT> handles
        # steering readout: index arrays of descending neurons the legs listen to (default: all)
        self.set_steering(steering)
        self._rL = self._rR = self._drive = 0.0
        self._init_dynamics()

    # ---- readout ---------------------------------------------------------------------------
    def set_steering(self, steering):
        """steering: None (all descending neurons) or a list of cell-type names, or (idx_L, idx_R)."""
        T, S = self.meta['primary_type'], self.meta['side']
        aL, bL = self.regions['motor_L']; aR, bR = self.regions['motor_R']
        if steering is None:
            self.steer_L, self.steer_R = np.arange(aL, bL), np.arange(aR, bR)
        elif isinstance(steering, tuple):
            self.steer_L, self.steer_R = np.asarray(steering[0]), np.asarray(steering[1])
        else:
            names = set(steering)
            self.steer_L = np.array([i for i in range(aL, bL) if T[i] in names], dtype=int)
            self.steer_R = np.array([i for i in range(aR, bR) if T[i] in names], dtype=int)
        self.steering_names = sorted(set(T[self.steer_L]) | set(T[self.steer_R]))

    def _motor_rates(self, spiked):
        p = self.params
        aL, bL = self.regions['motor_L']; aR, bR = self.regions['motor_R']
        rL = float(spiked[self.steer_L].mean()) if self.steer_L.size else 0.0
        rR = float(spiked[self.steer_R].mean()) if self.steer_R.size else 0.0
        drive = 0.5 * (float(spiked[aL:bL].mean()) + float(spiked[aR:bR].mean()))
        k = 1.0 / max(1.0, p['motor_tau'])                          # leaky integration = nerve-cord stand-in
        self._rL += (rL - self._rL) * k; self._rR += (rR - self._rR) * k; self._drive += (drive - self._drive) * k
        return self._rL, self._rR, 2.0 * self._drive

    def restore(self):
        super().restore()
        self._rL = self._rR = self._drive = 0.0

    def describe(self):
        R = self.regions
        return (f"FlyWire v783: {self.n:,} neurons, {self.W.nnz:,} connections. "
                f"nose L/R = {R['sens_L'][1]-R['sens_L'][0]}/{R['sens_R'][1]-R['sens_R'][0]} banana ORNs "
                f"({', '.join(g.replace('ORN_','') for g in BANANA_GLOMERULI)}); "
                f"legs L/R = {R['motor_L'][1]-R['motor_L'][0]}/{R['motor_R'][1]-R['motor_R'][0]} descending neurons, "
                f"steering read from {self.steer_L.size}/{self.steer_R.size} of them; central = {R['central'][1]-R['central'][0]:,}.")


def select_steering_neurons(brain, world_cls, steps=300, seed=0, top_frac=0.05, verbose=True):
    """Find the descending neurons whose firing depends on which side the banana is on.
    Each DN gets a score = rate(banana on its own side) - rate(banana on the other side), in units
    of its own trial-to-trial noise; the top `top_frac` on each side are returned as (idx_L, idx_R).
    Use one seed to select and a DIFFERENT seed to test (check_fly.py), so the choice is not fitted to noise."""
    rng = np.random.default_rng(seed)
    w = world_cls(seed=seed)
    aL, bL = brain.regions['motor_L']; aR, bR = brain.regions['motor_R']
    dn = np.arange(aL, bR)
    rates = {+1: [], -1: []}
    for rep in range(3):
        for side in (+1, -1):
            brain.v[:] = brain.params['v_rest'] + rng.normal(0, 2, brain.n); brain.refrac[:] = 0
            w.sugar_x, w.sugar_y = 0.0, 5.0 * side
            acc = np.zeros(dn.size)
            for i in range(60 + steps):
                s, _ = w.get_sensory(0, 0, 0); _, sp = brain.step(s)
                if i >= 60: acc += sp[dn]
            rates[side].append(acc / steps)
    L = np.array(rates[+1]); Rr = np.array(rates[-1])          # (reps, dn): banana LEFT / RIGHT
    own = np.where(dn < bL, L, Rr); other = np.where(dn < bL, Rr, L)   # "own side" of each DN
    d = own.mean(0) - other.mean(0)
    sd = np.sqrt(0.5 * (own.var(0) + other.var(0))) + 1e-4
    score = d / sd
    def top(mask):
        idx = dn[mask]; sc = score[mask]
        k = max(3, int(top_frac * idx.size))
        pick = idx[np.argsort(-sc)[:k]]
        return pick[score[np.isin(dn, pick)] > 0] if False else pick
    selL, selR = top(dn < bL), top(dn >= bL)
    if verbose:
        T = brain.meta['primary_type']
        print(f"  steering DNs selected (score = own-side minus other-side rate, in noise units): "
              f"{selL.size} left, {selR.size} right; mean score L {score[np.isin(dn, selL)].mean():.2f} R {score[np.isin(dn, selR)].mean():.2f}")
        print("  left :", sorted(set(T[selL]))[:20]); print("  right:", sorted(set(T[selR]))[:20])
    return selL, selR, score


if __name__ == '__main__':
    W, regions, meta = build()
    b = FlyWireBrain(seed=0)
    print(b.describe())
