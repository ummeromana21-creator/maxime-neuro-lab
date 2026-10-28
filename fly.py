"""
fly.py  -  MAXIME: a 139,255-neuron spiking fly brain, a body, and a world.

Maxime is a simulated female fruit fly (Drosophila melanogaster), FlyWire-scale.
Three classes, used by run.py (the viewer), server.py (the lab) and check_fly.py (the referee):

    FlyBrain  leaky integrate-and-fire network, FlyWire-scale (139,255 neurons),
              random but LATERALISED wiring: left smell -> left central -> left motor.
    FlyBody   turns motor-pool firing into movement; walls keep it in the arena.
    World     a banana that smells stronger when near and on the side it is on.

Honesty rules this file obeys (do not break them when editing):
  1. The ONLY path from smell to movement is spikes -> synapses (W) -> motor spikes.
     `forward` and `turn` are computed from motor-pool firing rates and nothing else.
  2. No spikes are ever forced, injected or faked. If firing is wrong, tune params.
  3. The wiring is NOT the FlyWire connectome. It borrows FlyWire's neuron count and
     a left/right structure. Say "FlyWire-scale", never "the real connectome".

Everything tunable lives in FlyBrain.params and FlyBody.params so that experiments
(drugs, lesions) can change them without touching the code.
"""
import numpy as np
from scipy import sparse

NAME = "Maxime"             # the fly's name
N_FLYWIRE = 139255          # neurons in the FlyWire female adult brain (v783)


class FlyBrain:
    # ------------------------------------------------------------------ build
    def __init__(self, n_neurons=N_FLYWIRE, seed=None):
        self.rng = np.random.default_rng(seed)
        self.n = n = int(n_neurons)

        # ---- regions (fractions of n, so a small brain works too) --------------
        nS = max(1, int(n * 5000 / N_FLYWIRE))       # sensory neurons per side
        nM = max(1, int(n * 9500 / N_FLYWIRE))       # motor neurons per side
        c0, c1 = 2 * nS, n - 2 * nM                  # central = everything between
        cm = (c0 + c1) // 2
        self.regions = {
            'sens_L':  (0, nS),
            'sens_R':  (nS, 2 * nS),
            'central': (c0, c1),
            'motor_L': (c1, c1 + nM),
            'motor_R': (c1 + nM, n),
        }
        self.central_L = (c0, cm)                    # left and right halves of central
        self.central_R = (cm, c1)

        # ---- everything tunable --------------------------------------------------
        self.params = {
            'v_rest':   -65.0,   # mV
            'v_thresh': -50.0,   # mV   spike threshold
            'v_reset':  -65.0,   # mV
            'tau':        8.0,   # steps membrane time constant
            'refrac':       2,   # steps refractory period
            'adapt_inc':    0.0, # mV of adaptation current per spike in central neurons (Ca-activated
                                 #   K current). OFF by default; an experimenter may switch it on.
            'adapt_tau':   60.0, # steps: how slowly adaptation decays (the slow variable bursts need)
            'I_bg':       1.3,   # background drive to every neuron (keeps brain awake)
                                 #   1.0 -> brain nearly silent, 1.7 -> runaway; 1.3 is the sweet spot
            'noise':      0.8,   # std of per-step current noise
            'I_sens':     2.5,   # current per unit smell into sensory neurons
            'g_exc':      0.5,   # gain of excitatory synapses (drugs: nicotine up, anaesthetics down)
            'g_inh':      0.5,   # gain of inhibitory synapses (drugs: picrotoxin -> 0, ethanol up)
            'ttx_frac':   0.0,   # fraction of neurons whose sodium channels are blocked (cannot spike)
            'forward_base': 0.3, # body: minimum forward drive
            'forward_gain': 3.0, # body: forward per unit (rate_L + rate_R)
            'turn_gain':   15.0, # body: turn per unit (rate_L - rate_R)
                                 #   below ~10 the turning circle is wider than the banana -> orbits
            'motor_out':    1.0, # scales motor output (0 = neuromuscular block)
        }
        self._defaults = dict(self.params)

        # ---- synapses ---------------------------------------------------------------
        self.W = self._build_connectome()
        self.W0 = self.W.copy()                      # pristine copy for restore()
        self._WT, self._WT_neg, self._WT_src = None, None, None   # caches of W.T (see step)

        # ---- state -------------------------------------------------------------------
        p = self.params
        self.v = self.rng.normal(p['v_rest'], 2.0, n).astype(np.float32)
        self.refrac = np.zeros(n, dtype=np.int16)
        self.adapt = np.zeros(n, dtype=np.float32)   # spike-frequency adaptation current
        self.adapt_mask = np.zeros(n, dtype=np.float32)
        self.adapt_mask[c0:c1] = 1.0                 # adaptation lives in the central (recurrent) pool
        self.blocked = np.zeros(n, dtype=bool)       # TTX: neurons that cannot spike
        self._blocked_frac = 0.0
        self.scrambled = False

    # recurrent wiring of the central brain (k synapses per neuron, mean weight, fraction
    # inhibitory, inhibitory strength relative to excitatory).  These values give a
    # DRIVE-DOMINATED network: activity is set by background current, not held down by
    # inhibition.  See validation.md - a balanced regime (e.g. 40, 1.0, 0.2, 4.0 with
    # adapt_inc > 0) produces burst dynamics under picrotoxin but clamps the central pool
    # so hard that smell no longer steers the fly.  Open problem, documented, not hidden.
    REC_K, REC_W, REC_INH_FRAC, REC_INH_SCALE = 20, 0.6, 0.2, 1.0

    def _build_connectome(self):
        """Random sparse wiring with left/right structure. Fully vectorised: builds
        ~8M synapses in about a second. W[src, tgt] = weight; negative = inhibitory."""
        rng = self.rng
        R = self.regions
        sL, sR, cL, cR, mL, mR = (R['sens_L'], R['sens_R'], self.central_L,
                                  self.central_R, R['motor_L'], R['motor_R'])
        rows, cols, data = [], [], []

        def connect(src, tgt, k, w, inhib_frac=0.0, inhib_scale=1.0):
            """Every neuron in src makes k synapses onto random neurons in tgt.
            A fraction inhib_frac of them are inhibitory, inhib_scale times as strong."""
            a, b = src
            c, d = tgt
            s = np.repeat(np.arange(a, b, dtype=np.int32), k)
            t = rng.integers(c, d, size=s.size, dtype=np.int32)
            wts = np.abs(rng.normal(w, 0.3 * w, size=s.size)).astype(np.float32)
            if inhib_frac > 0:
                wts[rng.random(s.size) < inhib_frac] *= -inhib_scale
            keep = s != t                                   # no autapses
            rows.append(s[keep]); cols.append(t[keep]); data.append(wts[keep])

        # smell -> central, mostly same side (lateralised), a little crosstalk
        connect(sL, cL, 80, 3.0);   connect(sL, cR, 10, 1.5)
        connect(sR, cR, 80, 3.0);   connect(sR, cL, 10, 1.5)
        # central recurrent (within side): balanced excitation / inhibition
        connect(cL, cL, self.REC_K, self.REC_W, self.REC_INH_FRAC, self.REC_INH_SCALE)
        connect(cR, cR, self.REC_K, self.REC_W, self.REC_INH_FRAC, self.REC_INH_SCALE)
        # contralateral inhibition between the halves (push-pull)
        connect(cL, cR, 8, 0.8, inhib_frac=1.0)
        connect(cR, cL, 8, 0.8, inhib_frac=1.0)
        # central -> motor, same side excites, other side gets a little inhibition
        connect(cL, mL, 7, 1.5);    connect(cL, mR, 3, 1.0, inhib_frac=1.0)
        connect(cR, mR, 7, 1.5);    connect(cR, mL, 3, 1.0, inhib_frac=1.0)
        # weak random long-range connections everywhere (keeps it brain-like)
        connect((0, self.n), (0, self.n), 10, 0.3, inhib_frac=0.2)

        rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
        W = sparse.csr_matrix((data, (rows, cols)), shape=(self.n, self.n), dtype=np.float32)
        W.sum_duplicates()
        return W

    # ------------------------------------------------------------------ dynamics
    def step(self, sensory):
        """One time step. sensory[0] = left smell, sensory[1] = right smell (0..1).
        Returns (motor, spiked): motor = [forward, turn]; spiked = bool array (n,)."""
        p = self.params
        n = self.n
        sensory = np.asarray(sensory, dtype=np.float32).ravel()
        left = float(sensory[0]) if sensory.size > 0 else 0.0
        right = float(sensory[1]) if sensory.size > 1 else left

        # input current: background + noise + smell into the sensory neurons
        I = self.rng.normal(p['I_bg'], p['noise'], n).astype(np.float32)
        a, b = self.regions['sens_L']; I[a:b] += p['I_sens'] * left
        a, b = self.regions['sens_R']; I[a:b] += p['I_sens'] * right

        # leaky integrate-and-fire with spike-frequency adaptation
        v = self.v
        v += (p['v_rest'] - v) / p['tau'] + I - self.adapt
        self.adapt -= self.adapt / p['adapt_tau']
        in_ref = self.refrac > 0
        v[in_ref] = p['v_reset']
        self.refrac[in_ref] -= 1
        spiked = v >= p['v_thresh']
        if p['ttx_frac'] > 0.0:                         # sodium-channel block: these neurons never fire
            if p['ttx_frac'] != self._blocked_frac:
                self.blocked = self.rng.random(n) < p['ttx_frac']
                self._blocked_frac = p['ttx_frac']
            spiked &= ~self.blocked
        elif self._blocked_frac:
            self.blocked[:] = False; self._blocked_frac = 0.0
        v[spiked] = p['v_reset']
        self.refrac[spiked] = p['refrac']
        self.adapt[spiked] += p['adapt_inc'] * self.adapt_mask[spiked]

        # synaptic transmission: spikes travel along W to their targets
        if spiked.any():
            if self._WT_src is not self.W:              # W was replaced (lesion, scramble, restore)
                self._WT = self.W.T.tocsr()
                self._WT_neg = -self._WT.minimum(0)     # |inhibitory part|, for separate g_inh
                self._WT_src = self.W
            s = spiked.astype(np.float32)
            g_exc, g_inh = p['g_exc'], p['g_inh']
            inp = self._WT.dot(s) * g_exc               # = g_exc*(E - I)
            if g_inh != g_exc:                          # g_exc*E - g_inh*I = g_exc*(E-I) + (g_exc-g_inh)*I
                inp += self._WT_neg.dot(s) * (g_exc - g_inh)
            v += inp.astype(np.float32)
        np.clip(v, -90.0, p['v_thresh'], out=v)        # keep voltages sane

        # motor output: ONLY motor-pool firing rates reach the body
        a, b = self.regions['motor_L']; rate_L = float(spiked[a:b].mean())
        a, b = self.regions['motor_R']; rate_R = float(spiked[a:b].mean())
        forward = (p['forward_base'] + p['forward_gain'] * (rate_L + rate_R)) * p['motor_out']
        turn = p['turn_gain'] * (rate_L - rate_R) * p['motor_out']
        return np.array([forward, turn], dtype=np.float32), spiked

    def rates(self, spiked):
        """Fraction of neurons that fired in each region (for displays / experiments)."""
        return {k: float(spiked[a:b].mean()) for k, (a, b) in self.regions.items()}

    # ------------------------------------------------------------------ experiments
    def scramble(self):
        """Keep every synapse's source and weight, send it to a random target.
        Same neurons, same number of synapses, no structure: the control experiment."""
        coo = self.W.tocoo()
        new_cols = self.rng.integers(0, self.n, size=coo.nnz, dtype=np.int32)
        self.W = sparse.csr_matrix((coo.data, (coo.row, new_cols)),
                                   shape=(self.n, self.n), dtype=np.float32)
        self.scrambled = True

    def restore(self):
        """Undo scramble / lesions / drug changes: original synapses and parameters."""
        self.W = self.W0.copy()
        self.params.update(self._defaults)
        self.blocked[:] = False; self._blocked_frac = 0.0
        self.scrambled = False


class FlyBody:
    """Turns [forward, turn] into position. Walls at +-10 bounce the fly back."""
    ARENA = 10.0

    def __init__(self):
        self.x, self.y, self.angle = 0.0, 0.0, 0.0
        self.leg_phases = np.random.rand(6) * 2 * np.pi    # cosmetic, for the legs
        self.params = {'step': 0.15, 'turn_step': 0.15, 'turn_max': 3.0}

    def move(self, motor):
        p = self.params
        forward = float(motor[0])
        turn = float(np.clip(motor[1], -p['turn_max'], p['turn_max']))
        self.angle += turn * p['turn_step']
        nx = self.x + forward * np.cos(self.angle) * p['step']
        ny = self.y + forward * np.sin(self.angle) * p['step']
        lim = self.ARENA - 0.1
        if abs(nx) > lim:                                    # hit an x wall: reflect heading
            self.angle = np.pi - self.angle
            nx = float(np.clip(nx, -lim, lim))
        if abs(ny) > lim:                                    # hit a y wall
            self.angle = -self.angle
            ny = float(np.clip(ny, -lim, lim))
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi
        self.x, self.y = nx, ny
        self.leg_phases += 0.6 * forward + 0.05
        return self.x, self.y, self.angle


class World:
    """One banana. Smell is stronger when close and on the side the banana is on."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.sugar_x, self.sugar_y = 5.0, 0.0
        self.catch_radius = 0.9

    def randomize_sugar(self, lim=6.0):
        self.sugar_x = float(self.rng.uniform(-lim, lim))
        self.sugar_y = float(self.rng.uniform(-lim, lim))

    def get_sensory(self, x, y, angle):
        dx, dy = self.sugar_x - x, self.sugar_y - y
        dist = float(np.hypot(dx, dy))
        diff = np.arctan2(dy, dx) - angle                      # banana bearing relative to heading
        base = 1.0 / (1.0 + dist / 4.0)                        # never exactly zero: a gradient everywhere
        side = 0.6 * np.sin(diff)                              # +ve when the banana is on the left
        left = float(np.clip(base * (1.0 + side), 0.0, 1.0))
        right = float(np.clip(base * (1.0 - side), 0.0, 1.0))
        return np.array([left, right], dtype=np.float32), dist
