"""
run.py  -  live viewer for the fly in fly.py.

    python3 run.py                 # window: arena + brain raster + region rates
    python3 run.py --steps 400 --snapshot frame.png   # headless smoke test

Keys in the window:   S scramble synapses   R restore   B move banana   Q quit

Narration (optional): set META_API_KEY to have Muse Spark narrate what the fly senses.
The narrator only RECEIVES numbers and writes a sentence; it never steers the fly.
    export META_API_KEY='LLM_...'   (optional: META_MODEL, default muse-spark-1.3)
"""
import os, sys, time, threading, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib.gridspec import GridSpec
import fly

# ----------------------------------------------------------------------------- args
ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=0, help='stop after N steps (0 = run until window closes)')
ap.add_argument('--snapshot', type=str, default='', help='save the last frame to this PNG')
ap.add_argument('--seed', type=int, default=None)
ap.add_argument('--draw-every', type=int, default=2, help='draw one frame every N brain steps')
ap.add_argument('--no-narrator', action='store_true')
args = ap.parse_args()
if args.snapshot and args.steps:
    matplotlib.use('Agg')                       # headless test: no window needed

# ----------------------------------------------------------------------------- narrator
class Narrator:
    """Asks Muse Spark for a one-line first-person thought every few steps.
    Gets numbers in, text out. Never touches the brain or the body."""
    FALLBACK = "(local) {where}, {pct:.1f}% of my brain firing, {doing}"

    def __init__(self, enabled=True):
        self.thought = "Maxime is waking up..."
        self.lock = threading.Lock()
        self.in_flight = False
        self.client, self.model, self.errors = None, None, 0
        self.source = 'local'
        key = os.environ.get('META_API_KEY', '')
        if not enabled or not key or key.startswith('your_'):
            print("Narrator: local fallback (set META_API_KEY for Muse Spark narration)")
            return
        try:
            from openai import OpenAI
        except ImportError:
            print("Narrator: 'openai' package missing -> pip3 install openai ; using local fallback")
            return
        base = os.environ.get('META_BASE_URL', 'https://api.meta.ai/v1')
        want = os.environ.get('META_MODEL', 'muse-spark-1.3')
        try:
            self.client = OpenAI(api_key=key, base_url=base)
            names = [m.id for m in self.client.models.list().data]
            if want in names:
                self.model = want                       # exact match only, never a "-contributor" variant
            else:
                print(f"Narrator: model '{want}' not in {names}; using it anyway")
                self.model = want
            self.source = 'muse'
            print(f"Narrator: Muse via {base}  model={self.model}")
        except Exception as e:
            print(f"Narrator: could not reach Meta API ({e!r}); using local fallback")
            self.client = None

    def local_line(self, st):
        where = ("banana on my left" if st['left'] > st['right'] + 0.05 else
                 "banana on my right" if st['right'] > st['left'] + 0.05 else
                 "banana straight ahead" if st['dist'] < 4 else "faint banana somewhere")
        doing = ("turning left" if st['turn'] > 0.1 else "turning right" if st['turn'] < -0.1 else "walking on")
        if st['scrambled']:
            doing = "wires scrambled, wandering"
        return self.FALLBACK.format(where=where, pct=100 * st['rate_all'], doing=doing)

    def ask(self, st):
        """Non-blocking. Call every N steps; one request in flight at a time."""
        if self.client is None:
            with self.lock:
                self.thought = self.local_line(st)
            return
        if self.in_flight:
            return
        self.in_flight = True
        prompt = (f"Live numbers from my brain simulation: left smell {st['left']:.2f}, right smell {st['right']:.2f}, "
                  f"banana {st['dist']:.1f} units away, {100*st['rate_all']:.1f}% of my {st['n']} neurons firing, "
                  f"motor left {100*st['rate_L']:.1f}% right {100*st['rate_R']:.1f}%, forward {st['forward']:.2f}, "
                  f"turn {st['turn']:+.2f}, synapses {'SCRAMBLED' if st['scrambled'] else 'intact'}. "
                  f"Say what I sense and what I am doing in ONE sentence, max 14 words, first person, no emojis.")

        def call():
            try:
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=40, temperature=0.8,
                    messages=[{"role": "system",
                               "content": "You are Maxime, a fruit fly whose 139,255-neuron brain is being simulated. "
                                          "You narrate; you do not control anything. Reply with the sentence only."},
                              {"role": "user", "content": prompt}])
                text = r.choices[0].message.content.strip().strip('"')
                with self.lock:
                    self.thought = text[:110]
            except Exception as e:
                self.errors += 1
                if self.errors == 1:
                    print(f"Narrator: first error from Meta API (further errors silent): {e!r}")
                with self.lock:
                    self.thought = self.local_line(st)
            finally:
                self.in_flight = False
        threading.Thread(target=call, daemon=True).start()

    def get(self):
        with self.lock:
            return self.thought

# ----------------------------------------------------------------------------- build
print("Building the fly brain...")
t0 = time.time()
brain = fly.FlyBrain(seed=args.seed)
body = fly.FlyBody()
world = fly.World(seed=args.seed)
world.randomize_sugar()
body.leg_phases = np.array([0, np.pi, np.pi, 0, 0, np.pi])          # tripod gait
narrator = Narrator(enabled=not args.no_narrator)
print(f"  {brain.n:,} neurons, {brain.W.nnz:,} synapses, built in {time.time()-t0:.1f}s")

# ----------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(14, 7.2), facecolor='white')
gs = GridSpec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[2.4, 1.0], figure=fig, wspace=0.22, hspace=0.35)
ax_world = fig.add_subplot(gs[:, 0])
ax_raster = fig.add_subplot(gs[0, 1])
ax_rates = fig.add_subplot(gs[1, 1])
fig.canvas.manager.set_window_title('Maxime - 139,255-neuron fly') if hasattr(fig.canvas, 'manager') and fig.canvas.manager else None

# --- arena ---------------------------------------------------------------------
A = fly.FlyBody.ARENA
ax_world.set_xlim(-A, A); ax_world.set_ylim(-A, A); ax_world.set_aspect('equal')
ax_world.set_facecolor('#fdf6e3'); ax_world.set_xticks([]); ax_world.set_yticks([])
ax_world.add_patch(Rectangle((-A, -A), 2 * A, 2 * A, fill=False, lw=3, ec='#c8c0a8'))
trail_line, = ax_world.plot([], [], color='#b0b0b0', lw=1, alpha=0.7, zorder=1)
banana = Circle((world.sugar_x, world.sugar_y), world.catch_radius, color='#FFD23F', alpha=0.9, zorder=3)
banana_core = Circle((world.sugar_x, world.sugar_y), 0.35, color='#E0A800', zorder=4)
ax_world.add_patch(banana); ax_world.add_patch(banana_core)
abdomen = Ellipse((0, 0), 0.8, 0.5, color='#8B4513', zorder=10)
thorax = Ellipse((0, 0), 0.6, 0.45, color='#5C3317', zorder=11)
head = Circle((0, 0), 0.22, color='#2F1B0C', zorder=12)
eye_l = Circle((0, 0), 0.13, color='#E53935', zorder=13)
eye_r = Circle((0, 0), 0.13, color='#E53935', zorder=13)
wing_l = Ellipse((0, 0), 1.0, 0.35, color='gray', alpha=0.3, zorder=9)
wing_r = Ellipse((0, 0), 1.0, 0.35, color='gray', alpha=0.3, zorder=9)
for p in (wing_l, wing_r, abdomen, thorax, head, eye_l, eye_r):
    ax_world.add_patch(p)
legs = [ax_world.plot([], [], color='black', lw=1.8, zorder=12, solid_capstyle='round')[0] for _ in range(6)]
stats_txt = ax_world.set_xlabel('', fontsize=9, family='monospace', color='#333', loc='left')
help_txt = ax_world.text(0.99, 0.985, '[S] scramble   [R] restore   [B] banana   [Q] quit',
                         transform=ax_world.transAxes, fontsize=8, va='top', ha='right', color='#777')
title = ax_world.set_title('', fontsize=11, fontweight='bold', loc='left')

def update_fly(x, y, ang, phases):
    c, s = np.cos(ang), np.sin(ang)
    deg = np.degrees(ang)
    abdomen.set_center((x - c * 0.5, y - s * 0.5)); abdomen.set_angle(deg)
    thorax.set_center((x, y)); thorax.set_angle(deg)
    hx, hy = x + c * 0.45, y + s * 0.45
    head.set_center((hx, hy))
    eye_l.set_center((hx - s * 0.12, hy + c * 0.12)); eye_r.set_center((hx + s * 0.12, hy - c * 0.12))
    wing_l.set_center((x - c * 0.1 - s * 0.25, y - s * 0.1 + c * 0.25)); wing_l.set_angle(deg + 40)
    wing_r.set_center((x - c * 0.1 + s * 0.25, y - s * 0.1 - c * 0.25)); wing_r.set_angle(deg - 40)
    k = 0
    for side in (-1, 1):
        for i, off in enumerate((0.25, 0.0, -0.25)):
            swing = np.sin(phases[(i * 2 + (0 if side == -1 else 1)) % 6]) * 0.22
            bx, by = x + c * off, y + s * off
            cx, cy = bx + side * s * 0.15, by - side * c * 0.15
            fx, fy = cx + side * s * 0.28 + c * swing, cy - side * c * 0.28 + s * swing
            tx, ty = fx + side * s * 0.22 + c * swing * 0.6, fy - side * c * 0.22 + s * swing * 0.6
            legs[k].set_data([bx, cx, fx, tx], [by, cy, fy, ty]); k += 1

# --- brain raster: a fixed sample of neurons, time runs left -> right ----------------
SAMPLE = {'sens_L': 100, 'sens_R': 100, 'central': 500, 'motor_L': 150, 'motor_R': 150}
sample_idx, bands = [], []
for name, cnt in SAMPLE.items():
    a, b = brain.regions[name]
    sample_idx.append(np.linspace(a, b - 1, cnt).astype(int)); bands.append((name, cnt))
sample_idx = np.concatenate(sample_idx)
T = 240
raster = np.zeros((sample_idx.size, T), dtype=np.uint8)
img = ax_raster.imshow(raster, aspect='auto', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
y0, ticks, labels = 0, [], []
for name, cnt in bands:
    ticks.append(y0 + cnt / 2); labels.append(name)
    y0 += cnt
    if y0 < sample_idx.size:
        ax_raster.axhline(y0 - 0.5, color='#d33', lw=0.8, alpha=0.6)
ax_raster.set_yticks(ticks); ax_raster.set_yticklabels(labels, fontsize=8)
ax_raster.set_xticks([]); ax_raster.set_xlabel(f'last {T} steps  ({sample_idx.size} of {brain.n:,} neurons)', fontsize=8)
raster_title = ax_raster.set_title('', fontsize=10, loc='left')

# --- region firing rates ---------------------------------------------------------
names = list(brain.regions)
bars = ax_rates.bar(names, [0] * 5, color=['#1f77b4', '#1f77b4', '#7f7f7f', '#d62728', '#d62728'])
ax_rates.set_ylim(0, 15); ax_rates.set_ylabel('% firing / step', fontsize=8)
ax_rates.tick_params(labelsize=8)
rate_txts = [ax_rates.text(i, 0.3, '', ha='center', va='bottom', fontsize=8) for i in range(5)]

# ----------------------------------------------------------------------------- keys
state = {'quit': False, 'caught': 0, 'trail': []}

def on_key(ev):
    if ev.key in ('s', 'S'):
        brain.scramble(); print(">>> SCRAMBLED: every synapse sent to a random target")
    elif ev.key in ('r', 'R'):
        brain.restore(); print(">>> RESTORED: original synapses back")
    elif ev.key in ('b', 'B'):
        world.randomize_sugar(); state['trail'].clear(); print(">>> banana moved")
    elif ev.key in ('q', 'Q', 'escape'):
        state['quit'] = True
fig.canvas.mpl_connect('key_press_event', on_key)

# ----------------------------------------------------------------------------- loop
print("Running. Keys in the window: S scramble, R restore, B banana, Q quit.")
if matplotlib.get_backend().lower() != 'agg':
    plt.ion()
    plt.show(block=False)                       # open the window; the loop below drives it
step, t_fps, fps = 0, time.time(), 0.0
spiked = np.zeros(brain.n, dtype=bool)
try:
    while not state['quit'] and plt.fignum_exists(fig.number):
        sensory, dist = world.get_sensory(body.x, body.y, body.angle)
        if dist < world.catch_radius:
            state['caught'] += 1
            print(f"BANANA #{state['caught']} at step {step}")
            world.randomize_sugar(); state['trail'].clear()
        motor, spiked = brain.step(sensory)
        x, y, ang = body.move(motor)
        state['trail'].append((x, y))
        if len(state['trail']) > 300:
            del state['trail'][0]
        raster[:, :-1] = raster[:, 1:]
        raster[:, -1] = spiked[sample_idx]

        if step % 25 == 0:
            r = brain.rates(spiked)
            narrator.ask({'left': float(sensory[0]), 'right': float(sensory[1]), 'dist': dist,
                          'rate_all': float(spiked.mean()), 'rate_L': r['motor_L'], 'rate_R': r['motor_R'],
                          'forward': float(motor[0]), 'turn': float(motor[1]), 'n': brain.n,
                          'scrambled': brain.scrambled})
        if step % 50 == 0 and step:
            now = time.time(); fps = 50 / (now - t_fps); t_fps = now

        if step % args.draw_every == 0:
            update_fly(x, y, ang, body.leg_phases)
            banana.set_center((world.sugar_x, world.sugar_y)); banana_core.set_center((world.sugar_x, world.sugar_y))
            if state['trail']:
                tx, ty = zip(*state['trail']); trail_line.set_data(tx, ty)
            else:
                trail_line.set_data([], [])
            r = brain.rates(spiked)
            for bar, txt, name in zip(bars, rate_txts, names):
                bar.set_height(100 * r[name]); txt.set_text(f"{100*r[name]:.1f}"); txt.set_y(100 * r[name] + 0.3)
            img.set_data(raster)
            n_spk = int(spiked.sum())
            scr = brain.scrambled
            title.set_text(narrator.get())
            title.set_color('#b00020' if scr else '#222')
            raster_title.set_text(f"{'SCRAMBLED  ' if scr else ''}{n_spk:,} / {brain.n:,} neurons firing")
            raster_title.set_color('#b00020' if scr else '#222')
            stats_txt.set_text(f"step {step}  fps {fps:4.1f}  dist {dist:4.1f}  bananas {state['caught']}  "
                               f"fwd {float(motor[0]):.2f} turn {float(motor[1]):+.2f}")
            fig.canvas.draw_idle(); fig.canvas.flush_events()

        if step % 100 == 0:
            print(f"step {step:6d}  fps {fps:4.1f}  dist {dist:4.1f}  bananas {state['caught']}  "
                  f"firing {100*spiked.mean():.2f}%  {'SCRAMBLED ' if brain.scrambled else ''}| {narrator.get()}")
        step += 1
        if args.steps and step >= args.steps:
            break
except KeyboardInterrupt:
    pass

if args.snapshot:
    fig.savefig(args.snapshot, dpi=110)
    print(f"saved {args.snapshot}")
print(f"Done: {step} steps, {state['caught']} bananas.")
plt.close('all')
