"""
narrator.py  -  optional Muse Spark narration of what the fly senses, shared by run.py-style
viewers and server.py.  Numbers in, one sentence out.  It never controls anything.

    export META_API_KEY='LLM_...'            (optional: META_MODEL, default muse-spark-1.3)
"""
import os, threading


class Narrator:
    FALLBACK = "(local) {where}, {pct:.1f}% of my brain firing, {doing}"

    def __init__(self, enabled=True, quiet=False):
        self.thought = "waking up..."
        self.lock = threading.Lock()
        self.in_flight = False
        self.client, self.model, self.errors = None, None, 0
        self.source = 'local'
        try:                                       # same .env file agent.py reads (git-ignored)
            import agent as _a; _a.load_env()
        except Exception:
            pass
        key = os.environ.get('META_API_KEY', '')
        if not enabled or not key or key.startswith('your_'):
            if not quiet:
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
            if want not in names:
                print(f"Narrator: model '{want}' not in {names}; using it anyway")
            self.model = want                          # exact name only, never a '-contributor' variant
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
        if st.get('scrambled'):
            doing = "wires scrambled, wandering"
        if st.get('drugs'):
            doing += " under " + ", ".join(f"{k} {v:.2f}" for k, v in st['drugs'].items())
        return self.FALLBACK.format(where=where, pct=100 * st['rate_all'], doing=doing)

    def ask(self, st):
        """Non-blocking; one request in flight at a time."""
        if self.client is None:
            with self.lock:
                self.thought = self.local_line(st)
            return
        if self.in_flight:
            return
        self.in_flight = True
        drugs = ", ".join(f"{k} {v:.2f}" for k, v in st.get('drugs', {}).items()) or "none"
        prompt = (f"Live numbers from my brain simulation: left smell {st['left']:.2f}, right smell {st['right']:.2f}, "
                  f"banana {st['dist']:.1f} units away, {100*st['rate_all']:.1f}% of my {st['n']} neurons firing, "
                  f"motor left {100*st['rate_L']:.1f}% right {100*st['rate_R']:.1f}%, forward {st['forward']:.2f}, "
                  f"turn {st['turn']:+.2f}, synapses {'SCRAMBLED' if st.get('scrambled') else 'intact'}, "
                  f"drugs on board: {drugs}. Say what I sense and what I am doing in ONE sentence, "
                  f"max 14 words, first person, no emojis.")

        def call():
            try:
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=40, temperature=0.8,
                    messages=[{"role": "system",
                               "content": "You are Maxime, a simulated fruit fly whose 139,255-neuron brain is being "
                                          "simulated in a lab. You narrate what you sense; you do not control anything "
                                          "and you never talk about humans or medicine. Reply with the sentence only."},
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
