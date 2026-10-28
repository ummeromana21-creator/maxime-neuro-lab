"""
check_server.py - talks protocol.md to a running server and checks the answers.
    python3 server.py        (or fake_server.py)  in one terminal
    python3 check_server.py  in another
"""
import asyncio, json, sys, time
import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else 'ws://localhost:8765'
results = {}

async def main():
    async with websockets.connect(URL, max_size=2 ** 20) as ws:
        hello = json.loads(await ws.recv())
        ok = hello.get('type') == 'hello' and 'regions' in hello and 'sample' in hello
        print(f"hello: mode={hello.get('mode')} n={hello.get('n')} drugs={sorted(hello.get('drugs', {}))}")
        results['hello'] = ok

        async def wait_for(pred, timeout=60):
            t0 = time.time()
            while time.time() - t0 < timeout:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout))
                if pred(msg):
                    return msg
            raise TimeoutError

        async def cmd(c, **kw):
            cid = f"c{int(time.time()*1000) % 100000}"
            await ws.send(json.dumps(dict(cmd=c, id=cid, **kw)))
            ack = await wait_for(lambda m: m.get('type') == 'ack' and m.get('id') == cid)
            return ack

        # states arrive and carry the fields the lab needs
        st = await wait_for(lambda m: m.get('type') == 'state')
        need = ['fly', 'banana', 'rates', 'fired', 'scrambled', 'drugs', 'thought', 'sps']
        results['state_fields'] = all(k in st for k in need)
        print(f"state: t={st['t']} sps={st['sps']} fired={len(st['fired'])} rates.all={st['rates'].get('all')}")

        # 30 Hz-ish
        n, t0 = 0, time.time()
        while time.time() - t0 < 1.0:
            m = json.loads(await ws.recv()); n += m.get('type') == 'state'
        print(f"state rate: {n}/s"); results['rate'] = 15 <= n <= 45

        a = await cmd('ping'); results['ping'] = a['ok']
        a = await cmd('dose', drug='isoflurane', amount=0.8); print("dose:", a['ok'], a['data']); results['dose'] = a['ok']
        a = await cmd('dose', drug='not_a_drug', amount=1); results['bad_drug_rejected'] = (not a['ok']); print("bad drug ->", a['error'])
        a = await cmd('washout'); results['washout'] = a['ok']
        a = await cmd('restore'); results['restore'] = a['ok']
        a = await cmd('speed', sps=0); results['speed'] = a['ok']

        # the by-hand experiment from protocol.md
        a = await cmd('run_trial', steps=120, label='baseline'); cid = a['id']
        r1 = await wait_for(lambda m: m.get('type') == 'result' and m.get('id') == cid, 120)
        a = await cmd('dose', drug='tetanus_toxin', amount=1.0)
        a = await cmd('run_trial', steps=120, label='tetanus 1.0'); cid = a['id']
        r2 = await wait_for(lambda m: m.get('type') == 'result' and m.get('id') == cid, 120)
        print(f"baseline: activity {r1['metrics']['activity']:.3f} firing {100*r1['metrics']['firing_all']:.2f}%")
        print(f"tetanus : activity {r2['metrics']['activity']:.3f} firing {100*r2['metrics']['firing_all']:.2f}%")
        results['trials'] = r1['metrics']['steps'] == 120 and r2['metrics']['activity'] < 0.5 * r1['metrics']['activity']
        a = await cmd('restore'); a = await cmd('speed', sps=60)

        # log relay
        await ws.send(json.dumps({'cmd': 'log', 'source': 'check', 'text': 'hello wall'}))
        lg = await wait_for(lambda m: m.get('type') == 'log' and m.get('text') == 'hello wall', 10)
        results['log'] = lg is not None

    print("\n==== SUMMARY ====")
    for k, v in results.items():
        print(f"   {k:20s} {'PASS' if v else 'FAIL'}")
    print("ALL PASS" if all(results.values()) else "NOT YET")

asyncio.run(main())
