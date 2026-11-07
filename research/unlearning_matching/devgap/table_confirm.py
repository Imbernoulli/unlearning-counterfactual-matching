"""LaTeX rows for the confirmation table: all methods over the 12 confirmation runs."""
import json, glob, statistics as st
W = '/mnt/datablaze2/bohanlyu/unlearning-20261106/'
F = {'ua': 5.4222, 'ra': 99.9994, 'ta': 94.29, 'mia': 12.7028}
K = ['ua', 'ra', 'ta', 'mia']
runs = {}
def add(path, mapping):
    d = json.load(open(path))
    for rid, r in d['runs'].items():
        for src, dst in mapping.items():
            if src in r:
                runs.setdefault(rid, {})[dst] = r[src]
add(W + 'devgap/confirm/A_k9_sh-1.5_e10/final_test.json', {'retrain': 'Paired Retrain', 'original': 'Original', 'salun': 'SalUn', 'ours': 'A, ours'})
add(W + 'devgap/confirm/A2_dec2_map_sh-1/final_test.json', {'ours': 'A2, ours'})
add(W + 'devgap/confirm/A3_dec2_sh-2/final_test.json', {'ours': 'A3, ours'})
names = {'ga': 'Gradient ascent', 'ft': 'Fine-tuning', 'l1_sparse': '$\\ell_1$-sparse FT', 'rl': 'Random labelling'}
for f in sorted(glob.glob(W + 'baselines_confirm/eval/out*.json')):
    add(f, names)
order = ['Paired Retrain', 'Original', 'Gradient ascent', 'Fine-tuning', '$\\ell_1$-sparse FT', 'Random labelling', 'SalUn', 'A, ours', 'A2, ours', 'A3, ours']
ids = sorted(runs)
for n in order:
    have = [i for i in ids if n in runs[i]]
    if not have:
        print('%', n, 'missing'); continue
    m = {k: st.mean(runs[i][n][k] for i in have) for k in K}
    cells = [f"{m[k]:.2f}" for k in K]
    if n == 'Paired Retrain':
        cells += ['0', '0', '0', '', '']
    else:
        gap = lambda i: sum(abs(runs[i][n][k] - runs[i]['Paired Retrain'][k]) for k in K) / 4
        g = [gap(i) for i in have]; gs = [gap(i) for i in have if i.startswith('s')]; go = [gap(i) for i in have if i.startswith('o')]
        fx = st.mean(sum(abs(runs[i][n][k] - F[k]) for k in K) / 4 for i in have)
        sal = lambda i: sum(abs(runs[i]['SalUn'][k] - runs[i]['Paired Retrain'][k]) for k in K) / 4
        wins = '' if n == 'SalUn' else f"{sum(gap(i) < sal(i) for i in have)}/{len(have)}"
        cells += [f"${st.mean(g):.3f}\\pm{st.stdev(g):.3f}$".replace('0.', '.') if len(g) > 1 else f"{g[0]:.3f}",
                  f"{st.mean(gs):.3f}" if gs else '', f"{st.mean(go):.3f}" if go else '', f"{fx:.3f}", wins]
    note = '' if len(have) == 12 else f' % n={len(have)}'
    print(f"{n} & " + ' & '.join(cells) + '\\\\' + note)
