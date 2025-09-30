# Selection Simulator — Wright–Fisher (Deterministic + Stochastic)

> Interactive, classroom‑ready simulator of single‑locus natural selection with both deterministic and Wright–Fisher (stochastic) dynamics, genotype time series **and** a phenotype **bell‑curve** view. Change presets or fitness mid‑run; the simulation updates next generation without resetting.

<img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/bd3b754d-2795-4e94-bb71-dbe9bef4283d" />
<img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/23182db9-0f3a-4766-81c4-9e22da5636f0" />


---

## Key features

- **Two trajectories at once:** deterministic (infinite‑population) and **stochastic** (Wright–Fisher sampling) are both simulated every generation; a toggle selects which one you **view**.
- **Mid‑run edits:** switch **presets** or modify `s`, `h`, or direct **fitnesses** (`wAA`, `wAĀ`, `wĀĀ`) while running; changes apply from the **next generation** without resetting the population.
- **Two visualization modes:**
  - **Time series:** allele frequency `p(A)`, genotype frequencies (AA, AĀ, ĀĀ), mean fitness `w̄`, and Δp per generation.
  - **Bell curve (phenotype):** mixture of three Gaussians (means **ĀĀ = 0**, **AĀ = 0.5**, **AA = 1**) with a histogram of sampled phenotypes.
- **Arena of individuals:** jittered grid of colored circles (AA=blue, AĀ=orange, ĀĀ=green) reflects the currently viewed trajectory.
- **Early stop rule:** Run auto‑stops if counts are unchanged **and** `|Δp| < 1e−6` for **10** consecutive generations.
- **Reproducibility:** set a random seed; the stochastic path will be repeatable after **Reset**.
- **Fast & responsive:** vectorized updates; bell‑curve histogram + PDF mixture rendered smoothly.

---

## Notation & model

- **Alleles:** `A` and **`Ā`** (using a macron to avoid the lowercase “recessive” misconception).
- **Genotypes:** `AA`, `AĀ`, `ĀĀ`.
- **Allele frequency:** `p = freq(A)`; `q = 1 − p`.
- **Viability selection:** fitness triplet `(wAA, wAĀ, wĀĀ)` applied before reproduction each generation.
- **Directional selection parameterization:**

\[ w_{AA} = 1 + s, \quad w_{AĀ} = 1 + h\,s, \quad w_{ĀĀ} = 1. \]

- **Interpretation:** `s > 0` favors **A**, `s < 0` favors **Ā`. `h` is **dominance of A in fitness** (not phenotype).

---

## Presets

| Preset | Meaning | Default `h` | Fitness used (directional) |
|---|---|:--:|---|
| **A favored — recessive** | A is beneficial and recessive | 0.0 | `wAA=1+s`, `wAĀ=1`, `wĀĀ=1` |
| **A favored — additive** | A has additive effect | 0.5 | `wAA=1+s`, `wAĀ=1+0.5s`, `wĀĀ=1` |
| **A favored — dominant** | A is beneficial and dominant | 1.0 | `wAA=1+s`, `wAĀ=1+s`, `wĀĀ=1` |
| **Overdominance (AĀ advantage)** | Heterozygote advantage | – | fixed `(0.90, 1.00, 0.80)` |
| **Underdominance (AĀ disadvantage)** | Heterozygote disadvantage | – | fixed `(1.00, 0.90, 1.00)` |
| **Custom (s,h) / (w’s)** | Two modes: use **s,h** *or* direct `w` sliders | varies | see below |

**Custom behavior**  
- **Custom (s,h)**: uses the directional formula above.  
- **Custom (w’s)**: touching any of the `wAA`, `wAĀ`, or `wĀĀ` sliders switches to **direct‑fitness** mode (the `s,h` sliders are dimmed and ignored). Re‑select *Custom* to go back to the `s,h` form.

> Tip: To make **Ā** the beneficial allele, simply set **`s < 0`**. No extra toggle needed.

---

## Views

### Time series
- **p(A):** allele frequency over time for the currently **viewed** trajectory.
- **Genotypes:** `AA`, `AĀ`, `ĀĀ` frequencies.
- **Mean fitness (w̄):** `w̄ = p² wAA + 2pq wAĀ + q² wĀĀ`.
- **Δp:** change per generation (speed and sign).

### Bell curve (phenotype)
- Mixture of **three Gaussians** with **fixed means**: `ĀĀ=0`, `AĀ=0.5`, `AA=1` (keeps heterozygotes visually clear regardless of fitness dominance).
- Histogram is drawn from **N sampled phenotypes** at each generation; line is the normalized **mixture PDF**.

<img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/4ea4fd57-f583-40e3-ad6e-11b7be44e99b" />


---

## Controls

- **Presets:** switch fitness scheme mid‑run (applies next gen; no reset).
- **Mode:** choose **which trajectory to view**: Deterministic or Stochastic (both are always simulated).
- **View:** toggle the Genotype panel between **Time series** and **Bell curve**.
- **p₀ (initial A):** starting `p(A)` used only when you **Reset**.
- **s (selection):** magnitude and sign (`s>0` favors A; `s<0` favors Ā).
- **h (dominance of A in fitness):** sets `wAĀ = 1 + h·s` (active in directional & Custom(s,h)).
- **N:** population size (controls drift strength in stochastic path).
- **wAA, wAĀ, wĀĀ:** visible in **Custom (w’s)**; set fitnesses directly.
- **Gens:** how many generations one **Run** executes from now.
- **Seed:** RNG seed applied on **Reset** for reproducible stochastic runs.
- **Buttons:** **Step** (advance 1), **Run** (animate), **Stop** (pause), **Reset** (rebuild from `p₀` and `N`).

**Early stop:** Run halts automatically if **counts are identical and `|Δp| < 1e−6` for 10 consecutive generations** in the viewed trajectory.

---

## Installation

```bash
# Python 3.10+ recommended
pip install numpy matplotlib
```

> On some systems you may need `pip3` and/or a virtual environment.

---

## Running the simulator

```bash
python selection_sim.py
```

The app opens a Matplotlib window. Use the controls at the bottom to explore presets, switch views, and run/step/pause.

---

## Project layout

```
.
├── genetics_core.py     # Hardy–Weinberg, mean fitness, deterministic & WF updates
├── presets.py           # Preset catalogue and fitness computation
├── viz_bell.py          # Bell-curve view (histogram + mixture PDF)
└── selection_sim.py     # UI app: arena, plots, controls, help overlay
```

---

## Teaching notes

- Dominance is encoded in **fitness** (via `h`), not by lowercase/uppercase letters.
- Overdominance (AĀ advantage) maintains a stable internal mix; underdominance creates a threshold toward fixation/extinction depending on initial `p₀`.
- The bell curve visualizes **phenotype means** (0, 0.5, 1) independently of fitness dominance so that heterozygotes remain visible in all scenarios.

---

## Troubleshooting

- **Unicode labels (Ā) render as boxes?** Install a font with macron support (DejaVu Sans is usually fine; Matplotlib includes it by default).
- **Nothing happens when changing `s`/`h` in Over/Under dominance?** Those presets use **fixed** fitnesses; `s,h` are ignored by design.
- **Preset changes reset my population?** Only **Reset**, changing `p₀`, or changing `N` rebuild the population. Preset/fitness edits apply at the **next generation** without rebuilding.

---


If you use this in teaching or a report, please cite:
> *Selection Simulator — Wright–Fisher (Deterministic + Stochastic).* https://github.com/PurplePoet25/Selection-Simulator-Wright-Fisher-Deterministic-Stochastic-
