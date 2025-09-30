
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.patches import FancyBboxPatch

from genetics_core import mean_w, next_p_det, geno_freqs_after_sel, hw_from_p, next_p_stoch
from presets import PRESETS, compute_w, default_h_for_preset
from viz_bell import BellView, COLOR_AA, COLOR_Aa, COLOR_aa

BG_FACE   = "#f7fafc"

plt.rcParams.update({
    "figure.facecolor": BG_FACE,
    "axes.facecolor": "white",
    "figure.figsize": (14.5, 9.5),
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "toolbar": "toolbar2",
})

class SelectionSim:
    def __init__(self):
        # State
        self.p0 = 0.10
        self.s = 0.05
        self.h = 0.50                 # fitness dominance (for directional/custom s,h)
        self.N = 120
        self.preset = PRESETS[0]
        self.custom_ws = (1.0, 1.0, 1.0)
        self.custom_mode = "sh"       # "sh" or "w"
        self.max_gens = 300
        self.rng = np.random.default_rng()
        self.gen = 0
        self.target_gen = self.max_gens
        self.running = False
        self.mode = "Stochastic"      # which trajectory to show
        self.view_mode = "Time series"

        # Early stop
        self.stability_window = 10
        self.stability_tol = 1e-6

        # Trajectories
        self.S = {
            "Deterministic": self._init_traj(self.p0, det=True),
            "Stochastic":   self._init_traj(self.p0, det=False),
        }

        self._build_ui()

    # ---- Trajectories ----
    def _init_traj(self, p_start: float, det: bool):
        fAA0, fAĀ0, fĀĀ0 = hw_from_p(p_start)
        if det:
            expected = np.array([fAA0, fAĀ0, fĀĀ0], dtype=float) * self.N
            base = np.floor(expected).astype(int)
            short = self.N - base.sum()
            if short > 0:
                rema = expected - base
                base[np.argsort(-rema)[:short]] += 1
            counts0 = base
            genotypes0 = np.repeat([0,1,2], counts0)
        else:
            counts0 = self.rng.multinomial(self.N, [fAA0, fAĀ0, fĀĀ0])
            genotypes0 = np.repeat([0,1,2], counts0)

        w = compute_w(self.preset, self.s, self.h, self.custom_ws, self.custom_mode)
        return {
            "p": p_start,
            "p_hist": [p_start],
            "fAA_hist": [counts0[0]/self.N],
            "fAĀ_hist": [counts0[1]/self.N],
            "fĀĀ_hist": [counts0[2]/self.N],
            "wbar_hist": [mean_w(p_start, w.wAA, w.wAĀ, w.wĀĀ)],
            "delp_hist": [0.0],
            "counts_hist": [counts0.copy()],
            "genotypes": genotypes0,
        }

    def _advance_one_mode(self, mode_key: str):
        S = self.S[mode_key]
        p = S["p"]
        w = compute_w(self.preset, self.s, self.h, self.custom_ws, self.custom_mode)

        prev_p = p
        if mode_key == "Deterministic":
            p = next_p_det(p, w.wAA, w.wAĀ, w.wĀĀ)
            fAA, fAĀ, fĀĀ = hw_from_p(p)
            expected = np.array([fAA, fAĀ, fĀĀ], dtype=float) * self.N
            base = np.floor(expected).astype(int)
            short = self.N - base.sum()
            if short > 0:
                rema = expected - base
                base[np.argsort(-rema)[:short]] += 1
            counts = base
        else:
            p, counts = next_p_stoch(p, w.wAA, w.wAĀ, w.wĀĀ, self.N, self.rng)
            fAA, fAĀ, fĀĀ = counts[0]/self.N, counts[1]/self.N, counts[2]/self.N

        S["p"] = p
        S["p_hist"].append(p)
        S["fAA_hist"].append(fAA)
        S["fAĀ_hist"].append(fAĀ)
        S["fĀĀ_hist"].append(fĀĀ)
        S["wbar_hist"].append(mean_w(p, w.wAA, w.wAĀ, w.wĀĀ))
        S["delp_hist"].append(p - prev_p)
        S["counts_hist"].append(counts.copy())
        S["genotypes"] = np.repeat([0,1,2], counts)

    # ---- UI ----
    def _build_ui(self):
        self.fig = plt.figure(constrained_layout=True)
        outer = GridSpec(2, 1, height_ratios=[0.82, 0.18], figure=self.fig)
        top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.6])
        right = top[1].subgridspec(2, 2)

        self.ax_arena = self.fig.add_subplot(top[0, 0])
        self.ax_p     = self.fig.add_subplot(right[0, 0])
        self.ax_geno  = self.fig.add_subplot(right[0, 1])
        self.ax_wbar  = self.fig.add_subplot(right[1, 0])
        self.ax_delp  = self.fig.add_subplot(right[1, 1])

        self.ax_controls_bg = self.fig.add_subplot(outer[1, 0])
        self.ax_controls_bg.axis("off")
        self.ax_controls_bg.set_facecolor("#ffffff")

        self._init_arena()
        self._init_plots()
        self._init_controls()
        self._build_help_overlay()

        try:
            self.fig.canvas.manager.set_window_title("Selection Simulator — Modular v1")
        except Exception:
            pass

        self._redraw_all()
        plt.show()

    # Arena
    def _init_arena(self):
        self.ax_arena.set_title("Population (circles): AA (blue), AĀ (orange), ĀĀ (green)")
        self.ax_arena.set_xticks([]); self.ax_arena.set_yticks([])
        self.ax_arena.set_xlim(0, 1); self.ax_arena.set_ylim(0, 1)

        cols = int(np.ceil(np.sqrt(self.N)))
        rows = int(np.ceil(self.N/cols))
        jitter = 0.016
        xs, ys = [], []
        for i in range(self.N):
            r = i // cols; c = i % cols
            xs.append((c+0.5)/cols + np.random.uniform(-jitter, jitter))
            ys.append((rows-r-0.5)/rows + np.random.uniform(-jitter, jitter))
        self.pos = np.column_stack([xs, ys])

        self.scat = self.ax_arena.scatter(self.pos[:,0], self.pos[:,1],
                                          s=90, edgecolor="white", linewidth=0.6)
        self.scat.set_edgecolors("white")
        self._update_arena_colors()

        self.legend_txt = self.ax_arena.text(
            0.02, 0.02, f"Preset: {self.preset}",
            ha="left", va="bottom", fontsize=10,
            bbox=dict(facecolor="#edf2f7", edgecolor="none", boxstyle="round,pad=0.3")
        )

    def _update_arena_colors(self):
        S = self.S[self.mode]
        genotypes = S["genotypes"]
        colors = np.where(genotypes==0, COLOR_AA,
                 np.where(genotypes==1, COLOR_Aa, COLOR_aa))
        self.scat.set_facecolors(colors)

    # Plots
    
    def _init_plots(self):
        self.ax_p.set_xlabel("Generation")
        self.ax_p.set_ylabel("p")
        self.line_p, = self.ax_p.plot([], [], lw=2)
        self.title_p = self.ax_p.set_title("")

        self.ax_geno.set_title("Genotype frequencies")
        self.ax_geno.set_xlabel("Generation")
        self.ax_geno.set_ylabel("Frequency")
        self.line_fAA, = self.ax_geno.plot([], [], color=COLOR_AA, label="AA", lw=2)
        self.line_fAĀ, = self.ax_geno.plot([], [], color=COLOR_Aa, label="AĀ", lw=2)
        self.line_fĀĀ, = self.ax_geno.plot([], [], color=COLOR_aa, label="ĀĀ", lw=2)
        self.ax_geno.legend(loc="upper right", frameon=False)

        # Bell view
        self.bell = BellView(self.ax_geno, bins=22, sigma=0.08)

        # Mean fitness
        self.ax_wbar.set_title("Mean fitness (w̄)")
        self.ax_wbar.set_xlabel("Generation")
        self.ax_wbar.set_ylabel("w̄")
        self.line_wbar, = self.ax_wbar.plot([], [], lw=2)

        # Δp per generation
        self.ax_delp.set_title("Δp per generation (speed of change)")
        self.ax_delp.set_xlabel("Generation")
        self.ax_delp.set_ylabel("Δp")
        self.line_delp, = self.ax_delp.plot([], [], lw=2)


    def _init_controls(self):
        fig = self.fig

        self.ax_presets = fig.add_axes([0.052, 0.074, 0.21, 0.12])
        self.rb_presets = RadioButtons(self.ax_presets, PRESETS, active=0)
        self.rb_presets.on_clicked(self._on_preset)

        self.ax_mode = fig.add_axes([0.262, 0.074, 0.086, 0.12])
        self.rb_mode = RadioButtons(self.ax_mode, ("Deterministic", "Stochastic"))
        self.rb_mode.set_active(1)
        self.rb_mode.on_clicked(self._on_mode)

        self.ax_view = fig.add_axes([0.347, 0.074, 0.073, 0.12])
        self.rb_view = RadioButtons(self.ax_view, ("Time series", "Bell curve"))
        self.rb_view.set_active(0 if self.view_mode == "Time series" else 1)
        self.rb_view.on_clicked(self._on_view)

        self.ax_p0 = fig.add_axes([0.5, 0.15, 0.32, 0.04])
        self.sl_p0 = Slider(self.ax_p0, "p (initial A)", 0.0, 1.0, valinit=self.p0, valstep=0.01)
        self.sl_p0.on_changed(self._on_p0)

        self.ax_s  = fig.add_axes([0.5, 0.10, 0.32, 0.04])
        self.sl_s  = Slider(self.ax_s, "s (selection)", -0.2, 0.2, valinit=self.s, valstep=0.005)
        self.sl_s.on_changed(self._on_s)

        self.ax_h  = fig.add_axes([0.5, 0.05, 0.32, 0.04])
        self.sl_h  = Slider(self.ax_h, "h (dominance of A)", 0.0, 1.0, valinit=self.h, valstep=0.01)
        self.sl_h.on_changed(self._on_h)

        self.ax_N  = fig.add_axes([0.5, 0.00, 0.32, 0.04])
        self.sl_N  = Slider(self.ax_N, "N (population size)", 20, 600, valinit=self.N, valstep=2)
        self.sl_N.on_changed(self._on_N)

        # Custom w sliders (hidden until Custom/w selected)
        self.ax_wAA = fig.add_axes([0.88, 0.10, 0.1, 0.04])
        self.sl_wAA = Slider(self.ax_wAA, "wAA", 0.0, 2.0, valinit=1.0, valstep=0.01)
        self.ax_wAĀ = fig.add_axes([0.88, 0.05, 0.1, 0.04])
        self.sl_wAĀ = Slider(self.ax_wAĀ, "wAĀ", 0.0, 2.0, valinit=1.0, valstep=0.01)
        self.ax_wĀĀ = fig.add_axes([0.88, 0.00, 0.1, 0.04])
        self.sl_wĀĀ = Slider(self.ax_wĀĀ, "wĀĀ", 0.0, 2.0, valinit=1.0, valstep=0.01)
        for sl in (self.sl_wAA, self.sl_wAĀ, self.sl_wĀĀ):
            sl.on_changed(self._on_custom_ws)
        self._toggle_custom_ws(False)

        # Buttons + textboxes
        self.ax_step  = fig.add_axes([0.052, 0.015, 0.08, 0.06])
        self.ax_run   = fig.add_axes([0.132, 0.015, 0.08, 0.06])
        self.ax_stop  = fig.add_axes([0.212, 0.015, 0.08, 0.06])
        self.ax_reset = fig.add_axes([0.292, 0.015, 0.08, 0.06])
        self.ax_help  = fig.add_axes([0.002, 0.014, 0.05, 0.18])

        self.bt_step  = Button(self.ax_step,  "Step")
        self.bt_run   = Button(self.ax_run,   "Run")
        self.bt_stop  = Button(self.ax_stop,  "Stop")
        self.bt_reset = Button(self.ax_reset, "Reset")
        self.bt_help  = Button(self.ax_help,  "Help")

        self.bt_step.on_clicked(self._on_step)
        self.bt_run.on_clicked(self._on_run)
        self.bt_stop.on_clicked(self._on_stop)
        self.bt_reset.on_clicked(self._on_reset)
        self.bt_help.on_clicked(self._on_help)

        self.ax_gens = fig.add_axes([0.90, 0.15, 0.03, 0.02])
        self.tb_gens = TextBox(self.ax_gens, "Gens", initial=str(self.max_gens))
        self.ax_seed = fig.add_axes([0.96, 0.15, 0.03, 0.02])
        self.tb_seed = TextBox(self.ax_seed, "Seed", initial="")

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self._tick)

    def _toggle_custom_ws(self, visible: bool):
        for ax in (self.ax_wAA, self.ax_wAĀ, self.ax_wĀĀ):
            ax.set_visible(visible)
        self.fig.canvas.draw_idle()

    def _build_help_overlay(self):
        """Comprehensive, two-column Help overlay with clear headings and legend."""
        fig = self.fig

        # === Panel placement (figure coords) ===
        W, H = 0.78, 1.12
        X, Y = (1.0 - W) / 2.0, (1.0 - H) / 2.0

        # An axes that lives on top of everything, used only for layout
        self.help_ax = fig.add_axes([X, Y, W, H], zorder=200)
        self.help_ax.set_axis_off()

        # Rounded white panel
        from matplotlib.patches import FancyBboxPatch, Rectangle
        self.help_panel = FancyBboxPatch(
            (0, 0), 1, 1,
            transform=self.help_ax.transAxes,
            boxstyle="round,pad=0.6",
            facecolor="white", edgecolor="#2d3748",
            linewidth=1.3, alpha=0.98, zorder=201
        )
        self.help_ax.add_patch(self.help_panel)

        # We'll collect all text/markers here to toggle visibility at once
        self._help_artists = []

        # Column layout (axes coords)
        Lx, Rx = 0.05, 0.54      # left and right column x
        Wcol   = 0.40
        y_top  = 0.95            # start near top; we’ll step downward

        def section(ax, x, y, title):
            t = ax.text(x, y, title, transform=ax.transAxes,
                        fontsize=12.5, fontweight="bold", color="#1a202c",
                        va="top", ha="left", zorder=202)
            self._help_artists.append(t)
            return y - 0.035

        def bullets(ax, x, y, lines, dy=0.028, mono=False):
            for s in lines:
                t = ax.text(x, y, s, transform=ax.transAxes,
                            fontsize=10.6, color="#1a202c",
                            va="top", ha="left", zorder=202,
                            fontfamily=("monospace" if mono else None))
                self._help_artists.append(t)
                y -= dy
            return y - 0.01

        # ========== LEFT COLUMN ==========
        y = y_top
        y = section(self.help_ax, Lx, y, "Selection Simulator — Help")

        y = section(self.help_ax, Lx, y, "Model (Wright–Fisher with selection)")
        y = bullets(self.help_ax, Lx, y, [
            "• One diploid locus; alleles A and Ā.  p = freq(A),  q = 1−p.",
            "• Viability selection with genotype fitnesses (wAA, wAĀ, wĀĀ).",
            "• Directional (textbook) parameterization:",
            "    wAA = 1 + s,   wAĀ = 1 + h·s,   wĀĀ = 1",  # monospace next:
        ], dy=0.026)
        y = bullets(self.help_ax, Lx+0.02, y, [
            "s > 0  → A favored      s < 0  → Ā favored",
        ], dy=0.026, mono=True)
        y = bullets(self.help_ax, Lx, y, [
            "• Two parallel trajectories updated every generation:",
            "  – Deterministic: infinite-population equations (no drift).",
            "  – Stochastic: Wright–Fisher sampling of N individuals (drift).",
        ])

        y = section(self.help_ax, Lx, y, "Presets — what they mean")
        y = bullets(self.help_ax, Lx, y, [
            "• A favored — recessive:   default h=0    → wAA=1+s, wAĀ=1,      wĀĀ=1.",
            "• A favored — additive:    default h=0.5  → wAA=1+s, wAĀ=1+0.5s, wĀĀ=1.",
            "• A favored — dominant:    default h=1    → wAA=1+s, wAĀ=1+s,    wĀĀ=1.",
            "  (Directional presets use your live s and h; the preset only snaps h.)",
            "• Overdominance (heterozygote advantage): fixed (0.90, 1.00, 0.80).",
            "• Underdominance (heterozygote disadvantage): fixed (1.00, 0.90, 1.00).",
            "  (In those two, s and h are ignored.)",
            "• Custom (s,h) / (w’s):",
            "  – Custom (s,h): uses wAA=1+s, wAĀ=1+h·s, wĀĀ=1.",
            "  – Custom (w’s): touch any w-slider (wAA, wAĀ, wĀĀ) to switch to",
            "    direct-fitness mode (s,h dimmed). Re-select Custom to go back to (s,h).",
        ])

        y = section(self.help_ax, Lx, y, "Controls")
        y = bullets(self.help_ax, Lx, y, [
            "• p₀ (initial A): starting allele-A frequency (used on Reset).",
            "• s (selection): magnitude & sign (s>0 favors A; s<0 favors Ā).",
            "• h (dominance of A in FITNESS): sets wAĀ = 1 + h·s.",
            "  – Active in directional presets & Custom(s,h).",
            "  – Greyed in Custom(w’s) and fixed Over/Under presets.",
            "• N: population size (controls drift strength in stochastic path).",
            "• Preset: choose fitness scheme (applies next gen; no reset).",
            "• Mode: choose which trajectory to VIEW (Deterministic / Stochastic).",
            "• View: switch Genotype panel between Time series and Bell curve.",
            "• w-sliders: visible in Custom(w’s); directly set wAA, wAĀ, wĀĀ.",
            "• Gens: how many generations the next Run will advance from ‘now’.",
            "• Seed: RNG seed (applies on Reset for repeatable stochastic runs).",
            "• Buttons: Step (advance 1), Run (animate), Stop (pause),",
            "           Reset (rebuild population from p₀ & N; reseed if set).",
        ])

        # ========== RIGHT COLUMN ==========
        yR = y_top

        # Small in-panel legend (colored markers)
        yR = section(self.help_ax, Rx, yR, "Legend & arena")
        # color dots
        dot_y = yR - 0.01
        for i, (lab, col) in enumerate([("AA", COLOR_AA), ("AĀ", COLOR_Aa), ("ĀĀ", COLOR_aa)]):
            dx = Rx + 0.01 + i*0.12
            m = self.help_ax.scatter([dx], [dot_y], s=70, c=[col], marker='o',
                                    zorder=203, transform=self.help_ax.transAxes)
            self._help_artists.append(m)
            tt = self.help_ax.text(dx+0.03, dot_y, lab,
                                transform=self.help_ax.transAxes, va="center", ha="left",
                                fontsize=10.8, color="#1a202c", zorder=203)
            self._help_artists.append(tt)
        yR -= 0.07
        yR = bullets(self.help_ax, Rx, yR, [
            "• Left panel shows N individuals as circles in a jittered grid.",
            "• Colors reflect genotypes in the currently VIEWED path.",
        ])

        yR = section(self.help_ax, Rx, yR, "Plots (Time series view)")
        yR = bullets(self.help_ax, Rx, yR, [
            "• p(A): allele A frequency over time.",
            "• Genotype frequencies: AA, AĀ, ĀĀ for the viewed path.",
            "• Mean fitness (w̄): population mean each generation.",
            "• Δp: change in p per generation (speed & sign).",
        ])

        yR = section(self.help_ax, Rx, yR, "Bell curve (phenotype view)")
        yR = bullets(self.help_ax, Rx, yR, [
            "• Three peaks at fixed means: ĀĀ=0, AĀ=0.5, AA=1 (independent of h).",
            "• Histogram = density of synthetic phenotypes sampled by current",
            "  genotype counts; smooth curve = normalized 3-Gaussian mixture.",
        ])

        yR = section(self.help_ax, Rx, yR, "Run logic")
        yR = bullets(self.help_ax, Rx, yR, [
            "• Changing preset / s / h / w applies next generation (no reset).",
            "• Early stop: Stop Run if counts identical and |Δp| < 1e−6 for 10",
            "  consecutive generations in the viewed path.",
            "• Reset only when you press Reset or change p₀ / N.",
        ])

        yR = section(self.help_ax, Rx, yR, "Tips")
        yR = bullets(self.help_ax, Rx, yR, [
            "• Make Ā beneficial by setting s < 0 (no extra toggle needed).",
            "• ‘Recessive’ ≈ h≈0 ; ‘additive’ ≈ h≈0.5 ; ‘dominant’ ≈ h≈1.",
            "• Overdominance keeps a stable internal mix; underdominance shows",
            "  a threshold toward fixation.",
        ])

        # Keep a tiny title text handle so old code doesn't break (not used to render body)
        self.help_title = self.help_ax.text(Lx, y_top+0.008, "", transform=self.help_ax.transAxes, alpha=0.0)
        self.help_box  = self.help_panel  # for backward-compat with _on_help using .get_visible()
        self.help_txt  = self.help_title  # dummy handle; real content is in _help_artists

        # Start hidden
        self._set_help_visible(False)


    # ---- Callbacks ----
    def _apply_controls_to_state(self) -> bool:
        self.p0 = float(self.sl_p0.val)
        self.s = float(self.sl_s.val)
        self.h = float(self.sl_h.val)
        newN = int(self.sl_N.val)
        n_changed = (newN != self.N)
        self.N = newN
        self.mode = str(self.rb_mode.value_selected)
        self.view_mode = str(self.rb_view.value_selected)
        return n_changed

    def _promote_to_custom_sh(self):
        if self.preset != "Custom (s,h) / (w's)":
            # set radio to Custom
            idx = PRESETS.index("Custom (s,h) / (w's)")
            self.rb_presets.set_active(idx)
            self.preset = "Custom (s,h) / (w's)"
        self.custom_mode = "sh"
        self._toggle_custom_ws(False)
        self.legend_txt.set_text(f"Preset: {self.preset}")

    def _promote_to_custom_ws(self):
        if self.preset != "Custom (s,h) / (w's)":
            idx = PRESETS.index("Custom (s,h) / (w's)")
            self.rb_presets.set_active(idx)
            self.preset = "Custom (s,h) / (w's)"
        self.custom_mode = "w"
        self._toggle_custom_ws(True)
        self.legend_txt.set_text(f"Preset: {self.preset}")

    def _on_preset(self, label: str):
        self.preset = label
        self.legend_txt.set_text(f"Preset: {self.preset}")

        if label == "Custom (s,h) / (w's)":
            # initialize custom w's from current s,h (wAA=1+s, wAĀ=1+h*s, wĀĀ=1)
            wAA = 1.0 + self.s
            wAĀ = 1.0 + self.h * self.s
            wĀĀ = 1.0
            self.custom_ws = (wAA, wAĀ, wĀĀ)

            # push values into sliders so UI matches engine
            self.sl_wAA.set_val(wAA)
            self.sl_wAĀ.set_val(wAĀ)
            self.sl_wĀĀ.set_val(wĀĀ)

            # use the direct-w mode & show the controls
            self.custom_mode = "w"
            self._toggle_custom_ws(True)

            # (optional) gray out s,h while in direct-w mode
            self._set_sh_active(False)
        else:
            # directional or balanced presets
            self.custom_mode = "sh"
            self._toggle_custom_ws(False)
            self._set_sh_active(True)

            # set canonical h for directional presets (if applicable)
            h0 = default_h_for_preset(label)
            if h0 is not None:
                self.sl_h.set_val(h0)

        self._redraw_all()

    def _on_mode(self, label: str):
        self.mode = label
        self._redraw_all()

    def _on_view(self, label: str):
        self.view_mode = str(label)
        if self.view_mode == "Time series":
            self.bell.set_visible(False)
            self._set_timeseries_visible(True)
            self.ax_geno.set_title("Genotype frequencies over time")
            self.ax_geno.set_xlabel("Generation")
            self.ax_geno.set_ylabel("Frequency")
        else:
            self._set_timeseries_visible(False)
            self.bell.set_visible(True)
            self.ax_geno.set_title("Population distribution (μ: ĀĀ=0, AĀ=0.5, AA=1)")
            self.ax_geno.set_xlabel("Trait value (0 … 1)")
            self.ax_geno.set_ylabel("Density")
        self._redraw_all()

    def _on_p0(self, val):
        self.p0 = float(val)
        self._reset_live(force_rebuild=False)

    def _on_s(self, val):
        self.s = float(val)
        # If preset is Over/Under, s does nothing but we still allow viewing effect when switching back
        if self.preset != "Custom (s,h) / (w's)":
            self.custom_mode = "sh"
        self._redraw_all()

    def _on_h(self, val):
        self.h = float(val)
        # If the preset is a directional one and h moved away from its default, promote to Custom(s,h)
        h0 = default_h_for_preset(self.preset)
        if h0 is not None and abs(self.h - h0) > 1e-9:
            self._promote_to_custom_sh()
        self._redraw_all()

    def _on_N(self, val):
        self.N = int(val)
        self._reset_live(force_rebuild=True)
    
    def _set_sh_active(self, active: bool):
        # disable/enable the s,h sliders so the UI reflects which inputs are "live"
        try:
            self.sl_s.set_active(active)
            self.sl_h.set_active(active)
            # dim the labels a bit for visual feedback
            alpha = 1.0 if active else 0.35
            self.sl_s.ax.set_alpha(alpha)
            self.sl_h.ax.set_alpha(alpha)
        except Exception:
            pass


    def _on_custom_ws(self, _):
        self.custom_ws = (self.sl_wAA.val, self.sl_wAĀ.val, self.sl_wĀĀ.val)
        self._promote_to_custom_ws()   # keeps custom_mode="w" and the sliders visible
        self._set_sh_active(False)     # (optional) gray out s,h
        self._redraw_all()

    def _on_step(self, _):
        self._advance_both()
        self._redraw_all()

    def _on_run(self, _):
        self._update_max_gens_from_textbox()
        self.target_gen = self.gen + self.max_gens
        if not self.running:
            self.running = True
            self.timer.start()

    def _on_stop(self, _):
        self.running = False
        self.timer.stop()

    def _on_reset(self, _):
        self._reset_live(force_rebuild=True, from_button=True)

    def _on_help(self, _):
        # Toggle based on the panel's current visibility
        vis_now = getattr(self.help_panel, "get_visible", lambda: False)()
        self._set_help_visible(not vis_now)


    # ---- Engine loop ----
    def _tick(self):
        if not self.running: return
        for _ in range(2):
            self._advance_both()
            if self._should_early_stop(self.mode):
                self.running = False
                break
            if self.gen >= self.target_gen:
                self.running = False
                break
        self._redraw_all()

    def _advance_both(self):
        self._advance_one_mode("Deterministic")
        self._advance_one_mode("Stochastic")
        self.gen += 1

    def _update_max_gens_from_textbox(self):
        try:
            self.max_gens = max(1, min(50000, int(self.tb_gens.text.strip())))
        except Exception:
            pass

    def _reset_live(self, force_rebuild: bool, from_button: bool=False):
        self._on_stop(None)

        n_changed = self._apply_controls_to_state()
        rebuild = force_rebuild or n_changed

        self._update_max_gens_from_textbox()
        seed_txt = self.tb_seed.text.strip()
        if seed_txt:
            try:
                self.rng = np.random.default_rng(int(seed_txt))
            except Exception:
                pass

        if rebuild:
            self.ax_arena.cla()
            self._init_arena()

        # Rebuild trajectories from p0, N
        self.gen = 0
        self.S = {
            "Deterministic": self._init_traj(self.p0, det=True),
            "Stochastic":   self._init_traj(self.p0, det=False),
        }
        self.target_gen = self.gen + self.max_gens
        self._redraw_all()

    # ---- Helpers ----
    def _set_timeseries_visible(self, vis: bool):
        for artist in (self.line_fAA, self.line_fAĀ, self.line_fĀĀ):
            artist.set_visible(vis)
        leg = getattr(self.ax_geno, "legend_", None)
        if leg is not None: leg.set_visible(vis)

    def _should_early_stop(self, mode_key: str) -> bool:
        S = self.S[mode_key]
        w = int(self.stability_window)
        if len(S["p_hist"]) < w + 1:
            return False
        recent_delp = np.abs(np.diff(S["p_hist"][-(w+1):]))
        if np.all(recent_delp < self.stability_tol):
            recent_counts = S["counts_hist"][-w:]
            base = recent_counts[0]
            same_counts = all(np.array_equal(rc, base) for rc in recent_counts[1:])
            return same_counts
        return False
    
    def _set_help_visible(self, vis: bool):
        """Show/hide the entire Help overlay."""
        if hasattr(self, "help_ax"):
            self.help_ax.set_visible(vis)
        if hasattr(self, "help_panel"):
            self.help_panel.set_visible(vis)
        if hasattr(self, "_help_artists"):
            for a in self._help_artists:
                a.set_visible(vis)
        self.fig.canvas.draw_idle()


    # ---- Draw ----
    def _redraw_all(self):
        # Arena
        self._update_arena_colors()

        # Displayed trajectory
        S = self.S[self.mode]
        xs = np.arange(len(S["p_hist"]))

        # p(t)
        self.line_p.set_data(xs, S["p_hist"])
        self.ax_p.set_xlim(0, max(10, len(xs)-1))
        self.ax_p.set_ylim(0, 1)
        self.title_p.set_text(f"Allele frequency p (A) — gen {self.gen} | mode={self.mode} | p={S['p']:.3f}")

        # Genotype panel
        if self.view_mode == "Time series":
            self._set_timeseries_visible(True)
            self.bell.set_visible(False)
            self.line_fAA.set_data(xs, S["fAA_hist"])
            self.line_fAĀ.set_data(xs, S["fAĀ_hist"])
            self.line_fĀĀ.set_data(xs, S["fĀĀ_hist"])
            self.ax_geno.set_xlim(0, max(10, len(xs)-1))
            self.ax_geno.set_ylim(0, 1)
        else:
            self._set_timeseries_visible(False)
            self.bell.set_visible(True)
            self.bell.update(S["counts_hist"][-1], self.rng)

        
        # Mean fitness
        self.line_wbar.set_data(xs, S["wbar_hist"])
        self.ax_wbar.set_xlim(0, max(10, len(xs)-1))
        if S["wbar_hist"]:
            mn, mx = min(S["wbar_hist"]), max(S["wbar_hist"])
            pad = max(0.005, 0.10*(mx-mn if mx>mn else 1.0))
            self.ax_wbar.set_ylim(max(0, mn-pad), mx+pad)

        # Δp
        self.line_delp.set_data(xs, S["delp_hist"])
        self.ax_delp.set_xlim(0, max(10, len(xs)-1))
        if len(S["delp_hist"]) > 1:
            a = max(abs(min(S["delp_hist"])), abs(max(S["delp_hist"])))
            a = max(a, 0.01)
            self.ax_delp.set_ylim(-a*1.2, a*1.2)
        else:
            self.ax_delp.set_ylim(-0.02, 0.02)


        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    SelectionSim()
