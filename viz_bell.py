from __future__ import annotations
import numpy as np

# --- Colors (ASCII names exported) ---
COLOR_AA = "#2b6cb0"   # blue
COLOR_Aa = "#ed8936"   # orange (heterozygote)
COLOR_aa = "#2f855a"   # green (ĀĀ)

# Optional: Unicode aliases so either naming scheme works everywhere
COLOR_AĀ = COLOR_Aa
COLOR_ĀĀ = COLOR_aa


class BellView:
    """Phenotype bell view with fixed genotype means: ĀĀ=0, AĀ=0.5, AA=1."""
    def __init__(self, ax_geno, bins: int = 22, sigma: float = 0.08):
        self.ax = ax_geno
        self.bins = bins
        self.sigma = sigma
        self._pdf_x = np.linspace(0, 1, 400)
        self._edges = np.linspace(0, 1, bins+1)
        self._bars = None
        self._pdf_line = None
        self._v_aa = None
        self._v_Aa = None
        self._v_AA = None
        self._init_artists()

    def _init_artists(self):
        widths = np.diff(self._edges)
        self._bars = self.ax.bar(
            self._edges[:-1],
            np.zeros_like(widths),
            width=widths,
            align='edge',
            alpha=0.25,
            edgecolor="#4a5568",
            linewidth=0.5,
            visible=False,
            zorder=1,
        )
        self._pdf_line, = self.ax.plot(
            self._pdf_x, np.zeros_like(self._pdf_x),
            lw=2, visible=False, zorder=3
        )

        # vertical reference lines at genotype means (use ASCII names)
        self._v_aa = self.ax.axvline(0.0, color=COLOR_aa, lw=1.2, alpha=0.6, visible=False, zorder=2)
        self._v_Aa = self.ax.axvline(0.5, color=COLOR_Aa, lw=1.2, alpha=0.6, visible=False, zorder=2)
        self._v_AA = self.ax.axvline(1.0, color=COLOR_AA, lw=1.2, alpha=0.6, visible=False, zorder=2)

        self.ax.set_xlim(-0.02, 1.02)

    @staticmethod
    def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        inv = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        z = (x - mu) / sigma
        return inv * np.exp(-0.5 * z * z)

    def mixture_pdf(self, weights: np.ndarray) -> np.ndarray:
        """weights = [fAA, fAĀ, fĀĀ] (or [fAA,fAa,faa]); means: 1.0, 0.5, 0.0"""
        sigma = self.sigma
        pdf = (weights[0] * self._normal_pdf(self._pdf_x, 1.0, sigma) +
               weights[1] * self._normal_pdf(self._pdf_x, 0.5, sigma) +
               weights[2] * self._normal_pdf(self._pdf_x, 0.0, sigma))
        area = np.trapz(pdf, self._pdf_x)
        if area > 0:
            pdf = pdf / area
        return pdf

    def sample_phenotypes(self, counts: np.ndarray, rng) -> np.ndarray:
        """Sample N phenotypes with fixed means [1.0, 0.5, 0.0] and common sigma."""
        N = int(counts.sum())
        if N <= 0:
            return np.empty(0, dtype=float)
        mus = np.array([1.0, 0.5, 0.0])  # AA, AĀ, ĀĀ
        phen = np.empty(N, dtype=float)
        start = 0
        for g in (0, 1, 2):
            c = int(counts[g])
            if c <= 0:
                continue
            seg = rng.normal(loc=mus[g], scale=self.sigma, size=c)
            np.clip(seg, 0.0, 1.0, out=seg)
            phen[start:start+c] = seg
            start += c
        return phen[:start]

    def set_visible(self, vis: bool):
        for b in self._bars:
            b.set_visible(vis)
        self._pdf_line.set_visible(vis)
        self._v_aa.set_visible(vis)
        self._v_Aa.set_visible(vis)
        self._v_AA.set_visible(vis)

    def update(self, counts: np.ndarray, rng):
        """Update histogram + mixture curve from genotype counts (AA, AĀ, ĀĀ)."""
        N = counts.sum()
        if N == 0:
            for b in self._bars:
                b.set_height(0.0)
            self._pdf_line.set_ydata(np.zeros_like(self._pdf_x))
            self.ax.set_ylim(0, 1.0)
            return

        fAA, fAĀ, fĀĀ = counts[0]/N, counts[1]/N, counts[2]/N

        phen = self.sample_phenotypes(counts, rng)
        heights, _ = np.histogram(phen, bins=self._edges, range=(0, 1), density=True)
        for bar, hgt in zip(self._bars, heights):
            bar.set_height(hgt)

        pdf = self.mixture_pdf(np.array([fAA, fAĀ, fĀĀ]))
        self._pdf_line.set_ydata(pdf)

        y_max = 0.0
        if heights.size:
            y_max = max(y_max, float(np.max(heights)))
        if pdf.size:
            y_max = max(y_max, float(np.max(pdf)))
        if not np.isfinite(y_max) or y_max <= 0.0:
            y_max = 1.0 / (self.sigma * np.sqrt(2.0 * np.pi))
        self.ax.set_ylim(0, y_max * 1.15)
        self.ax.set_xlim(-0.02, 1.02)
