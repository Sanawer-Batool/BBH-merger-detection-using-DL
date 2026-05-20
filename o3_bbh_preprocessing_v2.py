
# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Scientific stack ──────────────────────────────────────────────────────────
import numpy as np
import torch

# ── GW libraries ──────────────────────────────────────────────────────────────
try:
    from gwpy.timeseries import TimeSeries
    from gwosc.datasets import event_gps
except ImportError:
    raise ImportError("Run:  pip install gwpy gwosc")

# ── Plotting (optional) ───────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # no GUI needed
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
#  O3 CONFIRMED BBH EVENTS  (from GWTC-3, all on GWOSC)
# ══════════════════════════════════════════════════════════════════════════════

O3_BBH_EVENTS = {
    # name          : (detector_1, detector_2)
    "GW190521"      : ("H1", "L1"),   # heaviest BBH in O3 (~150 solar masses total)
    "GW190814"      : ("H1", "L1"),   # most asymmetric mass ratio
    "GW190412"      : ("H1", "L1"),   # first unequal-mass detection
    "GW191109_010717": ("H1", "L1"),  # high-spin event
}

DEFAULT_EVENT    = "GW190521"
DEFAULT_DURATION = 16          # seconds  (8 s each side of merger)
SAMPLE_RATE      = 2048        # Hz  (standard for BBH pipelines)
HIGHPASS_HZ      = 20.0        # Hz  lower cutoff (seismic wall)
LOWPASS_HZ       = 512.0       # Hz  upper cutoff (above BBH band)
FFTLENGTH        = 2.0         # seconds  for PSD estimation
OVERLAP          = 1.0         # seconds  PSD overlap
FDURATION        = 2.0         # seconds  whitening filter duration


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 1 : fetch_o3_strain
# ══════════════════════════════════════════════════════════════════════════════

def fetch_o3_strain(
    event_name: str = DEFAULT_EVENT,
    duration:   float = DEFAULT_DURATION,
    target_rate: int  = SAMPLE_RATE,
) -> tuple[torch.Tensor, float]:
    """
    Download real O3 strain data from GWOSC for a confirmed BBH event.

    WHAT IS THIS DOING IN PLAIN ENGLISH?
    -------------------------------------
    GWOSC (Gravitational Wave Open Science Center) is LIGO's public data
    server.  Every confirmed gravitational wave event has its data stored
    there for free.  This function:

      1. Looks up the GPS timestamp of the event  (e.g. GW190521 happened
         at GPS second 1242442967 -- seconds since Jan 6 1980)
      2. Downloads 'duration' seconds of strain around that moment
         from BOTH the H1 (Hanford) and L1 (Livingston) detectors
      3. Resamples both to 2048 Hz  (standard for BBH pipelines)
      4. Stacks them into shape (1, 2, N_samples)
         -- batch=1, channels=2 (one per detector), samples=N

    WHY TWO DETECTORS?
    ------------------
    A real signal must appear in BOTH detectors within ~10 ms (light
    travel time between sites).  A glitch only appears in ONE detector.
    Using both channels lets the model learn coincidence -- a key
    discriminator between signals and noise.

    WHY 16 SECONDS?
    ---------------
    The whitening step needs ~4 s of data on each side to estimate the
    noise power spectrum accurately.  16 s gives plenty of baseline noise
    before and after the 0.2-0.4 s chirp signal.

    Returns
    -------
    tensor     : shape (1, 2, N_samples)  -- batch, detectors, time
    gps_centre : GPS time of the event    -- useful for labelling
    """
    detectors = O3_BBH_EVENTS.get(event_name, ("H1", "L1"))

    print(f"\n[DATA]  Event       : {event_name}")
    print(f"[DATA]  Detectors   : {detectors}")

    # Step 1 -- find GPS time of event
    gps_centre = event_gps(event_name)
    start_gps  = gps_centre - duration / 2
    end_gps    = gps_centre + duration / 2
    print(f"[DATA]  GPS centre  : {gps_centre:.1f}")
    print(f"[DATA]  Window      : {start_gps:.1f} → {end_gps:.1f}  ({duration} s)")

    channels = []
    for det in detectors:
        print(f"[DATA]  Downloading {det} ...", end=" ", flush=True)

        # Step 2 -- download from GWOSC (caches locally after first run)
        strain = TimeSeries.fetch_open_data(det, start_gps, end_gps, cache=True)

        # Step 3 -- resample to standard rate
        if int(strain.sample_rate.value) != target_rate:
            strain = strain.resample(target_rate)

        raw_np = strain.value  # numpy float64, values ~1e-21

        # ── CRITICAL: rescale before float32 cast ────────────────────────
        # LIGO strain values are ~1e-21.  float32 has only ~7 significant
        # digits of relative precision.  When many 1e-21 values are loaded
        # and operations (mean, std) are computed in float32, the tiny
        # relative differences vanish -- std collapses to 0.
        #
        # Fix: divide by the RMS so values land near 1.0 before casting.
        # The physical scale is irrelevant at this stage -- whitening will
        # re-normalise everything anyway.
        rms = float(np.sqrt(np.mean(raw_np ** 2)))
        if rms > 0:
            raw_np = raw_np / rms

        channels.append(torch.tensor(raw_np, dtype=torch.float32))
        print(f"done  ({len(strain.value)} samples,  rms-scaled by 1/{rms:.3e})")

    # Step 4 -- stack: shape (1, 2, N)
    tensor = torch.stack(channels, dim=0).unsqueeze(0)
    print(f"[DATA]  Output shape: {tuple(tensor.shape)}  "
          f"(batch=1, detectors={len(detectors)}, samples={tensor.shape[-1]})\n")

    return tensor, float(gps_centre)


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 1 : SpectralDensity  (unchanged from your original -- math is correct)
# ══════════════════════════════════════════════════════════════════════════════

class SpectralDensity:
    """
    Estimate the Power Spectral Density (PSD) using Welch's method.

    WHAT IS PSD IN PLAIN ENGLISH?
    ------------------------------
    PSD tells you HOW LOUD the noise is at each frequency.
    Imagine you're in a noisy room.  The PSD is like a frequency-by-
    frequency loudness map: "bass frequencies are very loud, mid
    frequencies are quieter, high frequencies are medium."

    LIGO's noise is NOT the same at every frequency.  It is extremely
    loud below 20 Hz (seismic vibrations from the ground) and much
    quieter between 100-500 Hz (where BBH signals live).  The PSD
    captures this shape precisely.

    WELCH'S METHOD in plain English:
    ---------------------------------
    Instead of taking one big Fourier transform (which is noisy), we:
      1. Cut the signal into many short overlapping windows
      2. Compute the FFT of each window
      3. Average them -- averaging cancels random noise, keeping the
         true noise floor shape

    This gives a stable, smooth estimate of the noise floor.

    Output shape: (batch, channels, num_frequency_bins)
    """

    def __init__(self, sample_rate: float, fftlength: float, overlap: float | None = None):
        self.sample_rate = sample_rate
        self.fftlength   = fftlength
        if overlap is None:
            overlap = fftlength / 2
        self.overlap  = overlap
        self.nperseg  = int(fftlength * sample_rate)
        self.nstride  = self.nperseg - int(overlap * sample_rate)
        self.window   = np.hanning(self.nperseg)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, _ = x.shape
        num_freqs = self.nperseg // 2 + 1
        psd = torch.zeros(batch_size, num_channels, num_freqs, dtype=x.dtype)

        for b in range(batch_size):
            for c in range(num_channels):
                signal = x[b, c].numpy()
                ffts   = []
                for start in range(0, len(signal) - self.nperseg + 1, self.nstride):
                    seg      = signal[start : start + self.nperseg]
                    windowed = seg * self.window
                    ffts.append(np.abs(np.fft.rfft(windowed)) ** 2)

                psd_vals = np.mean(ffts, axis=0) if ffts else np.ones(num_freqs)
                scale    = 1.0 / (self.sample_rate * np.sum(self.window ** 2))
                psd[b, c] = torch.from_numpy(psd_vals * scale).float()

        return psd


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 2 : Whiten  (unchanged -- math is correct)
# ══════════════════════════════════════════════════════════════════════════════

class Whiten:
    """
    Whiten the strain signal using the estimated PSD.

    WHAT IS WHITENING IN PLAIN ENGLISH?
    -------------------------------------
    Remember the PSD tells us how loud noise is at each frequency.
    Whitening divides the signal at each frequency by how loud the
    noise is there.

    Before whitening: low frequencies are 1000× louder than mid frequencies.
    After whitening:  all frequencies have roughly EQUAL loudness.

    Now the BBH chirp (which lives at 30-500 Hz) is no longer buried
    under the massive low-frequency noise -- it stands out clearly.

    ANALOGY: imagine turning down the bass on a stereo until the bass,
    midrange, and treble are all the same volume.  Suddenly you can
    hear the melody clearly that was always there but hidden by bass.

    HIGHPASS / LOWPASS:
    -------------------
    After whitening we also zero out frequencies we don't care about:
    - Below highpass_hz (20 Hz)  -- seismic noise, still unreliable
    - Above lowpass_hz  (512 Hz) -- BBH signals have no power here

    CROPPING:
    ---------
    Whitening uses a filter that needs time to "warm up" at the edges.
    We crop self.pad samples from each end to remove this edge artefact.
    """

    def __init__(
        self,
        fduration:   float,
        sample_rate: float,
        highpass:    float | None = None,
        lowpass:     float | None = None,
    ):
        self.fduration   = fduration
        self.sample_rate = sample_rate
        self.highpass    = highpass
        self.lowpass     = lowpass
        self.pad         = int(fduration * sample_rate / 2)

    def __call__(self, x: torch.Tensor, psd: torch.Tensor, crop: bool = True) -> torch.Tensor:
        batch_size, num_channels, num_samples = x.shape
        num_freqs = num_samples // 2 + 1

        # Interpolate PSD to match signal's frequency resolution
        if psd.size(-1) != num_freqs:
            psd_interp = torch.nn.functional.interpolate(
                psd, size=(num_freqs,), mode="linear"
            )
        else:
            psd_interp = psd

        whitened = torch.zeros_like(x)

        for b in range(batch_size):
            for c in range(num_channels):
                signal      = x[b, c] - x[b, c].mean()   # remove DC offset
                signal_fft  = torch.fft.rfft(signal.double(), norm="forward")
                psd_channel = psd_interp[b, c].clone().double()

                # Zero out unwanted frequency bands
                df = self.sample_rate / num_samples
                if self.highpass is not None:
                    psd_channel[: int(self.highpass / df)] = 1e-10
                if self.lowpass is not None:
                    psd_channel[int(self.lowpass / df) :] = 1e-10

                # Divide FFT by ASD (sqrt of PSD) to whiten
                asd            = torch.sqrt(psd_channel + 1e-10)
                signal_fft     = signal_fft / asd
                signal_whitened = torch.fft.irfft(
                    signal_fft, n=num_samples, norm="forward"
                ).float() / (self.sample_rate ** 0.5)

                whitened[b, c] = signal_whitened

        if crop:
            whitened = whitened[:, :, self.pad : -self.pad]

        return whitened


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 3 : ChannelWiseScaler  (unchanged -- math is correct)
# ══════════════════════════════════════════════════════════════════════════════

class ChannelWiseScaler:
    """
    Z-score normalise each detector channel independently.

    WHAT IS THIS IN PLAIN ENGLISH?
    --------------------------------
    After whitening, H1 and L1 may still have slightly different
    amplitude scales (different sensitivities that day, calibration
    differences, etc.).

    Z-score normalization transforms each channel so that:
        mean  = 0    (signal centred at zero)
        std   = 1    (signal spans roughly -3 to +3)

    Formula:  z = (x - mean) / std

    WHY CHANNEL-WISE (not global)?
    --------------------------------
    We compute mean and std SEPARATELY for H1 and L1.
    If we used a global mean/std, a louder H1 day would skew L1's
    normalization.  Channel-wise keeps each detector on its own scale.

    WHY NOT MIN-MAX INSTEAD?
    ------------------------
    Min-max squashes everything to [0, 1], destroying the negative
    half of the chirp oscillation.  Z-score keeps positive AND
    negative values, preserving the waveform's oscillation shape --
    critical because CNNs learn patterns from the wave's shape.
    """

    def __init__(self, num_channels: int | None = None):
        self.num_channels = num_channels
        self.mean = torch.zeros(1)
        self.std  = torch.ones(1)
        if num_channels is not None:
            self.mean = torch.zeros(num_channels, 1)
            self.std  = torch.ones(num_channels, 1)

    def fit(self, x: torch.Tensor, std_reg: float | None = 0.0) -> None:
        """Compute mean and std from data (call before __call__)."""
        if x.ndim == 1:
            self.mean = x.mean(dim=0, keepdim=True)
            self.std  = x.std(dim=0, correction=0, keepdim=True)
        elif x.ndim == 2:
            self.mean = x.mean(dim=-1, keepdim=True)
            self.std  = x.std(dim=-1, correction=0, keepdim=True)
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got shape {x.shape}")
        if std_reg is not None:
            self.std = self.std + std_reg

    def __call__(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            return (x - self.mean) / self.std
        else:
            return self.std * x + self.mean


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 2 : estimate_psd  (unchanged -- now receives REAL data)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_psd(
    x:           torch.Tensor,
    sample_rate: float,
    fftlength:   float,
    overlap:     float,
) -> torch.Tensor:
    """
    Wrapper: estimate Power Spectral Density from a batch of strain data.

    With real O3 data this now estimates the TRUE noise floor of H1 and L1
    on the day of the event -- not a synthetic Gaussian approximation.

    Input  : x shape (batch, channels, samples)
    Output : psd shape (batch, channels, freq_bins)
    """
    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        overlap=overlap,
    )
    return spectral_density(x)


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 3 : apply_preprocessing  (unchanged -- orchestrates the pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def apply_preprocessing(
    x:           torch.Tensor,
    sample_rate: float,
    fduration:   float,
    highpass:    float | None,
    lowpass:     float | None,
    fftlength:   float,
    overlap:     float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full preprocessing pipeline: PSD → Whiten → Normalise.

    PIPELINE SUMMARY IN PLAIN ENGLISH:
    ------------------------------------
    Step A  PSD estimation  : "What does the noise floor look like?"
    Step B  Whitening       : "Divide by that noise floor to flatten it."
    Step C  Normalisation   : "Scale each detector to zero-mean, unit-std."

    This is EXACTLY the preprocessing used in the 2024 paper you are
    reading (arXiv:2403.18661), applied now to actual O3 strain data.

    Input  : x shape (batch, channels, samples)  -- raw strain
    Output : (psd, whitened, normalized)  -- all useful for debugging
    """
    # A -- estimate noise floor from the data itself
    psd = estimate_psd(x, sample_rate=sample_rate, fftlength=fftlength, overlap=overlap)

    # B -- whiten using that noise floor
    whitener = Whiten(
        fduration=fduration,
        sample_rate=sample_rate,
        highpass=highpass,
        lowpass=lowpass,
    )
    whitened = whitener(x, psd)

    # C -- channel-wise Z-score normalisation
    num_channels = whitened.size(1)
    scaler       = ChannelWiseScaler(num_channels=num_channels)
    fit_data     = whitened.transpose(0, 1).reshape(num_channels, -1)
    scaler.fit(fit_data, std_reg=1e-8)
    normalized = scaler(whitened)

    return psd, whitened, normalized


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 4 : summarize  (small helper, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def summarize(name: str, x: torch.Tensor) -> None:
    """
    Print shape and statistics -- uses scientific notation for tiny values.
    Raw LIGO strain is ~1e-21, which rounds to 0.0000 in fixed notation.
    """
    mean_val = x.mean().item()
    std_val  = x.std(unbiased=False).item()
    if std_val == 0 or abs(std_val) < 0.001 or abs(std_val) > 1e4:
        fmt = ".4e"
    else:
        fmt = "+.4f"
    print(
        f"  {name:<12}  shape={str(tuple(x.shape)):<25}  "
        f"mean={mean_val:{fmt}}  std={std_val:{fmt}}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 5 : plot_pipeline  (NEW -- visualise every stage)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pipeline(
    raw:        torch.Tensor,
    whitened:   torch.Tensor,
    normalized: torch.Tensor,
    psd:        torch.Tensor,
    event_name: str,
    sample_rate: float,
    highpass:   float,
    lowpass:    float,
    save_path:  str = "pipeline_output.png",
) -> None:
    """
    Plot all preprocessing stages side by side.

    WHAT THIS SHOWS:
    ----------------
    Row 1 : Raw strain (H1 and L1) -- mostly noise, no visible signal
    Row 2 : After whitening        -- noise floor flattened, signal visible
    Row 3 : After normalisation    -- centred at zero, ready for CNN
    Row 4 : PSD of both detectors  -- noise floor shape (log-log scale)
            with the signal band [highpass, lowpass] highlighted
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    fig.suptitle(f"Preprocessing Pipeline  ·  {event_name}", fontsize=14, fontweight="bold")

    det_labels  = ["H1 (Hanford)", "L1 (Livingston)"]
    det_colors  = ["#E8593C", "#534AB7"]

    # Row 1: Raw strain
    n_raw = raw.shape[-1]
    t_raw = np.arange(n_raw) / sample_rate
    for ch in range(raw.shape[1]):
        axes[0].plot(t_raw, raw[0, ch].numpy(),
                     color=det_colors[ch], alpha=0.7,
                     linewidth=0.4, label=det_labels[ch])
    axes[0].set_title("Raw O3 Strain  —  noise completely dominates, no signal visible")
    axes[0].set_ylabel("Strain")
    axes[0].legend(fontsize=8)
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Row 2: Whitened
    n_wh = whitened.shape[-1]
    t_wh = np.arange(n_wh) / sample_rate
    for ch in range(whitened.shape[1]):
        axes[1].plot(t_wh, whitened[0, ch].numpy(),
                     color=det_colors[ch], alpha=0.7,
                     linewidth=0.5, label=det_labels[ch])
    axes[1].set_title(
        f"After Whitening  +  Bandpass [{highpass}–{lowpass} Hz]"
        "  —  noise floor flattened, chirp starting to be visible"
    )
    axes[1].set_ylabel("Whitened strain")
    axes[1].legend(fontsize=8)

    # Row 3: Normalised
    for ch in range(normalized.shape[1]):
        axes[2].plot(t_wh, normalized[0, ch].numpy(),
                     color=det_colors[ch], alpha=0.8,
                     linewidth=0.5, label=det_labels[ch])
    axes[2].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[2].set_title(
        "After Z-score Normalisation  —  mean=0, std=1, ready for CNN input"
    )
    axes[2].set_ylabel("Normalised strain (σ)")
    axes[2].legend(fontsize=8)

    # Row 4: PSD
    freqs     = np.fft.rfftfreq(int(2.0 * sample_rate), d=1.0 / sample_rate)
    num_freqs = psd.shape[-1]
    if len(freqs) != num_freqs:
        freqs = np.linspace(0, sample_rate / 2, num_freqs)

    for ch in range(psd.shape[1]):
        axes[3].loglog(freqs[1:], psd[0, ch].numpy()[1:],
                       color=det_colors[ch], linewidth=1,
                       alpha=0.8, label=det_labels[ch])

    axes[3].axvspan(highpass, lowpass, alpha=0.15, color="#0F6E56",
                    label=f"Signal band ({highpass}–{lowpass} Hz)")
    axes[3].set_xlim(10, sample_rate / 2)
    axes[3].set_title(
        "Power Spectral Density  —  shows the noise floor shape before whitening"
    )
    axes[3].set_xlabel("Frequency [Hz]")
    axes[3].set_ylabel("PSD [1/Hz]")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT]  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="O3 BBH preprocessing pipeline (real GWOSC data)"
    )
    parser.add_argument(
        "--event", type=str, default=DEFAULT_EVENT,
        choices=list(O3_BBH_EVENTS.keys()),
        help="O3 BBH event name (default: GW190521)"
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION,
        help="Seconds of data around event centre (default: 16)"
    )
    parser.add_argument("--sample-rate", type=float, default=float(SAMPLE_RATE))
    parser.add_argument("--fftlength",   type=float, default=FFTLENGTH)
    parser.add_argument("--overlap",     type=float, default=OVERLAP)
    parser.add_argument("--fduration",   type=float, default=FDURATION)
    parser.add_argument("--highpass",    type=float, default=HIGHPASS_HZ)
    parser.add_argument("--lowpass",     type=float, default=LOWPASS_HZ)
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a pipeline visualisation PNG"
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  O3 BBH Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Event       : {args.event}")
    print(f"  Duration    : {args.duration} s")
    print(f"  Sample rate : {args.sample_rate} Hz")
    print(f"  Bandpass    : {args.highpass} – {args.lowpass} Hz")
    print("=" * 60)

    # ── Step 1 : Download real O3 data ────────────────────────────────────────
    raw, gps_centre = fetch_o3_strain(
        event_name  = args.event,
        duration    = args.duration,
        target_rate = int(args.sample_rate),
    )

    # ── Step 2 : Run preprocessing ────────────────────────────────────────────
    print("[PREPROC]  Running PSD → Whiten → Normalise ...")
    psd, whitened, normalized = apply_preprocessing(
        raw,
        sample_rate = args.sample_rate,
        fduration   = args.fduration,
        highpass    = args.highpass,
        lowpass     = args.lowpass,
        fftlength   = args.fftlength,
        overlap     = args.overlap,
    )

    # ── Step 3 : Sanity-check statistics ──────────────────────────────────────
    print("\n[STATS]  Tensor statistics at each stage:")
    print(f"  {'name':<12}  {'shape':<25}  {'mean':>8}  {'std':>8}")
    print("  " + "-" * 60)
    summarize("raw",        raw)
    summarize("psd",        psd)
    summarize("whitened",   whitened)
    summarize("normalized", normalized)

    # ── Quick validation ───────────────────────────────────────────────────────
    norm_mean = normalized.mean().item()
    norm_std  = normalized.std(unbiased=False).item()
    mean_ok   = abs(norm_mean) < 0.05
    std_ok    = abs(norm_std - 1.0) < 0.05
    print(f"\n[CHECK]  Normalisation mean ≈ 0 : {'✓ PASS' if mean_ok else '✗ FAIL'}  ({norm_mean:+.4f})")
    print(f"[CHECK]  Normalisation std  ≈ 1 : {'✓ PASS' if std_ok  else '✗ FAIL'}  ({norm_std:.4f})")

    print(f"\n[READY]  normalized tensor shape : {tuple(normalized.shape)}")
    print("[READY]  This tensor is your CNN input.")
    print(f"[READY]  Feed it as:  model(normalized)  -- shape (1, 2, {normalized.shape[-1]})")

    # ── Optional plot ─────────────────────────────────────────────────────────
    if args.plot:
        plot_pipeline(
            raw         = raw,
            whitened    = whitened,
            normalized  = normalized,
            psd         = psd,
            event_name  = args.event,
            sample_rate = args.sample_rate,
            highpass    = args.highpass,
            lowpass     = args.lowpass,
            save_path   = f"{args.event}_pipeline.png",
        )

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
