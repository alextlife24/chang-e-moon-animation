"""Procedural soundtrack for Chang'e Flies to the Moon (20 s, 48 kHz stereo).

Everything is synthesised: D yu-mode pentatonic (D F G A C), guzheng-like plucks,
dizi-like flute, bells, drums, wind / water / insects ambience, reverb.
usage: python audio.py out.wav
"""
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
DUR = 20.0
N = int(SR * DUR)
T = np.arange(N) / SR
rng = np.random.default_rng(2024)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


NOTE = dict(D3=50, F3=53, G3=55, A3=57, C4=60, D4=62, F4=65, G4=67, A4=69, C5=72, D5=74, F5=77, G5=79, A5=81,
            C6=84, D6=86, F6=89, G6=91, A6=93)


def env_ar(n, a, r, sr=SR):
    e = np.ones(n)
    na, nr = int(a * sr), int(r * sr)
    if na:
        e[:na] = np.linspace(0, 1, na) ** 1.5
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr) ** 2
    return e


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ramp(a, b):
    return smooth((T - a) / (b - a))


class Bus:
    def __init__(self):
        self.x = np.zeros((N, 2))

    def add(self, t0, mono, gain=1.0, pan=0.0):
        i0 = int(t0 * SR)
        if i0 >= N:
            return
        if i0 < 0:
            mono = mono[-i0:]
            i0 = 0
        n = min(len(mono), N - i0)
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        self.x[i0:i0 + n, 0] += mono[:n] * gain * l * 1.414
        self.x[i0:i0 + n, 1] += mono[:n] * gain * r * 1.414

    def add_st(self, st, gain=1.0):
        self.x += st * gain


# ---------------------------------------------------------------- instruments
def zheng(m, dur=2.5, amp=1.0, bend=0.0, vib=0.0):
    """additive plucked string with guzheng-style press bend & vibrato."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = mtof(m)
    fcurve = f0 * 2 ** ((bend * (1 - np.exp(-np.maximum(t - 0.12, 0) * 9)) * (t > 0.12)) / 12)
    fcurve *= 1 + vib * 0.007 * np.sin(2 * np.pi * 5.6 * t) * smooth((t - 0.25) / 0.3)
    ph = 2 * np.pi * np.cumsum(fcurve) / SR
    y = np.zeros(n)
    for k in range(1, 11):
        inh = 1 + 0.0004 * k * k
        y += np.sin(ph * k * inh) * (1 / k ** 1.15) * np.exp(-t * (1.1 + 0.75 * k)) * (1 if k * f0 < 9000 else 0)
    click = rng.standard_normal(n) * np.exp(-t * 180) * 0.25
    b, a = signal.butter(2, [1500, 6000], 'bandpass', fs=SR)
    y += signal.lfilter(b, a, click)
    y *= env_ar(n, 0.002, 0.05)
    return y * amp * 0.35


def flute_line(notes, amp=1.0, glide=0.07):
    """legato dizi-like line; notes = [(start, dur, midi), ...]."""
    t0 = notes[0][0]
    t1 = notes[-1][0] + notes[-1][1]
    n = int((t1 - t0 + 0.4) * SR)
    t = np.arange(n) / SR + t0
    f = np.full(n, mtof(notes[0][2]))
    env = np.zeros(n)
    for i, (s, d, m) in enumerate(notes):
        fprev = mtof(notes[i - 1][2]) if i else mtof(m)
        g = smooth((t - s) / glide)
        f = np.where(t >= s, fprev + (mtof(m) - fprev) * g, f)
        a = smooth((t - s) / 0.06) * (1 - smooth((t - (s + d)) / 0.18))
        env = np.maximum(env, a * (0.85 + 0.15 * smooth((t - s) / 0.4)))
    # vibrato grows within each held note
    vib = np.zeros(n)
    for s, d, m in notes:
        vib += np.where((t >= s) & (t < s + d), smooth((t - s - 0.25) / 0.35), 0)
    f = f * (1 + 0.006 * vib * np.sin(2 * np.pi * 5.3 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) + 0.32 * np.sin(2 * ph + 0.3) + 0.12 * np.sin(3 * ph) + 0.05 * np.sin(4 * ph + 1)
    y = np.tanh(y * 1.3) / 1.3
    breath = rng.standard_normal(n)
    b, a = signal.butter(2, [1800, 5200], 'bandpass', fs=SR)
    breath = signal.lfilter(b, a, breath) * 0.16
    out = (y + breath) * env
    b, a = signal.butter(2, 7000, 'lowpass', fs=SR)
    return t0, signal.lfilter(b, a, out) * amp * 0.22


def bell(m, dur=5.0, amp=1.0, bright=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = mtof(m)
    y = np.zeros(n)
    for ratio, a, dec in ((0.5, 0.35, 0.6), (1.0, 1.0, 0.9), (2.0, 0.55, 1.4), (2.76, 0.45 * bright, 2.2),
                          (5.4, 0.25 * bright, 3.6), (8.93, 0.12 * bright, 5.5)):
        if f0 * ratio > 16000:
            continue
        beat = 1 + 0.0015 * np.sin(2 * np.pi * 0.7 * t)
        y += a * np.sin(2 * np.pi * f0 * ratio * t * beat) * np.exp(-t * dec)
    y *= env_ar(n, 0.003, 0.3)
    return y * amp * 0.18


def pad(chord, t0, t1, amp, fade=1.5):
    n = int((t1 - t0) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for m in chord:
        f = mtof(m)
        for det in (-0.25, 0.0, 0.31):
            ff = f + det
            y += (np.sin(2 * np.pi * ff * t + rng.uniform(0, 6)) + 0.25 * np.sin(4 * np.pi * ff * t)
                  + 0.08 * np.sin(6 * np.pi * ff * t)) / 3
    e = smooth(t / fade) * (1 - smooth((t - (t1 - t0 - fade)) / fade))
    return y * e * amp * 0.05


def drum(amp=1.0, f0=95):
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    f = f0 * (0.62 + 0.38 * np.exp(-t * 18))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6.5)
    nz = rng.standard_normal(n) * np.exp(-t * 40)
    b, a = signal.butter(2, 900, 'lowpass', fs=SR)
    y += signal.lfilter(b, a, nz) * 0.5
    return y * env_ar(n, 0.001, 0.1) * amp * 0.5


def svf_noise(dur, fc_curve, q=0.7, seed=0):
    """time-varying band-pass filtered noise (whoosh / wind)."""
    n = int(dur * SR)
    x = np.random.default_rng(seed).standard_normal(n)
    fc = np.interp(np.arange(n), np.linspace(0, n, len(fc_curve)), fc_curve)
    g = np.tan(np.pi * np.clip(fc, 20, SR * 0.45) / SR)
    k = 1 / q
    out = np.zeros(n)
    ic1 = ic2 = 0.0
    a1 = 1 / (1 + g * (g + k))
    for i in range(n):
        v3 = x[i] - ic2
        v1 = a1[i] * ic1 + g[i] * a1[i] * v3
        v2 = ic2 + g[i] * v1
        ic1 = 2 * v1 - ic1
        ic2 = 2 * v2 - ic2
        out[i] = v1
    return out


# ---------------------------------------------------------------- build
def build():
    music, sfx, amb = Bus(), Bus(), Bus()
    D = NOTE

    # ---- ambience: night wind, lake water, insects
    for ch in range(2):
        nz = np.random.default_rng(10 + ch).standard_normal(N)
        b, a = signal.butter(2, [120, 900], 'bandpass', fs=SR)
        w = signal.lfilter(b, a, nz)
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * T + ch) * np.sin(2 * np.pi * 0.07 * T + 2)
        lvl = 0.05 + 0.10 * ramp(11.5, 13.5) * (1 - ramp(16, 17.5)) + 0.02 * ramp(16, 17)
        amb.x[:, ch] += w * lfo * lvl
        b, a = signal.butter(2, 380, 'lowpass', fs=SR)
        water = signal.lfilter(b, a, np.random.default_rng(20 + ch).standard_normal(N))
        laps = np.zeros(N)
        for tc in np.arange(0.2, 10, 0.75) + rng.uniform(0, 0.3, 14):
            laps += np.exp(-((T - tc) / 0.18) ** 2)
        amb.x[:, ch] += water * laps * 0.06 * (1 - ramp(8.5, 11))
    for tc in np.arange(0.4, 10.0, 1.05):
        tc += rng.uniform(-0.15, 0.15)
        pan = rng.uniform(-0.8, 0.8)
        f = rng.uniform(4100, 4700)
        for k in range(3):
            n = int(0.035 * SR)
            t = np.arange(n) / SR
            ch = np.sin(2 * np.pi * f * t) * np.sin(np.pi * t / 0.035) ** 2 * (0.6 + 0.4 * np.sin(2 * np.pi * 220 * t))
            g = 0.035 * (1 - smooth((tc - 7.5) / 2.5))
            amb.add(tc + k * 0.06, ch, g, pan)

    # ---- pad / drone
    music.add(0.0, pad([D['D3'] - 12, D['A3'] - 12, D['D3']], 0.0, 9.6, 0.7, fade=2.5))
    music.add(8.6, pad([D['D3'] - 12, D['A3'] - 12, D['D3'], D['F3'], D['A3']], 8.6, 12.6, 1.0, fade=1.2))
    music.add(11.8, pad([D['D3'] - 12, D['G3'] - 12, D['D3'], D['G3'], D['C4'], D['F4']], 11.8, 16.6, 1.25, fade=1.0))
    music.add(16.0, pad([D['D3'] - 12, D['A3'] - 12, D['D3'], D['A3'], D['D4'], D['F4'], D['A4']], 16.0, 20.0, 1.2, fade=1.2))

    # ---- A: world (zheng)
    for s, m, b_, v in ((0.3, 'D4', 0, 0), (1.2, 'A4', 0, 0), (2.1, 'D5', 0, 1), (2.95, 'C5', 0, 0), (3.45, 'G4', 2, 1)):
        music.add(s, zheng(D[m], 3.0, 0.9, bend=b_, vib=v), 1.0, -0.2)
    music.add(0.3, zheng(D['D3'], 3.5, 0.8), 1.0, -0.35)
    # ---- B: Chang'e appears -> bell + flute
    music.add(3.95, bell(D['A5'], 5, 0.8), 1.0, 0.3)
    t0, fl = flute_line([(4.2, 0.75, D['A4']), (4.95, 0.5, D['C5']), (5.45, 1.25, D['D5']), (6.7, 0.45, D['F5']),
                         (7.15, 0.45, D['D5']), (7.6, 1.0, D['C5'])], amp=1.0)
    music.add(t0, fl, 1.0, 0.15)
    music.add(4.0, zheng(D['D3'], 3.0, 0.7), 1.0, -0.4)
    music.add(4.02, zheng(D['A3'], 3.0, 0.55), 1.0, -0.3)
    music.add(6.0, zheng(D['F3'], 3.0, 0.6), 1.0, -0.4)
    music.add(6.02, zheng(D['C4'], 3.0, 0.5), 1.0, -0.3)
    # ---- C: tremolo while she raises the elixir
    for i, s in enumerate(np.arange(7.85, 8.95, 1 / 14)):
        g = 0.25 + 0.75 * ((s - 7.85) / 1.1) ** 1.5
        music.add(s, zheng(D['A4'], 0.5, 0.45 * g), 1.0, 0.1 if i % 2 else -0.1)
    # reverse swell into the swallow
    sw = svf_noise(0.9, np.geomspace(300, 5000, 50), q=1.2, seed=3)
    sfx.add(8.1, sw * np.linspace(0, 1, len(sw)) ** 2 * 0.25, 1.0, 0.0)
    # 9.0 swallow: boom + bell + shimmer
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(68 * (0.55 + 0.45 * np.exp(-t * 3))) / SR) * np.exp(-t * 1.6)
    sfx.add(9.0, boom * env_ar(n, 0.004, 0.3), 0.55, 0.0)
    music.add(9.0, bell(D['D5'], 6, 1.0, 1.2), 1.0, -0.1)
    music.add(9.0, bell(D['A5'], 6, 0.6, 1.2), 1.0, 0.25)
    for i in range(14):
        s = 9.05 + i * 0.045
        music.add(s, bell([D['D6'], D['F6'], D['G6'], D['A6']][i % 4] + (12 if i > 9 else 0), 1.5, 0.18), 1.0,
                  -0.7 + i * 0.1)
    # lift-off flute rising
    t0, fl = flute_line([(9.6, 0.6, D['D5']), (10.2, 0.5, D['F5']), (10.7, 0.5, D['G5']), (11.2, 0.9, D['A5'])], amp=1.05)
    music.add(t0, fl, 1.0, 0.1)
    for s, m in ((9.5, 'D3'), (10.5, 'G3'), (11.3, 'A3')):
        music.add(s, zheng(D[m], 2.5, 0.6), 1.0, -0.4)
    # guzheng glissando (刮奏) up into the ascent
    scale = [D[k] for k in ('D4', 'F4', 'G4', 'A4', 'C5', 'D5', 'F5', 'G5', 'A5', 'C6', 'D6')]
    for i, m in enumerate(scale):
        music.add(11.7 + i * 0.04, zheng(m, 2.0, 0.55), 1.0, -0.6 + i * 0.12)
    # ---- D: climax
    for s, g in ((12.0, 0.7), (12.9, 0.7), (13.6, 0.8), (14.2, 0.85), (14.7, 0.9), (15.1, 0.95), (15.45, 1.0),
                 (15.75, 1.0), (16.05, 1.1), (16.4, 1.25)):
        sfx.add(s, drum(g), 0.9, 0.0)
    t0, fl = flute_line([(12.3, 1.2, D['A5']), (13.5, 0.6, D['C6']), (14.1, 1.35, D['D6']), (15.45, 0.3, D['C6']),
                         (15.75, 0.55, D['A5'])], amp=1.0)
    music.add(t0, fl, 1.0, 0.1)
    arp = [D[k] for k in ('D4', 'A4', 'D5', 'F5', 'A5', 'F5', 'D5', 'A4')]
    for i, s in enumerate(np.arange(12.5, 16.2, 0.125)):
        g = 0.35 + 0.35 * (s - 12.5) / 3.7
        music.add(s, zheng(arp[i % 8] + (2 if (i // 8) % 2 else 0) * 0, 1.2, g), 1.0, -0.5 + 0.12 * (i % 8))
    # whooshes: lift, cloud passes
    for s, d, f0, f1, g, pan in ((11.9, 1.9, 250, 2600, 0.55, 0.0), (12.9, 0.9, 500, 3500, 0.4, -0.5),
                                 (14.25, 0.9, 600, 4000, 0.4, 0.5), (15.4, 1.1, 2600, 400, 0.3, 0.0)):
        wn = svf_noise(d, np.concatenate([np.geomspace(f0, f1, 40)]), q=1.4, seed=int(s * 10))
        e = np.sin(np.pi * np.linspace(0, 1, len(wn))) ** 1.5
        sfx.add(s, wn * e, g, pan)
    for i in range(26):
        s = 12.2 + rng.uniform(0, 4.6)
        music.add(s, bell(D[['D6', 'F6', 'G6', 'A6'][int(rng.integers(0, 4))]] + 12 * int(rng.integers(0, 2)), 1.2,
                          0.1), 1.0, rng.uniform(-0.9, 0.9))
    # ---- E: arrival at the moon palace
    for i, m in enumerate(reversed(scale)):
        music.add(16.15 + i * 0.035, zheng(m, 2.0, 0.4), 1.0, 0.6 - i * 0.12)
    music.add(16.4, bell(D['D4'], 7, 1.0, 1.0), 1.0, 0.0)
    music.add(16.4, bell(D['A4'], 7, 0.55, 1.0), 1.0, -0.3)
    music.add(16.42, bell(D['D5'], 7, 0.45, 1.3), 1.0, 0.3)
    t0, fl = flute_line([(16.6, 0.8, D['A5']), (17.4, 0.4, D['G5']), (17.8, 2.0, D['D5'])], amp=0.9)
    music.add(t0, fl, 1.0, 0.1)
    for s, m in ((17.0, 'D4'), (17.6, 'A4'), (18.2, 'D5'), (18.9, 'A5')):
        music.add(s, zheng(D[m], 2.5, 0.5, vib=1), 1.0, -0.2)
    music.add(16.4, zheng(D['D3'], 3.5, 0.75), 1.0, -0.3)

    # ---------------------------------------------------------------- reverb
    def reverb(x, rt=2.8, seed=7):
        n = int(rt * SR)
        t = np.arange(n) / SR
        out = np.zeros_like(x)
        for ch in range(2):
            nz = np.random.default_rng(seed + ch).standard_normal(n)
            lo = signal.lfilter(*signal.butter(1, 2500, 'lowpass', fs=SR), nz) * np.exp(-t * 6.9 / rt)
            hi = (nz - lo) * np.exp(-t * 6.9 / (rt * 0.45))
            ir = (lo * 1.4 + hi * 0.5)
            ir[: int(0.012 * SR)] = 0
            ir /= np.sqrt(np.sum(ir ** 2))
            out[:, ch] = signal.fftconvolve(x[:, ch], ir)[:len(x)]
        return out

    mus = music.x + reverb(music.x) * 0.55
    fx = sfx.x + reverb(sfx.x, 2.0, 9) * 0.35
    mix = mus * 1.0 + fx * 0.9 + amb.x * 1.0
    b, a = signal.butter(2, 28, 'highpass', fs=SR)
    mix = signal.lfilter(b, a, mix, axis=0)
    fade = (smooth(T / 0.25) * (1 - smooth((T - 18.0) / 2.0)))[:, None]
    mix *= fade
    mix[-int(0.03 * SR):] = 0
    # gentle bus compression (RMS-follow) + safe peak normalisation
    peak = np.max(np.abs(mix))
    mix *= 0.80 / peak
    mix = np.tanh(mix * 1.05) / np.tanh(1.05)
    mix *= 0.84 / np.max(np.abs(mix))
    return mix


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'soundtrack.wav'
    mix = build()
    wavfile.write(out, SR, (mix * 32767).astype(np.int16))
    print('peak', np.max(np.abs(mix)), 'rms', np.sqrt(np.mean(mix ** 2)))
