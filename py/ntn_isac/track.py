"""Kalman tracking and trajectory forecasting (paper section II-C and III-B).

The paper carries a six-state Cartesian filter, x = [x, y, z, vx, vy, vz]
(eq 18-21), fed by fused measurements from several UEs (eq 24) and forecast
forward with eq 27. That state is observable there because each UE contributes
a bistatic range *and* an AoA from an 8-element array, and several UEs at known
positions are fused.

This testbed has one receiver and one antenna. The measurement per coherent
processing interval is a pair: excess delay and residual Doppler. A target
position in three dimensions lies on the intersection of an iso-delay
ellipsoid and an iso-Doppler cone, which is a curve, not a point - so 3-D
position is *not observable* from this geometry no matter how the filter is
written. A six-state filter fed two numbers per update would be reporting its
own process model back as a position estimate, with a covariance that looks
tight because nothing ever contradicts it. That is the failure mode this
module refuses.

So the state is the observable pair itself:

    x = [excess_range_m, excess_range_rate_mps]

with the constant-velocity transition of eq 20 reduced to two dimensions and
H = I2. The measurement function is then exactly linear and no Jacobian is
needed - which is the same simplification the paper reaches by its own route
(eq 25: "the measurement Jacobian H simplifies to an identity matrix,
eliminating the need for complex analytical derivatives"). The filter here is
a linear KF for a structural reason, not as an approximation.

What this costs, stated plainly for the talk: this tracker gives excess range
and range rate with a covariance, and forecasts both. It does not give a
Cartesian track. Recovering one needs a second receiver, which handoff
section 12 puts out of scope. The message interface carries enough for a
second node to be added later without changing anything here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C = 299792458.0


def transition(dt: float) -> np.ndarray:
    """Paper eq 20, reduced to [range, range rate]: constant velocity."""
    return np.array([[1.0, float(dt)], [0.0, 1.0]])


def process_noise(dt: float, sigma_accel: float) -> np.ndarray:
    """Discrete white-noise-acceleration Q for a constant-velocity model.

    sigma_accel is the manoeuvre the filter should tolerate, in m/s^2 of
    excess-range acceleration. Sizing it from the scenario rather than tuning
    it blind: the drone crosses at 15 m/s, and the excess-range rate it
    produces swings tens of m/s over the pass, so a few m/s^2 is the honest
    order. Too small and the filter lags the crossing; too large and it stops
    smoothing at all.
    """
    dt = float(dt)
    q = float(sigma_accel) ** 2
    return q * np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0],
                         [dt ** 3 / 2.0, dt ** 2]])


@dataclass
class Track:
    """One target hypothesis in (excess range, excess range rate)."""

    tid: int
    x: np.ndarray                    # [2]
    P: np.ndarray                    # [2, 2]
    t: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    history: list = field(default_factory=list)

    @property
    def excess_m(self) -> float:
        return float(self.x[0])

    @property
    def range_rate_mps(self) -> float:
        return float(self.x[1])

    def doppler_hz(self, fc: float) -> float:
        """Residual Doppler this state implies. Inverse of the one-way
        bistatic conversion: f_d = -Rdot / lambda."""
        return -self.range_rate_mps * fc / C

    def sigma(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.P), 0.0))

    def predict(self, dt: float, sigma_accel: float):
        """Paper eq 19: x <- F x, P <- F P F' + Q."""
        F = transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + process_noise(dt, sigma_accel)
        self.t += float(dt)
        return self

    def update(self, z: np.ndarray, R: np.ndarray):
        """Paper eq 26 with H = I2, so K = P (P + R)^-1."""
        z = np.asarray(z, float).reshape(2)
        S = self.P + np.asarray(R, float).reshape(2, 2)
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(2) - K) @ self.P
        # Force symmetry. (I-K)P is symmetric in exact arithmetic but drifts
        # over hundreds of updates, and an asymmetric P eventually produces a
        # negative Mahalanobis distance that silently breaks the gate.
        self.P = 0.5 * (self.P + self.P.T)
        self.hits += 1
        self.misses = 0
        return self

    def mahalanobis(self, z: np.ndarray, R: np.ndarray) -> float:
        """Normalised residual, for association gating."""
        z = np.asarray(z, float).reshape(2)
        S = self.P + np.asarray(R, float).reshape(2, 2)
        d = z - self.x
        return float(np.sqrt(max(d @ np.linalg.solve(S, d), 0.0)))

    def forecast(self, dt: float, n_steps: int, sigma_accel: float):
        """Paper eq 27: predicted state and covariance for l = 1..n_steps.

        The accumulated-Q sum is written as the recursion it is, rather than
        eq 27's explicit sum over F^j Q (F^T)^j: identical result, and it does
        not lose precision for large l.
        """
        F = transition(dt)
        Q = process_noise(dt, sigma_accel)
        xs, Ps, x, P = [], [], self.x.copy(), self.P.copy()
        for _ in range(int(n_steps)):
            x = F @ x
            P = F @ P @ F.T + Q
            xs.append(x.copy())
            Ps.append(P.copy())
        return np.array(xs), np.array(Ps)

    def gate_delay_s(self, n_sigma: float = 3.0, floor_m: float = 0.0) -> tuple:
        """(predicted delay, sigma) for the Gaussian range gate of paper eq 9.

        This is the handoff back to the coarse stage and into the fine stage's
        delay mask. `floor_m` keeps the gate at least one resolution cell wide:
        a converged filter can report a few metres of sigma, and a gate that
        tight will reject the target the moment the model is slightly wrong.
        """
        s = max(float(self.sigma()[0]), float(floor_m))
        return self.excess_m / C, n_sigma * s / C


class MultiTracker:
    """Nearest-neighbour association with Mahalanobis gating, M-of-N confirm.

    Deliberately not a JPDA or an MHT. The scene has three taps by
    construction (handoff section 5) and the ambiguity that actually hurts is
    the Doppler alias, which association cannot fix and prediction can.
    """

    def __init__(self, sigma_accel: float = 3.0, gate_sigma: float = 4.0,
                 confirm_hits: int = 3, drop_misses: int = 4):
        self.sigma_accel = float(sigma_accel)
        self.gate_sigma = float(gate_sigma)
        self.confirm_hits = int(confirm_hits)
        self.drop_misses = int(drop_misses)
        self.tracks: list = []
        self._next = 0
        self.t = None

    def step(self, t: float, zs, Rs):
        """One CPI: predict every track to `t`, associate, update, manage.

        `zs` is a list of [excess_m, range_rate_mps] measurements and `Rs` the
        matching 2x2 covariances. Returns the confirmed tracks.
        """
        t = float(t)
        dt = 0.0 if self.t is None else t - self.t
        if dt > 0:
            for tr in self.tracks:
                tr.predict(dt, self.sigma_accel)
        self.t = t

        zs = [np.asarray(z, float).reshape(2) for z in zs]
        Rs = [np.asarray(R, float).reshape(2, 2) for R in Rs]
        used = set()

        # Greedy on the global best pair first, so a strong detection is not
        # captured by whichever track happens to be first in the list.
        pairs = []
        for i, tr in enumerate(self.tracks):
            for j, (z, R) in enumerate(zip(zs, Rs)):
                d = tr.mahalanobis(z, R)
                if d <= self.gate_sigma:
                    pairs.append((d, i, j))
        pairs.sort()
        hit_tracks = set()
        for d, i, j in pairs:
            if i in hit_tracks or j in used:
                continue
            self.tracks[i].update(zs[j], Rs[j])
            self.tracks[i].history.append((t, self.tracks[i].x.copy()))
            hit_tracks.add(i)
            used.add(j)

        for i, tr in enumerate(self.tracks):
            if i not in hit_tracks:
                tr.misses += 1
            if tr.hits >= self.confirm_hits:
                tr.confirmed = True

        for j, (z, R) in enumerate(zip(zs, Rs)):
            if j in used:
                continue
            self.tracks.append(Track(tid=self._next, x=z.copy(), P=R.copy(), t=t,
                                     history=[(t, z.copy())]))
            self._next += 1

        self.tracks = [tr for tr in self.tracks if tr.misses < self.drop_misses]
        return [tr for tr in self.tracks if tr.confirmed]

    def best(self):
        """The track the gate and the dashboard should use, or None.

        Confirmed first, then most hits, then tightest range covariance. The
        last key is not cosmetic: with a plain `max` on hits alone, a set of
        tracks all sitting at one hit ties, `max` returns whichever came
        first, and a stale track that never associated again can hold the
        gate for the whole pass. That is exactly what happened when the
        coarse stage's Doppler ambiguity was blocking association - the
        symptom was a track frozen at its seed value with h1 forever.
        """
        cand = [tr for tr in self.tracks if tr.confirmed] or self.tracks
        if not cand:
            return None
        return max(cand, key=lambda tr: (tr.hits, -float(tr.P[0, 0])))


def measurement_cov(sigma_delay_s: float, sigma_doppler_hz: float, fc: float
                    ) -> np.ndarray:
    """R in (excess metres, metres per second) from delay and Doppler sigmas.

    Both conversions are linear scalings - excess = c*tau and Rdot = -lambda*fd
    - so the covariance transforms by the squared scale and stays diagonal.
    """
    s_r = float(sigma_delay_s) * C
    s_v = float(sigma_doppler_hz) * C / float(fc)
    return np.diag([s_r ** 2, s_v ** 2])
