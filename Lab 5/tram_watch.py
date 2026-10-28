"""
tram_watch.py: the live tram tracker.

    python tram_watch.py              # run until Ctrl-C (start it in tmux)
    python tram_watch.py --show       # with a window: the crop, the model's guess, the belief
    python tram_watch.py --predict    # no camera: just print the current prediction from the log

Every second:
  1. The camera crop goes through the trained detector (tram_det.pt) -> where the tram is and
     how sure it is; or, without a detector, the classifier -> just "how sure" (noisy either way:
     glare, birds, a bad frame).
  2. A two-state hidden Markov model (a Bayes filter, the same idea robots use to track where
     they are) combines p with what it believed a second ago. One odd frame barely moves it;
     several in a row do. Passes start when belief rises above 80% and end below 20%.
  2b. With the detector: the grip point is projected onto the cable (tram_cable.py), and a
     constant-velocity Kalman filter tracks position (0 = tower, 1 = station) and speed. It
     keeps predicting through missed frames and past the trees, giving live "reaches the
     station at ..." estimates.
  3. Each finished pass is logged to tram_events.jsonl with its start, end, length, and
     direction (from which way the changed area moved across the box).
  4. The gaps between passes form a renewal / semi-Markov model, grouped by time of day.
     Given how long it has been since the last pass, it predicts the next one with a range
     and the chance it comes within 5 minutes.
  5. Everything is written to tram_state.json, which tram_display.py (PiTFT) and later the
     API read. No images are saved.
"""
import argparse
import datetime
import json
import os
import time

import cv2
import numpy as np

from tram_common import (EVENTS, STATE, Exposure, crop, grab, in_service, load_classifier,
                         load_config, load_detector, motion_x, open_cam, period)

TRACKS = "tram_tracks.jsonl"    # every pass's (time, progress) points: data for the timing model

ENTER, LEAVE = 0.8, 0.2         # belief thresholds (hysteresis: no flapping around 50%)
MIN_PASS_S = 3                  # shorter blips are ignored
MAX_GAP_S = 45 * 60             # longer gaps (closures, missed passes overnight) are not learned
SCHEDULE_S = {"peak": 7.5 * 60, "off-peak": 15 * 60}     # prior until real gaps are learned


# ---------- the Bayes filter ----------

class TramFilter:
    """P(tram in view), updated once per step from the classifier's p."""

    def __init__(self, dt, mean_in_view=40, mean_gap=7 * 60):
        self.dt, self.belief = dt, 0.05
        self.set_rates(mean_in_view, mean_gap)

    def set_rates(self, mean_in_view, mean_gap):
        self.mean_in_view = mean_in_view
        self.p_leave = min(0.5, self.dt / max(mean_in_view, self.dt))   # in view -> gone
        self.p_enter = min(0.5, self.dt / max(mean_gap, self.dt))       # gone -> in view

    def update(self, p):
        p = float(np.clip(p, 0.02, 0.98))
        prior = self.belief * (1 - self.p_leave) + (1 - self.belief) * self.p_enter   # predict
        num = prior * p                                                               # correct
        self.belief = num / (num + (1 - prior) * (1 - p))
        return self.belief


# ---------- the Kalman filter: where along the cable, and how fast ----------

def progress(point, cable):
    """Project a grip point (0..1 crop coords) onto the cable: 0 = tower end, 1 = station end."""
    a, b = np.array(cable[0]), np.array(cable[1])
    ab = b - a
    return float(np.dot(np.array(point) - a, ab) / np.dot(ab, ab))


class CableKalman:
    """Constant-velocity Kalman filter on progress s along the cable (state = [s, v]).

    predict(dt): the cabin keeps its speed, with a little uncertainty added (it speeds up
    leaving a station and slows down arriving). update(z): blend in a measured position,
    trusting it according to how noisy detections are. When the detector misses a frame, or
    the cabin is hidden behind the trees near the station, we just keep predicting: that is
    how the filter "sees" the cabin reach the station even though the camera can't."""

    def __init__(self, s0, meas_sd=0.02, accel_sd=0.003):
        self.x = np.array([s0, 0.0])
        self.P = np.diag([meas_sd ** 2, 0.02 ** 2])     # unsure of the speed at first
        self.R = meas_sd ** 2
        self.q = accel_sd ** 2
        self.n = 1

    def predict(self, dt):
        F = np.array([[1, dt], [0, 1]])
        Q = self.q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.array([1.0, 0.0])
        innovation = z - H @ self.x
        S = H @ self.P @ H + self.R
        if innovation ** 2 > 16 * S and self.n > 3:      # >4 sigma: a bad detection, ignore it
            return
        K = self.P @ H / S
        self.x = self.x + K * innovation
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        self.n += 1

    @property
    def s(self):
        return float(self.x[0])

    @property
    def v(self):
        return float(self.x[1])

    def seconds_to(self, target):
        """Seconds until progress reaches `target` at the current speed, or None if not heading there."""
        if self.n < 3 or abs(self.v) < 1e-3 or (target - self.s) * self.v <= 0:
            return None
        return (target - self.s) / self.v


# ---------- the predictor ----------

def load_passes():
    if not EVENTS.exists():
        return []
    out = []
    with open(EVENTS) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "pass":
                out.append(e)
    return out


def learned_gaps(passes):
    """{period name: [gap seconds, ...]} from consecutive pass starts."""
    starts = sorted(datetime.datetime.fromisoformat(p["start"]) for p in passes)
    gaps = {}
    for a, b in zip(starts, starts[1:]):
        g = (b - a).total_seconds()
        if 20 < g <= MAX_GAP_S and in_service(a) and in_service(b):
            gaps.setdefault(period(a), []).append(g)
    return gaps


def schedule_prior(t):
    """Posted headways: ~7.5 min at rush hours and weekend mornings, ~15 min otherwise."""
    weekend, h = t.weekday() >= 5, t.hour
    peak = (not weekend and (7 <= h < 10 or 16 <= h < 20)) or (weekend and 6 <= h < 12)
    return SCHEDULE_S["peak" if peak else "off-peak"]


def predict(now, passes, gaps=None):
    """Next pass: estimate, 10-90% range, P(within 5 min), and what it is based on."""
    if not in_service(now):
        opens = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if opens < now:
            opens += datetime.timedelta(days=1)
        return {"status": "closed", "opens": opens.isoformat(timespec="minutes")}
    if not passes:
        return {"status": "no passes seen yet"}
    gaps = gaps if gaps is not None else learned_gaps(passes)
    last = max(datetime.datetime.fromisoformat(p["start"]) for p in passes)
    elapsed = (now - last).total_seconds()

    here = gaps.get(period(now), [])
    pool = here if len(here) >= 8 else [g for v in gaps.values() for g in v]
    basis = f"{len(here)} gaps ({period(now)})" if len(here) >= 8 else f"{len(pool)} gaps (all times)"

    if len(pool) >= 8:
        g = np.array(pool)
        survivors = g[g > elapsed] - elapsed          # gaps longer than what we've already waited
        if len(survivors) >= 3:
            lo, mid, hi = np.percentile(survivors, [10, 50, 90])
            return {"status": "ok", "since_last_s": round(elapsed),
                    "eta": (now + datetime.timedelta(seconds=mid)).isoformat(timespec="seconds"),
                    "low": (now + datetime.timedelta(seconds=lo)).isoformat(timespec="seconds"),
                    "high": (now + datetime.timedelta(seconds=hi)).isoformat(timespec="seconds"),
                    "p_within_5min": round(float(np.mean(survivors <= 300)), 2),
                    "typical_gap_s": round(float(np.median(g))), "based_on": basis}
        return {"status": "overdue", "since_last_s": round(elapsed),
                "typical_gap_s": round(float(np.median(g))), "based_on": basis}

    gap = schedule_prior(now)                         # not enough data yet: posted schedule
    remaining = gap - elapsed
    if remaining < 0:
        return {"status": "overdue", "since_last_s": round(elapsed), "typical_gap_s": gap,
                "based_on": "posted schedule (not enough passes learned yet)"}
    return {"status": "ok", "since_last_s": round(elapsed),
            "eta": (now + datetime.timedelta(seconds=remaining)).isoformat(timespec="seconds"),
            "p_within_5min": None, "typical_gap_s": gap,
            "based_on": "posted schedule (not enough passes learned yet)"}


# ---------- helpers ----------

def write_state(state):
    tmp = STATE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE)            # atomic: readers never see half a file


def direction_of(track, cfg):
    """Which way the cabin moved during a pass, from where the view changed (versus the empty
    background) over time."""
    pts = [(t, x) for t, x in track if x is not None]
    if len(pts) < 3:
        return None
    t, x = np.array(pts).T
    slope = np.polyfit(t - t[0], x, 1)[0]           # crop widths per second
    if abs(slope) < 0.002:
        return None
    raw = "left_to_right" if slope > 0 else "right_to_left"
    return cfg.get("directions", {}).get(raw, raw.replace("_", " "))


def log_pass(start, end, track, cfg, kf=None, points=None):
    """One finished pass. With the detector + Kalman filter it also records where the cabin was
    first and last seen along the cable, its speed, and the station arrival / departure times
    extrapolated past the edge of what the camera can see."""
    entry = {"event": "pass", "start": start.isoformat(timespec="seconds"),
             "end": end.isoformat(timespec="seconds"),
             "seconds": round((end - start).total_seconds())}
    if kf is not None and points and len(points) >= 3:
        ts = np.array([p[0] for p in points])
        ss = np.array([p[1] for p in points])
        speed = float(np.polyfit(ts - ts[0], ss, 1)[0])       # progress per second, whole pass
        entry.update({"first_s": round(float(ss[0]), 3), "last_s": round(float(ss[-1]), 3),
                      "speed": round(speed, 4)})
        if speed > 0.002:
            entry["direction"] = cfg.get("directions", {}).get("toward_station", "to Island")
            to_go = (1 - ss[-1]) / speed                   # remaining stretch hidden from view
            if 0 <= to_go < 120:
                entry["est_station_arrival"] = (end + datetime.timedelta(seconds=to_go)).isoformat(
                    timespec="seconds")
        elif speed < -0.002:
            entry["direction"] = cfg.get("directions", {}).get("toward_tower", "to Manhattan")
            since = (1 - ss[0]) / -speed                   # how long ago it left the station
            if 0 <= since < 120:
                entry["est_station_departure"] = (start - datetime.timedelta(seconds=since)).isoformat(
                    timespec="seconds")
        with open(TRACKS, "a") as f:
            f.write(json.dumps({"start": entry["start"], "points": [
                [round(t - points[0][0], 2), round(s_, 4)] for t, s_ in points]}) + "\n")
    else:
        entry["direction"] = direction_of(track, cfg)
        arrive = cfg.get("arrive_after_s", {}).get(entry["direction"]) if entry["direction"] else None
        if arrive is not None:
            entry["est_station_arrival"] = (end + datetime.timedelta(seconds=arrive)).isoformat(
                timespec="seconds")
    with open(EVENTS, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
    extra = ""
    if entry.get("est_station_arrival"):
        extra = f", reaches the station ~{entry['est_station_arrival'][11:19]}"
    if entry.get("est_station_departure"):
        extra = f", left the station ~{entry['est_station_departure'][11:19]}"
    print(f"{end:%H:%M:%S} pass: {entry['seconds']} s in view"
          + (f", {entry['direction']}" if entry.get("direction") else "") + extra, flush=True)
    return entry


# ---------- main loop ----------

def watch(args):
    cfg = load_config()
    find = load_detector() if cfg.get("cable") else None    # position mode needs both
    p_tram = None if find else load_classifier()
    print("sensor: " + ("detector + cable position (Kalman)" if find else
                        "classifier (presence only; train tram_det.pt and run tram_cable.py "
                        "for position)"), flush=True)
    cap = open_cam(cfg)
    exposure = Exposure(cap, cfg)
    passes = load_passes()
    gaps = learned_gaps(passes)
    durations = [p["seconds"] for p in passes if p.get("seconds")]
    filt = TramFilter(args.interval, mean_in_view=np.median(durations) if durations else 40)
    in_pass, start, track, last_pred = False, None, [], 0.0
    kf, points, last_t = None, [], time.time()
    background = None                 # running average of the crop with no tram in it
    print(f"watching (Ctrl-C to stop). {len(passes)} passes in the log so far.", flush=True)

    try:
        while True:
            t0 = time.time()
            now = datetime.datetime.now()
            try:
                img = crop(grab(cap, cfg), cfg)
            except RuntimeError as e:
                print("hiccup:", e, flush=True)
                time.sleep(args.interval)
                continue
            dt, last_t = t0 - last_t, t0

            exposure.update(img)
            box = None
            if find:
                p, box = find(img)
            else:
                p = p_tram(img)
            filt.set_rates(filt.mean_in_view, schedule_prior(now) if not gaps.get(period(now))
                           else float(np.median(gaps[period(now)])))
            b = filt.update(p)

            if b < LEAVE and not in_pass:   # learn what the empty sky looks like right now
                f = img.astype(np.float32)
                if background is None or background.shape != f.shape:
                    background = f
                else:
                    cv2.accumulateWeighted(f, background, 0.05)     # updates in place
            if not in_pass and b > ENTER:
                in_pass, start, track, kf, points = True, now, [], None, []
                print(f"{now:%H:%M:%S} tram in view", flush=True)
            if in_pass:
                track.append((t0, motion_x(img, background) if background is not None else None))
                if find:
                    if kf is not None:
                        kf.predict(dt)
                    if box is not None and p >= 0.3:
                        z = progress(((box[0] + box[2]) / 2, box[1]), cfg["cable"])
                        if kf is None:
                            kf = CableKalman(z)
                        else:
                            kf.update(z)
                        points.append((t0, z))
                if b < LEAVE:
                    in_pass = False
                    if (now - start).total_seconds() >= MIN_PASS_S:
                        passes.append(log_pass(start, now, track, cfg, kf, points))
                        gaps = learned_gaps(passes)
                        filt.mean_in_view = float(np.median([q["seconds"] for q in passes]))
                    else:
                        print(f"{now:%H:%M:%S} (blip ignored)", flush=True)

            if time.time() - last_pred >= 1 or in_pass:
                last = passes[-1] if passes else None
                today = now.date().isoformat()
                live = None
                if in_pass and kf is not None:
                    to_station, to_tower = kf.seconds_to(1.0), kf.seconds_to(0.0)
                    live = {"position": round(kf.s, 3), "speed": round(kf.v, 4),
                            "heading": "station" if kf.v > 0.002 else "tower" if kf.v < -0.002
                            else "stopped"}
                    if to_station is not None:
                        live["eta_station"] = (now + datetime.timedelta(seconds=to_station)).isoformat(
                            timespec="seconds")
                    if to_tower is not None:
                        live["eta_out_of_view"] = (now + datetime.timedelta(seconds=to_tower)).isoformat(
                            timespec="seconds")
                write_state({
                    "time": now.isoformat(timespec="seconds"),
                    "in_service": in_service(now),
                    "tram_in_view": in_pass,
                    "belief": round(b, 3), "model_p": round(p, 3),
                    "current_pass_started": start.isoformat(timespec="seconds") if in_pass else None,
                    "live": live,
                    "last_pass": last,
                    "passes_today": sum(1 for q in passes if q["start"].startswith(today)),
                    "next": predict(now, passes, gaps),
                })
                last_pred = time.time()

            if args.show:
                view = cv2.resize(img, None, fx=2, fy=2)
                H, W = view.shape[:2]
                if find and cfg.get("cable"):
                    (ax, ay), (bx, by) = cfg["cable"]
                    cv2.line(view, (int(ax * W), int(ay * H)), (int(bx * W), int(by * H)), (0, 160, 0), 1)
                if box is not None:
                    cv2.rectangle(view, (int(box[0] * W), int(box[1] * H)), (int(box[2] * W), int(box[3] * H)),
                                  (0, 0, 255) if p >= 0.3 else (128, 128, 128), 2)
                bar = int(W * b)
                cv2.rectangle(view, (0, H - 12), (bar, H), (0, 0, 255) if in_pass else (0, 200, 0), -1)
                txt = f"p {p:.2f}  belief {b:.2f}" + ("  TRAM" if in_pass else "")
                if in_pass and kf is not None:
                    txt += f"  s {kf.s:.2f} v {kf.v:+.3f}/s"
                cv2.putText(view, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.imshow("tram watch", view)
                if cv2.waitKey(1) == ord("q"):
                    break
            time.sleep(max(0.0, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    print("stopped", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between looks")
    parser.add_argument("--show", action="store_true", help="window with the crop and belief")
    parser.add_argument("--predict", action="store_true", help="print the prediction and exit")
    args = parser.parse_args()
    if args.predict:
        passes = load_passes()
        print(json.dumps(predict(datetime.datetime.now(), passes), indent=2))
        for name, g in sorted(learned_gaps(passes).items()):
            print(f"  {name}: {len(g)} gaps, median {np.median(g) / 60:.1f} min")
    else:
        watch(args)
