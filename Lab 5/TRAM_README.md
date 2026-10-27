# Roosevelt Island Tram Watcher

A camera in the window watches the tram's cable. A small neural network, trained on our own
labeled photos, decides each second whether a cabin is in view. A Bayes filter smooths that
into clean "pass" events. A semi-Markov model of the gaps between passes predicts the next one.
The Mini PiTFT shows a clock, the live status and the prediction.

It's an unofficial hobby project, not affiliated with the tram's operator. Predictions are
estimates; check the official schedule.

---

## Files

| File | What it does |
|---|---|
| `tram_common.py` | Shared code: camera, crop, our exposure control, change detection, model loading, service hours. |
| `tram_collect.py` | `--setup` marks the tram's box; then collects crops (every 4 s, 45-min sessions). |
| `tram_burst.py` | Press Enter when you see a tram: saves the last 3 s plus 50 photos every 0.5 s. |
| `tram_label.py` | Fast keyboard labeling: `t` tram, `n` no tram. Modes to find trams and confusing images; `--stats` for counts. |
| `tram_train.py` | Fine-tunes a pretrained image classifier, with pseudo-labeling (semi-supervised). |
| `tram_watch.py` | Live tracker: classifier → Bayes filter → pass log → prediction → `tram_state.json`. |
| `tram_display.py` | PiTFT clock and status screen, reading `tram_state.json`. |

Made by the scripts: `tram_config.json` (box, rotation, exposure), `tram_data/` (crops),
`tram_labels.csv`, `tram_model.pt`, `tram_events.jsonl` (one line per pass), `tram_state.json` (live).

---

## 1. Set up the camera (once)

Lens flat against the glass, with a dark cloth or cardboard tube around it so the room doesn't
reflect into the picture. Then, from the Pi desktop:

```bash
python tram_collect.py --setup --rotate none
```

Click one corner, then the opposite corner of a box around the cable (below the bridge deck,
above the dome, tower to where the cable disappears), and press Enter. Check `tram_preview.jpg`.
Only this box is ever saved or analyzed: no street, no cars, no windows.

```bash
python tram_collect.py --exposure-test
```

For 60 s it shows the box's brightness. Our exposure control meters only the box, so at night it
lengthens the exposure until the sky and cable are visible, instead of exposing for the bright
city lights. If it says the webcam refused manual exposure, run `v4l2-ctl --list-ctrls`.

## 2. Collect images

```bash
tmux new -s tram
python tram_collect.py --minutes 45     # one session now
python tram_collect.py --plan           # 06:15, 12:30, 18:15, 22:30, then stops
```

That's a crop every 4 s: about 675 per session, about 2,700 for all four kinds of light.

## 3. Label

```bash
python tram_label.py --find-trams    # most-changed images first: finds trams fast
python tram_label.py                 # random images: mostly "no tram"
```

Keys: **t** tram · **n** no tram · **s** skip · **b** back · **q** save and quit.
Saved after every key. A good first batch: about 50 trams via `--find-trams`, plus about 150
random. Label a cabin as "tram" even if it's only partly in the box or dim at night. Include
night images.

### Get more tram photos: bursts

Trams are only in view a few % of the time, so a day of collecting gives only a few dozen tram
photos. When you're home, catch passes by hand (the camera can only be used by one program, so
make sure `tram_collect.py` isn't running):

```bash
python tram_burst.py           # ENTER when you see a tram; q + ENTER to quit
python tram_label.py --bursts  # label just the burst photos, in time order
python tram_label.py --stats   # images / labeled / trams per day and hour
```

Catch some at dusk and at night too, not only in daylight; `--stats` shows the balance.
Training groups photos into 3-minute blocks, so near-identical burst photos never end up
on both sides of the train/test split (which would make the score look better than it is).

## 4. Train

```bash
python tram_train.py
```

Round 1 fine-tunes `yolo26n-cls` (pretrained on ImageNet) on your labels. Then it pseudo-labels
the unlabeled crops it's ≥95% sure about, and round 2 trains on both. Scores are reported only
on held-back images **you** labeled. On the Pi this takes 10–30 min; it's faster on a laptop
(copy `tram_data/`, `tram_labels.csv` and the `tram_*.py` files over, run it, then copy
`tram_model.pt` and `tram_model.json` back).

Then improve it where it's weakest:

```bash
python tram_label.py --uncertain     # the images the model is least sure about
python tram_train.py
```

Repeat until **recall** (trams caught) and **precision** (correct alarms) stop improving.
Ignore accuracy on its own: trams are rare, so "never a tram" already scores about 98%. The
trainer repeats tram photos so the two classes are closer to balanced, and keeps the model
with the best balance of recall and precision (F1).

## 5. Run it live

```bash
tmux new -s tram
python tram_watch.py                 # add --show on the Pi desktop to watch it think
```

Second tmux window (**Ctrl-B then C**):

```bash
python tram_display.py               # PiTFT; or --preview to write tram_display.png
```

PiTFT: **top button** switches clock ↔ stats; **bottom button** turns the backlight off/on.
If the PiTFT libraries are missing: `uv pip install adafruit-circuitpython-rgb-display adafruit-blinka`.

`python tram_watch.py --predict` prints the current prediction and the learned gaps, without the camera.

### Directions and island arrival (optional)

Each pass logs which way the change moved across the box (`left to right` / `right to left`).
After watching one pass yourself, name them in `tram_config.json`, and optionally say how many
seconds after leaving the box a cabin reaches the island:

```json
"directions": {"left_to_right": "to Island", "right_to_left": "to Manhattan"},
"arrive_after_s": {"to Island": 60}
```

---

## How it works (the robotics part)

1. **Detector:** the image classifier gives p = P(tram | this one frame). Single frames are
   noisy (glare, birds, a bad exposure).
2. **Bayes filter:** a two-state hidden Markov model, *tram in view* / *no tram*. Each second it
   *predicts* (a cabin stays about 40 s, and one arrives every 7.5–15 min) and then *corrects*
   with p. It's the same predict–update loop as robot localization. A pass starts when belief
   is above 80% and ends below 20%, so a single odd frame can't create or split a pass.
3. **Semi-Markov / renewal model:** the gaps between passes are grouped by time of day
   (weekday morning, weekend night, …). Given how long it has been since the last pass, it
   keeps only the past gaps longer than that (the conditional distribution), which gives the
   expected next pass, a 10–90% range and P(within 5 min). Until enough gaps are learned, it
   falls back to the posted headways (about 7.5 min at peak, 15 min otherwise).

## Privacy

Only the cable box is stored, small, and it's pointed at the sky. No audio. Nothing is
published, and no photos are saved while running live. Delete `tram_data/` when you no longer
need it for training.

## Later: API

`tram_state.json` already holds everything a website or API needs. The next step is a tiny web
server on the Pi that serves it (and a page that shows it), reachable via Tailscale.
