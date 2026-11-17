"""Forecast the Twitch viewers of a big game launch, hour by hour, from its
first hour.

The program reads one case file per run (the grader hands it a single
<input>/<id>.in) and writes one forecast file per case to the output
directory: for hours 2..24 of the release day, five quantiles of that
hour's concurrent viewers (levels 0.05, 0.25, 0.5, 0.75, 0.95).

How it forecasts
---------------
The case ships with the hourly viewer curves of past big launches
(`comparables`), each from 14 days before its release day to 3 days
after. For every past launch, the shape of its release day is expressed
relative to its own first hour: the log ratio of each later hour's
viewers to the first hour's. A launch that is already huge in hour 1
grows little afterwards; a launch that is still small in hour 1 can
grow several-fold before the day is out. So for each hour, the log
ratio is regressed on the log of the first hour's print across the
past launches, weighting each launch by how close its first-hour size
is to the target's, and the target's own first-hour print picks its
place on that line. The median is shrunk 10% toward flat persistence
(the house baseline's shape), because the regression's growth is the
noisiest part of the fit.

The spread is the scatter of the past launches around the fitted line,
taken from the launches whose first-hour size is close to the target's
(within 1.5 in log), floored at 0.10 and scaled by 0.9: wide bands
cost pinball loss at every hour, so the band is kept honest rather
than generous. Medians and spreads are smoothed over neighbouring
hours (3-hour moving average). The five quantiles are the median
shifted by the standard normal's quantiles times the spread, in log
space.

Everything is derived from the case file at run time: no curve is
hard-coded, so the same program forecasts any launch the case hands
it. Standard library only, no network, no randomness, no clock: two
runs on the same case write identical bytes.

Input:  $MAKEFUN_INPUT_DIR (default /input), every <id>.in, name order.
Output: $MAKEFUN_OUTPUT_DIR (default /output), <id>.out per case:
        {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]},
        23 rows, hours 2..24, each q a finite number above 0 and at
        most 1e9, non-decreasing, at most 65,536 bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at the scored levels.
NORMAL_Q = (-1.645, -0.674, 0.0, 0.674, 1.645)
FIRST_HOUR = 1
LAST_HOUR = 24
# How far the median is pulled toward flat persistence (1.0 = no pull).
MEDIAN_SHRINK = 0.9
# How far the spread is pulled toward narrow (1.0 = no pull).
SPREAD_SHRINK = 0.9
# Smallest spread allowed, in log units.
SPREAD_FLOOR = 0.10
# Past launches counted as "near" in size: |ln(p1) - ln(target p1)| <= this.
NEAR_LOG = 1.5
# Fallback spread when a case carries too little to fit (log units per
# square-root hour, the house baseline's widening).
FALLBACK_SIGMA = 0.15


def epoch_seconds(text):
    """An ISO 8601 UTC timestamp such as 2026-11-19T00:00:00Z, as seconds."""
    moment = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    return int(moment.replace(tzinfo=timezone.utc).timestamp())


def half_up_two_places(value):
    """Round to 2 decimals, halves away from zero, as a JSON number."""
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100.0
    return int(rounded) if rounded.is_integer() else rounded


def median_of(values):
    """Middle value of a sorted list (average of the two middles if even)."""
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def spread_of(values):
    """Population standard deviation of a list of numbers."""
    n = len(values)
    centre = sum(values) / n
    return math.sqrt(sum((v - centre) ** 2 for v in values) / n)


def weighted_line(xs, ys, ws):
    """Weighted least-squares line y = a + b*x; ws holds one weight per point."""
    total = sum(ws)
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total
    bottom = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    slope = 0.0
    if bottom:
        slope = (sum(w * (x - mean_x) * (y - mean_y)
                     for w, x, y in zip(ws, xs, ys)) / bottom)
    return mean_y - slope * mean_x, slope


def release_day_hours(launch):
    """A past launch's release-day curve as {hour: viewers}, hours 1..24.

    `launch` is one comparable: its `day_start` is the UTC midnight of
    its release date and its `hourly` points are one per hour (some
    hours may be missing for a quiet category). Hours outside the day
    are ignored; a point of 0 viewers cannot anchor a log ratio and
    is dropped.
    """
    start = epoch_seconds(launch["day_start"])
    hours = {}
    for point in launch["hourly"]:
        viewers = point.get("viewers")
        if not viewers:
            continue
        offset = (epoch_seconds(point["from"]) - start) // 3600
        if 1 <= offset + 1 <= LAST_HOUR:
            hours[offset + 1] = viewers
    return hours


def fit_shape(comparables, first_print):
    """Per-hour median log ratio and spread, fitted on the past launches.

    Returns ({hour: median}, {hour: spread}) for hours 2..24. The median
    is the size-weighted regression line evaluated at the target's
    first-hour print (clamped to the launches' range so it never
    extrapolates), shrunk toward flat. The spread is the scatter of the
    launches near the target's size around that line.
    """
    # Per hour: the (log first-hour print, log ratio) points to fit.
    points = {h: [] for h in range(FIRST_HOUR + 1, LAST_HOUR + 1)}
    sizes = []
    for launch in comparables:
        day = release_day_hours(launch)
        first = day.get(FIRST_HOUR)
        if not first:
            continue
        sizes.append(math.log(first))
        for hour in range(FIRST_HOUR + 1, LAST_HOUR + 1):
            later = day.get(hour)
            if later:
                points[hour].append((math.log(first), math.log(later / first)))
    if not sizes:
        return None, None
    low, high = min(sizes), max(sizes)
    target_log = min(max(math.log(first_print), low), high)

    medians = {}
    spreads = {}
    for hour in range(FIRST_HOUR + 1, LAST_HOUR + 1):
        sample = points[hour]
        if len(sample) >= 5:
            # Weight each past launch by size proximity to the target:
            # a launch one log unit farther away counts e^-1 as much.
            # The target's own size band drives the line; distant sizes
            # only steady it. (Unweighted, the small launches pulled the
            # line off for the biggest targets.)
            weights = [math.exp(-abs(x - target_log) / NEAR_LOG)
                       for x, _ in sample]
            intercept, slope = weighted_line([x for x, _ in sample],
                                             [y for _, y in sample], weights)
            medians[hour] = MEDIAN_SHRINK * (intercept + slope * target_log)
            leftovers = [(x, y - (intercept + slope * x)) for x, y in sample]
            near = [r for x, r in leftovers if abs(x - target_log) <= NEAR_LOG]
            pool = near if len(near) >= 5 else [r for _, r in leftovers]
            spreads[hour] = max(SPREAD_FLOOR, spread_of(pool)) * SPREAD_SHRINK
        elif sample:
            ratios = sorted(y for _, y in sample)
            medians[hour] = MEDIAN_SHRINK * median_of(ratios)
            spreads[hour] = max(SPREAD_FLOOR, spread_of(ratios)) * SPREAD_SHRINK
        else:
            medians[hour] = 0.0
            spreads[hour] = FALLBACK_SIGMA * math.sqrt(hour - 1)
    return medians, spreads


def smooth(values):
    """3-hour moving average over the hour keys of a {hour: value} map."""
    out = {}
    for hour in sorted(values):
        near = [values[k] for k in (hour - 1, hour, hour + 1) if k in values]
        out[hour] = sum(near) / len(near)
    return out


def first_hour_print(case):
    """Hour 1's print, floored at 1 viewer; falls back to the last
    history point, then to 1, if the case carries no hour-1 row."""
    for row in case.get("known", []):
        if row.get("hour") == FIRST_HOUR:
            return max(1, row["viewers"])
    history = case.get("history") or []
    if history:
        return max(1, history[-1]["viewers"])
    return 1


def forecast_case(case):
    """The forecast object for one case: {"hours": [...]}."""
    if list(case.get("levels", [])) != list(LEVELS):
        raise ValueError("case %s asks for levels %s, not %s"
                         % (case.get("id"), case.get("levels"), list(LEVELS)))
    first_print = first_hour_print(case)
    medians, spreads = fit_shape(case.get("comparables") or [], first_print)
    if medians is None:
        medians = {h: 0.0 for h in range(FIRST_HOUR + 1, LAST_HOUR + 1)}
        spreads = {h: FALLBACK_SIGMA * math.sqrt(h - 1)
                   for h in range(FIRST_HOUR + 1, LAST_HOUR + 1)}
    medians = smooth(medians)
    spreads = smooth(spreads)
    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        centre = medians.get(hour, 0.0)
        width = spreads.get(hour, FALLBACK_SIGMA * math.sqrt(hour - 1))
        quantiles = []
        for z in NORMAL_Q:
            value = first_print * math.exp(centre + z * width)
            value = min(max(value, 0.01), 1e9)
            quantiles.append(half_up_two_places(value))
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"hour": hour, "q": quantiles})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
