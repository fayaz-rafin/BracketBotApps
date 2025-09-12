# /// script
# dependencies = [
#   "bbos",
#   "bokeh"
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
# main.py
from bbos import Reader
from bokeh.plotting import figure
from bokeh.palettes import Category10, Category20, Viridis256
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.server.server import Server
import numpy as np, time

ROLLOVER = 4000
DT_MS = 1  # update period (ms)

def _sample_colors(n):
    """Return n visually distinct hex colors."""
    if n <= 10:
        return Category10[10][:n]
    if n <= 20:
        return Category20[20][:n]
    # Evenly sample Viridis256
    if n <= 1:
        return [Viridis256[128]]
    idxs = [round(i * (len(Viridis256) - 1) / (n - 1)) for i in range(n)]
    return [Viridis256[i] for i in idxs]

def make_document(doc):
    r = Reader("imu.orientation")
    while not r.ready():
        time.sleep(0.001)
    d0, dt = r.data, r.data.dtype

    # Build (label, extractor) pairs
    series = []
    for name in dt.names:
        if name == "timestamp":
            continue
        shape = getattr(dt[name], "shape", ())
        if shape == () or shape == (1,):
            series.append((name, (lambda nm=name: (lambda d: float(d[nm])))()))
        else:
            n = int(np.prod(shape))
            for i in range(n):
                series.append((str(i), (lambda nm=name, i=i: (lambda d: float(np.ravel(d[nm])[i])))()))

    # Assign distinct colors
    colors = _sample_colors(len(series))
    color_map = {label: colors[i] for i, (label, _) in enumerate(series)}

    # Time zero
    t0_wall = np.datetime64("now", "ms")
    has_ts  = ("timestamp" in dt.names)
    ts0     = int(d0["timestamp"]) if has_ts else None

    # Plot + sources
    p = figure(title="imu.orientation", x_axis_label="Δt (s)", y_axis_label="value",
               tools="pan,wheel_zoom,box_zoom,reset",
               sizing_mode="stretch_both",
               output_backend="webgl")

    hover = HoverTool(
        tooltips=[
            ("t (s)", "@dt_s{0.000}"),
            ("value", "@val{0.000}"),
        ],
        mode="vline"   # or "vline" if you want crosshair-style inspection
    )
    p.add_tools(hover)

    sources = {}
    for label, _ in series:
        src = ColumnDataSource(data=dict(dt_s=[], val=[]))
        sources[label] = src
        p.circle("dt_s", "val", source=src, size=3, alpha=0.9,
                 color=color_map[label], legend_label=label)

    p.legend.click_policy = "hide"
    p.legend.location = "top_left"
    doc.add_root(p)

    def tick():
        if r.ready():
            d = r.data
            if has_ts:
                dt_s = (int(d["timestamp"]) - ts0) / 1e9
            else:
                dt_s = float((np.datetime64("now", "ms") - t0_wall) / np.timedelta64(1, "ms")) / 1000.0
            for label, extract in series:
                sources[label].stream({"dt_s": [dt_s], "val": [extract(d)]}, rollover=ROLLOVER)

    doc.add_periodic_callback(tick, DT_MS)

if __name__ == "__main__":
    server = Server({"/": make_document}, port=5006,
                    allow_websocket_origin=["localhost:5006","127.0.0.1:5006"])
    server.start()
    print("Bokeh app at http://localhost:5006/")
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
