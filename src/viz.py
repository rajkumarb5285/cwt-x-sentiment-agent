"""Plotly charts: sentiment vs price over time, forecast fan, pred vs actual, equity curves."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fig_sentiment_vs_price(df, days=30, smooth=6):
    d = df.iloc[-int(days * 24):] if len(df) > days * 24 else df
    sent = d.sent_w.rolling(smooth, min_periods=1).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04,
                        specs=[[{"secondary_y": True}], [{}]])
    fig.add_trace(go.Scatter(x=d.index, y=d.close, name="Price", line=dict(color="#185FA5", width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=sent, name=f"Sentiment ({smooth}-bar avg)",
                             line=dict(color="#D85A30", width=1.4)), 1, 1, secondary_y=True)
    fig.add_trace(go.Bar(x=d.index, y=d.n_posts, name="Post volume", marker_color="#888780"), 2, 1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", secondary_y=True, row=1, col=1, zeroline=True)
    fig.update_yaxes(title_text="Posts/bar", row=2, col=1)
    fig.update_layout(title="X sentiment vs price", height=520, margin=dict(l=40, r=40, t=50, b=30),
                      legend=dict(orientation="h", y=1.08))
    return fig


def fig_forecast(df, fc, hist_bars=72):
    h = df.iloc[-hist_bars:]
    last_ts = h.index[-1]
    xs = [last_ts] + list(fc.target_ts)
    # target_ts is the OPEN of the bar whose close we predict; plot at the bar's close time
    bar = fc.target_ts.iloc[1] - fc.target_ts.iloc[0] if len(fc) > 1 else pd.Timedelta("1h")
    xs = [last_ts + bar] + [t + bar for t in fc.target_ts]
    ys = [float(h.close.iloc[-1])] + list(fc.pred_price)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index + bar, y=h.close, name="Actual close", line=dict(color="#185FA5")))
    fig.add_trace(go.Scatter(x=xs, y=ys, name="Forecast n+0..n+5", mode="lines+markers",
                             line=dict(color="#D85A30", dash="dash")))
    fig.update_layout(title="Price forecast, next 6 bars", height=380, margin=dict(l=40, r=20, t=50, b=30),
                      legend=dict(orientation="h", y=1.1))
    return fig


def fig_pred_vs_actual(oos, h):
    c = oos.close
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oos.index, y=c * np.exp(oos[f"y{h}"]), name="Actual", line=dict(color="#185FA5", width=1)))
    fig.add_trace(go.Scatter(x=oos.index, y=c * np.exp(oos[f"pred_y{h}"]), name="Predicted (OOS)",
                             line=dict(color="#D85A30", width=1)))
    fig.update_layout(title=f"Out-of-sample prediction vs actual, n+{h}", height=360, margin=dict(l=40, r=20, t=50, b=30))
    return fig


def fig_equity(bt):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt.equity, name="Sentiment strategy (net of fees)", line=dict(color="#D85A30")))
    fig.add_trace(go.Scatter(x=bt.index, y=bt.bh_equity, name="Buy & hold", line=dict(color="#888780")))
    fig.update_layout(title="Walk-forward equity curve (out-of-sample only)", height=360,
                      margin=dict(l=40, r=20, t=50, b=30), legend=dict(orientation="h", y=1.1))
    return fig


def fig_horizon_skill(hm):
    fig = go.Figure(go.Bar(x=hm.horizon, y=hm.dir_acc * 100, marker_color="#0F6E56"))
    fig.add_hline(y=50, line_dash="dot", line_color="#888780")
    fig.update_layout(title="Directional accuracy by horizon (%)", height=300, margin=dict(l=40, r=20, t=50, b=30),
                      yaxis_range=[40, 65])
    return fig


def save_all(figs, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    for name, f in figs.items():
        f.write_html(os.path.join(outdir, f"{name}.html"), include_plotlyjs="cdn")
