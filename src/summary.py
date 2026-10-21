"""One-page current sentiment outlook. Rule-based (always works); optional LLM polish if ANTHROPIC_API_KEY set."""
import os
import numpy as np
import pandas as pd


def _label(x):
    return ("strongly bullish" if x > .3 else "bullish" if x > .12 else "mildly bullish" if x > .04 else
            "neutral" if x > -.04 else "mildly bearish" if x > -.12 else "bearish" if x > -.3 else "strongly bearish")


def build_summary(cfg, df, posts, news, fc, metrics, synthetic=False):
    a = cfg["asset"]["name"]
    last = df.iloc[-1]
    s24, s7d = df.sent_w.iloc[-24:].mean(), df.sent_w.iloc[-168:].mean()
    pr24, pr7 = df.pos_ratio.iloc[-24:].mean(), df.neg_ratio.iloc[-24:].mean()
    vz = df.vol_z.iloc[-24:].mean()
    chg24 = df.close.iloc[-1] / df.close.iloc[-25] - 1
    p5 = fc.loc[fc.horizon == fc.horizon.max()].iloc[0]
    p0 = fc.loc[fc.horizon == 0].iloc[0]
    d5 = p5.pred_price / p5.base_close - 1
    L = [f"# {a} X sentiment outlook", f"_As of {df.index[-1] + pd.Timedelta(cfg['bar']):%Y-%m-%d %H:%M} UTC_", ""]
    if synthetic:
        L += ["> **DEMO MODE: synthetic data, not a real market view.**", ""]
    L += ["## Headline",
          f"Weighted X sentiment over the last 24h is **{_label(s24)}** ({s24:+.2f}), versus {s24 - s7d:+.2f} against its 7-day "
          f"average ({s7d:+.2f}). Price moved {chg24:+.1%} over the same 24h.", "",
          "## Crowd behaviour",
          f"- Positive posts: {pr24:.0%}, negative posts: {pr7:.0%} (24h average).",
          f"- Posting volume is {'elevated' if vz > .5 else 'subdued' if vz < -.5 else 'near normal'} (z-score {vz:+.1f}).",
          f"- Sentiment/price divergence: " + (
              "sentiment is rising while price is falling, a possible contrarian-bullish setup." if s24 > s7d and chg24 < 0 else
              "sentiment is falling while price is rising, which can precede a pullback." if s24 < s7d and chg24 > 0 else
              "sentiment and price are moving in the same direction."), ""]
    if posts is not None and len(posts):
        cut = posts.ts.max() - pd.Timedelta(hours=24)
        top = posts[posts.ts >= cut].assign(eng=lambda d: d.likes + 2 * d.reposts).nlargest(3, "eng")
        L += ["## Most engaged recent posts"] + [f"- @{r.author}: \"{r.text[:140]}\" ({r.likes} likes)" for r in top.itertuples()] + [""]
    if news is not None and len(news):
        L += ["## News context (Tavily)"] + [f"- {r.title}" for r in news.sort_values("ts", ascending=False).head(5).itertuples()] + [""]
    L += ["## Model forecast", f"- Last closed price: ${p0.base_close:,.0f}",
          f"- n+0: ${p0.pred_price:,.0f} ({p0.pred_ret:+.2%}); n+{int(p5.horizon)}: ${p5.pred_price:,.0f} ({d5:+.2%})"]
    if metrics:
        s = metrics["strategy"]
        L += [f"- Walk-forward track record: Sharpe {s['sharpe']:.2f}, Sortino {s['sortino']:.2f}, "
              f"max drawdown {s['max_drawdown']:.1%} (out-of-sample, after fees)."]
    L += ["", "_Caveat: sentiment-only models have low signal-to-noise; treat forecasts as probabilistic, not advice._"]
    md = "\n".join(L)
    return _polish(md) if os.getenv("ANTHROPIC_API_KEY") else md


def _polish(md):
    try:
        import anthropic
        c = anthropic.Anthropic()
        r = c.messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), max_tokens=900, messages=[{
            "role": "user", "content": "Rewrite this market sentiment brief as a crisp one-page analyst note. Keep every number "
                                       "and the caveat unchanged, add nothing new:\n\n" + md}])
        return r.content[0].text
    except Exception as e:
        print(f"[summary] LLM polish skipped: {e}")
        return md
