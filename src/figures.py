"""Draws every figure from the saved results (no recomputation)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from common import RES, FIG, ROLES, ELECTION

S1, S2 = "#2a78d6", "#eb6834"                 # validated categorical slots 1-2
SURF, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "font.size": 9.5, "legend.frameon": False, "figure.dpi": 150,
})
J = lambda f: json.load(open(os.path.join(RES, f)))
save = lambda fig, name: (fig.savefig(os.path.join(FIG, name), bbox_inches="tight"), plt.close(fig))


def title(fig, t, sub=None):
    """Reserve a fixed header band (in inches) above the constrained layout."""
    h = fig.get_size_inches()[1]
    band = (0.62 if sub else 0.4) / h
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - band))
    fig.text(0.008, 1 - 0.08 / h, t, ha="left", va="top", fontweight="bold", fontsize=13)
    if sub:
        fig.text(0.008, 1 - 0.36 / h, sub, ha="left", va="top", color=INK2, fontsize=9.5)


def fig1_waterfall():
    w = pd.read_csv(os.path.join(RES, "weekly_volume_by_role.csv"), index_col=0, parse_dates=True)[ROLES]
    fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
    im = ax.imshow(w.T.values + 1, aspect="auto", cmap="Blues", norm=LogNorm(vmin=10, vmax=w.values.max()),
                   interpolation="nearest")
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_yticks(range(len(ROLES))); ax.set_yticklabels(ROLES)
    ticks = [i for i, d in enumerate(w.index) if d.month in (1, 7) and d.day <= 7]
    ax.set_xticks(ticks); ax.set_xticklabels([w.index[i].strftime("%b %Y") for i in ticks])
    e = np.argmin(np.abs(w.index - ELECTION))
    ax.axvline(e, color=INK, lw=1, ls="--")
    ax.text(e + 1, 4, "US election, 8 Nov 2016", fontsize=8.5, va="center")
    cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9); cb.set_label("English tweets / week"); cb.outline.set_visible(False)
    title(fig, "The operation as a waterfall: weekly output of each IRA unit",
          "1.79 M English tweets from 1,118 accounts, Jun 2015 – Dec 2017 (roles coded by Linvill & Warren, Clemson)")
    save(fig, "fig1_operation_waterfall.png")


def fig2_clock():
    d = J("rhythm.json")["describe"]
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.9), sharey=False, constrained_layout=True)
    for ax, role in zip(axes, ROLES):
        p = 100 * np.array(d[role]["hour_profile_msk"])
        ax.axvspan(9.5, 18.5, color=GRID, alpha=0.7, lw=0)
        ax.bar(range(24), p, width=0.8, color=S1)
        ax.set_title(role, fontsize=10)
        ax.set_xticks([0, 6, 12, 18]); ax.set_xticklabels(["00\n16", "06\n22", "12\n04", "18\n10"], fontsize=8)
        ax.set_xlim(-0.7, 23.7); ax.grid(axis="x", visible=False)
        off = d[role]["share_msk_office_10_19"]; wk = d[role]["share_weekend_msk"]
        ax.set_xlabel(f"{off:.0%} in Moscow office hours\n{wk:.0%} on weekends", fontsize=8.5, color=INK)
    axes[0].set_ylabel("% of tweets")
    title(fig, "Two clocks: when each unit tweets",
          "Hour labels: Moscow time (top) / US Eastern standard time (bottom).  Shaded: Moscow 10:00–19:00")
    save(fig, "fig2_daily_clock.png")


def fig3_dst():
    r = J("rhythm.json")["dst"]
    rows = ["ALL", "NewsFeed", "HashtagGamer", "LeftTroll", "RightTroll"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for key, c, dy, lab in (("real", S1, 0.13, "at US clock changes"), ("placebo", S2, -0.13, "placebo dates (no change)")):
        for yi, role in zip(y, rows):
            v = r[key][role]
            a1.errorbar(v["median_shift_min"], yi + dy, xerr=[[v["median_shift_min"] - v["median_shift_ci95"][0]],
                        [v["median_shift_ci95"][1] - v["median_shift_min"]]], fmt="o", color=c, ms=6, capsize=0, lw=2,
                        label=lab if role == "ALL" else None)
            s = 100 * v["share_us_clock"]; ci = [100 * x for x in v["share_us_clock_ci95"]]
            a2.errorbar(s, yi + dy, xerr=[[s - ci[0]], [ci[1] - s]], fmt="o", color=c, ms=6, capsize=0, lw=2,
                        label=lab if role == "ALL" else None)
    for ax in (a1, a2):
        ax.set_yticks(y); ax.set_yticklabels([f"{k}  (n={r['real'][k]['accounts']})" for k in rows]); ax.grid(axis="y", visible=False)
    a1.axvline(0, color=INK2, lw=1, ls="--"); a1.axvline(60, color=INK2, lw=1, ls="--")
    a1.text(1, y[0] + 0.45, "no DST clock\n(e.g. Moscow)", fontsize=8, color=INK2); a1.text(59, y[0] + 0.45, "US clock", fontsize=8, color=INK2, ha="right")
    a1.set_xlim(-30, 75); a1.set_ylim(-0.6, len(rows) - 0.1)
    a1.set_xlabel("median shift of each account's daily rhythm (minutes, aligned so US clock = +60)")
    a1.set_title("How far does the daily rhythm move?")
    a2.set_xlabel("% of accounts whose rhythm fits a US clock better than a Moscow clock")
    a2.set_title("Which clock does each account follow?"); a2.set_xlim(0, 100); a2.set_ylim(-0.6, len(rows) - 0.1)
    a2.legend(loc="lower right", fontsize=8)
    title(fig, "Natural experiment 1: US daylight-saving changes (Russia has none since 2014)",
          "Each account compared with itself, 4 weeks before vs after; 95% CIs from an account-level bootstrap")
    save(fig, "fig3_dst_experiment.png")


def fig4_holidays():
    h = J("rhythm.json")["holidays"]
    rows = ["ALL"] + ROLES
    fig, ax = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for key, c, dy, lab in (("russian", S1, 0.17, "Russian holidays (US workdays)"),
                            ("us", S2, -0.17, "US holidays (Russian workdays)")):
        for yi, role in zip(y, rows):
            v = h[role][key]
            ax.plot(v["null_95_pct"], [yi + dy] * 2, color=NEUTRAL, lw=6, solid_capstyle="round", alpha=0.7)
            ax.plot(v["change_pct"], yi + dy, "o", color=c, ms=7, label=lab if role == "ALL" else None)
            if key == "russian" and v["p_one_sided_drop"] < 0.05:
                ax.text(v["change_pct"] - 3, yi + dy, f"p={v['p_one_sided_drop']:.2f}", ha="right", va="center", fontsize=8)
    ax.plot([], [], color=NEUTRAL, lw=6, label="95% range on random ordinary weekdays")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(rows); ax.grid(axis="y", visible=False)
    ax.set_xlabel("change in active accounts vs the same weekday 1–2 weeks either side (%)")
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.0, 1.0))
    title(fig, "Natural experiment 2: who takes the day off?")
    save(fig, "fig4_holiday_experiment.png")


def fig5_roles():
    r = J("roles.json")
    fam = list(r["cv"].keys())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    v = [r["cv"][f]["macro_f1"] for f in fam]; e = [r["cv"][f]["sd"] for f in fam]
    a1.barh(range(len(fam))[::-1], v, xerr=e, color=S1, height=0.6, error_kw=dict(ecolor=INK2, lw=1))
    for i, x in zip(range(len(fam))[::-1], v):
        a1.text(x + 0.012, i, f"{x:.2f}", va="center", fontsize=8.5)
    a1.set_yticks(range(len(fam))[::-1]); a1.set_yticklabels(fam); a1.set_xlim(0, 1.08); a1.grid(axis="y", visible=False)
    a1.axvline(0.2, color=INK2, lw=1, ls=":"); a1.text(0.21, len(fam) - 0.6, "chance", fontsize=8, color=INK2)
    a1.set_xlabel("macro-F1, 5 roles (5-fold × 2, split by account)")
    a1.set_title(f"Recovering each account's role ({r['accounts']} accounts)")
    d = r["drift"]["families"]
    yy = np.arange(len(fam))[::-1]
    for yi, f in zip(yy, fam):
        a2.plot([d[f]["pre_to_post"], d[f]["pre_to_pre"]], [yi, yi], color=NEUTRAL, lw=3, zorder=1)
    a2.scatter([d[f]["pre_to_pre"] for f in fam], yy, color=S1, s=45, zorder=2, label="tested before the election")
    a2.scatter([d[f]["pre_to_post"] for f in fam], yy, color=S2, s=45, zorder=2, label="tested after the election")
    for yi, f in zip(yy, fam):
        a2.text(d[f]["pre_to_post"] - 0.015, yi, f"−{d[f]['drop']:.2f}", ha="right", va="center", fontsize=8.5)
    a2.set_yticks(yy); a2.set_yticklabels(fam); a2.set_xlim(0.2, 1.05); a2.grid(axis="y", visible=False)
    a2.set_xlabel(f"macro-F1, {len(r['drift']['roles'])} roles, {r['drift']['accounts']} accounts active in both periods")
    a2.set_title("Trained before the election: does it still work after?"); a2.legend(loc="lower left", fontsize=8)
    title(fig, "The persona lives in the language; the schedule does not")
    save(fig, "fig5_role_recovery.png")


def fig6_network():
    r = J("network.json")
    M = pd.DataFrame(r["cross_role_share"]).T.loc[ROLES, ROLES]
    lags = np.load(os.path.join(RES, "co_post_lags.npy"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), constrained_layout=True,
                                 gridspec_kw=dict(width_ratios=[1, 1.15]))
    im = a1.imshow(100 * M.values, cmap="Blues", vmin=0, vmax=100)
    a1.grid(False); [s.set_visible(False) for s in a1.spines.values()]
    a1.set_xticks(range(5)); a1.set_xticklabels(ROLES, rotation=30, ha="right"); a1.set_yticks(range(5)); a1.set_yticklabels(ROLES)
    for i in range(5):
        for j in range(5):
            val = 100 * M.values[i, j]
            a1.text(j, i, f"{val:.0f}" if val >= 0.5 else "·", ha="center", va="center", fontsize=8.5,
                    color=SURF if val > 55 else INK)
    a1.set_title("% of each unit's coordination links, by partner unit")
    a1.set_xlabel("partner unit")
    b = np.logspace(-2, np.log10(3600), 46)
    a2.hist(np.clip(lags, 0.01, 3600), bins=b, color=S1, edgecolor=SURF, lw=0.6)
    a2.set_xscale("log"); a2.set_xlabel("seconds between two accounts posting the identical text")
    a2.set_ylabel("account pairs × texts")
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    a2.set_title(f"{(lags <= 1).mean():.0%} of copies land within 1 s, {(lags <= 60).mean():.0%} within 1 min")
    a2.set_xticks([0.01, 0.1, 1, 10, 60, 600, 3600])
    a2.set_xticklabels(["≤10 ms", "0.1 s", "1 s", "10 s", "1 min", "10 min", "1 h"])
    title(fig, "Coordination: identical posts relayed within minutes, inside siloed units",
          f"{r['network_nodes']} accounts linked by ≥{r['min_co']} identical posts within 1 h; "
          f"{r['within_role_link_share']:.0%} of links stay within a unit (random-label null: {r['within_role_null_mean']:.0%})")
    save(fig, "fig6_coordination.png")


if __name__ == "__main__":
    for f in (fig1_waterfall, fig2_clock, fig3_dst, fig4_holidays, fig5_roles, fig6_network):
        f(); print("drew", f.__name__)
