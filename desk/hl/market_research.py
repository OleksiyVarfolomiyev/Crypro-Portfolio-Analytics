"""Rank HL markets using the same definitions as the app's market selector.

The views mirror the Crypto, Tradfi, HIP-3, and Trending tabs. Formulas and filters
follow the app's public code as checked on 2026-11-03; HL can change them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from desk.hl.trade_analytics import post_info

TOP_N = 20
TRADFI_CATEGORIES = {"stocks", "indices", "commodities", "fx", "preipo", "rates"}
TRENDING_MIN_VOLUME = 30_000_000.0
CHANGE = "24h Change (%)"
FUNDING = "8h Funding (%)"
VOLUME = "24h Volume ($)"
OPEN_INTEREST = "OI ($)"
TABLE_COLUMNS = ["Ticker", CHANGE, FUNDING, VOLUME, OPEN_INTEREST]
VIEW_LABELS = {"crypto": "Crypto", "tradfi": "Tradfi", "hip3": "HIP-3", "trending": "Trending"}


def asset_class(dex: str | None, category: str | None) -> str:
    """Name a market's single class: Crypto, Tradfi, or HIP-3 for other HIP-3 markets."""
    if not dex:
        return "Crypto"
    return "Tradfi" if str(category or "").lower() in TRADFI_CATEGORIES else "HIP-3"


def _number(value: object) -> float:
    """Convert an API value to a float, using ``NaN`` for missing or invalid values."""
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else float("nan")


def _day_change_pct(mid_price: float, previous_price: float) -> float:
    """Return mid versus prior-day price, or ``NaN`` for markets without both prices."""
    if not mid_price > 0 or not previous_price > 0:
        return float("nan")
    return (mid_price / previous_price - 1) * 100


def fetch_perp_markets() -> pd.DataFrame:
    """Fetch active perps from the main DEX and every HIP-3 DEX."""
    categories = {
        coin: str(category).lower() for coin, category in post_info({"type": "perpCategories"})
    }
    dexes = [""] + [dex["name"] for dex in post_info({"type": "perpDexs"}) if dex and dex.get("name")]
    rows = []
    for dex in dexes:
        meta, contexts = post_info({"type": "metaAndAssetCtxs", "dex": dex})
        for asset, context in zip(meta["universe"], contexts):
            if asset.get("isDelisted"):
                continue
            rows.append({
                "Ticker": asset["name"],
                "Dex": dex,
                "Category": categories.get(asset["name"], ""),
                CHANGE: _day_change_pct(_number(context.get("midPx")), _number(context.get("prevDayPx"))),
                # The API rate is hourly and fractional; the app shows it per 8h in percent.
                FUNDING: _number(context.get("funding")) * 8 * 100,
                VOLUME: _number(context.get("dayNtlVlm")),
                # The API reports open interest in coins.
                OPEN_INTEREST: _number(context.get("openInterest")) * _number(context.get("markPx")),
            })
    return pd.DataFrame(rows)


def fetch_canonical_spot_markets() -> pd.DataFrame:
    """Fetch canonical spot pairs, the only spot markets eligible for Trending."""
    meta, contexts = post_info({"type": "spotMetaAndAssetCtxs"})
    contexts_by_coin = {context.get("coin"): context for context in contexts}
    rows = []
    for pair in meta["universe"]:
        context = contexts_by_coin.get(pair["name"])
        if not pair.get("isCanonical") or context is None:
            continue
        rows.append({
            "Ticker": pair["name"],
            "Dex": None,
            "Category": "spot",
            CHANGE: _day_change_pct(_number(context.get("midPx")), _number(context.get("prevDayPx"))),
            FUNDING: np.nan,
            VOLUME: _number(context.get("dayNtlVlm")),
            OPEN_INTEREST: np.nan,
        })
    return pd.DataFrame(rows)


def select_view(markets: pd.DataFrame, view: str) -> pd.DataFrame:
    """Return one view's market universe; Trending returns its volume-eligible pool."""
    if view == "crypto":
        return markets.loc[markets["Dex"] == ""]
    if view == "hip3":
        return markets.loc[markets["Dex"].notna() & (markets["Dex"] != "")]
    if view == "tradfi":
        return markets.loc[
            markets["Dex"].notna() & (markets["Dex"] != "") & markets["Category"].isin(TRADFI_CATEGORIES)
        ]
    if view == "trending":
        return markets.loc[markets[VOLUME] > TRENDING_MIN_VOLUME]
    raise ValueError(f"Unknown market view: {view!r}")


def trending_markets(markets: pd.DataFrame) -> pd.DataFrame:
    """Return what the Trending tab lists: the biggest loser and three biggest gainers."""
    ranked = select_view(markets, "trending").dropna(subset=[CHANGE]).sort_values(CHANGE)
    picked = pd.concat([ranked.iloc[:1], ranked.iloc[-3:]])
    return picked[~picked.index.duplicated()].sort_values(CHANGE, ascending=False)


def rank_markets(markets: pd.DataFrame, metric: str, top: bool, count: int = TOP_N) -> pd.DataFrame:
    """Return the highest or lowest markets by ``metric``, skipping missing values."""
    return markets.dropna(subset=[metric]).sort_values(metric, ascending=not top).head(count)


HIGH_FUNDING_8H = 0.05  # % per 8h; about 55% a year, which makes carrying the paying side expensive


def market_highlights(universe: pd.DataFrame, view: str) -> tuple[list, list]:
    """What stands out in one market view: biggest moves, funding extremes, and where the volume is."""
    well, attention = [], []
    if universe.empty:
        return well, [("Markets", "No markets returned for this view.")]

    def names(frame: pd.DataFrame, column: str, fmt: str) -> str:
        return ", ".join(f"{row['Ticker']} {format(row[column], fmt)}" for _, row in frame.iterrows())

    changes = universe.dropna(subset=[CHANGE])
    if len(changes):
        up = (changes[CHANGE] > 0).mean()
        well.append(("Breadth", f"{up:.0%} of {len(changes):,} markets are up over 24h; median change "
                                f"{changes[CHANGE].median():+.2f}%."))
        well.append(("Biggest gainers", names(changes.nlargest(3, CHANGE), CHANGE, "+.2f") + " (%)."))
        attention.append(("Biggest losers", names(changes.nsmallest(3, CHANGE), CHANGE, "+.2f") + " (%)."))
    funding = universe.dropna(subset=[FUNDING])
    if len(funding):
        hot = funding[funding[FUNDING].abs() >= HIGH_FUNDING_8H].sort_values(FUNDING, key=abs, ascending=False)
        if len(hot):
            attention.append((f"Funding beyond ±{HIGH_FUNDING_8H}%/8h",
                              f"{len(hot)} markets; " + names(hot.head(5), FUNDING, "+.4f")
                              + " (%; positive means longs pay)."))
        top, bottom = funding.loc[funding[FUNDING].idxmax()], funding.loc[funding[FUNDING].idxmin()]
        well.append(("Funding range", f"highest {top['Ticker']} {top[FUNDING]:+.4f}% "
                                      f"(~{top[FUNDING] * 3 * 365:+.0f}%/yr), lowest {bottom['Ticker']} "
                                      f"{bottom[FUNDING]:+.4f}% (~{bottom[FUNDING] * 3 * 365:+.0f}%/yr)."))
    volume = universe.dropna(subset=[VOLUME])
    if len(volume) and volume[VOLUME].sum() > 0:
        leaders = volume.nlargest(3, VOLUME)
        share = leaders[VOLUME].sum() / volume[VOLUME].sum()
        well.append(("Most traded", ", ".join(f"{row['Ticker']} ${row[VOLUME] / 1e6:,.0f}M" for _, row in leaders.iterrows())
                     + f" ({share:.0%} of the view's 24h volume)."))
    if view == "trending":
        well.append(("Trending pool", f"{len(universe):,} markets above ${TRENDING_MIN_VOLUME / 1e6:,.0f}M 24h volume."))
    return well, attention


def show_market_research(view: str) -> None:
    """Display the top and bottom tables for one market view."""
    from desk.hl.account_analytics import display_interactive_table

    markets = fetch_perp_markets()
    if view == "trending":
        markets = pd.concat([markets, fetch_canonical_spot_markets()], ignore_index=True)
    universe = select_view(markets, view)
    label = VIEW_LABELS[view]

    def signed_style(row_index, column, value, row):
        if column in {CHANGE, FUNDING} and pd.notna(value):
            return "color:#188038;" if value > 0 else "color:#c62828;" if value < 0 else ""
        return ""

    def show(title: str, frame: pd.DataFrame) -> None:
        print(f"=== {title} ===")
        display_interactive_table(
            frame,
            columns=TABLE_COLUMNS,
            cell_style=signed_style,
            formatters={
                CHANGE: lambda v: f"{v:.2f}%" if pd.notna(v) else "n/a",
                FUNDING: lambda v: f"{v:.4f}%" if pd.notna(v) else "n/a",
                VOLUME: lambda v: f"${v:,.0f}" if pd.notna(v) else "n/a",
                OPEN_INTEREST: lambda v: f"${v:,.0f}" if pd.notna(v) else "n/a",
            },
        )

    from desk.notebook_highlights import show as show_highlights

    print("=== HIGHLIGHTS ===")
    show_highlights(*market_highlights(universe, view))
    print(f"{label}: {len(universe):,} markets. Change is mid vs prior-day price; funding is the "
          "current hourly rate x 8. Positive funding means longs pay shorts.")
    if view == "trending":
        print(f"Trending lists the biggest loser and three biggest gainers with over "
              f"${TRENDING_MIN_VOLUME:,.0f} 24h volume. The ranked tables below use that same pool.")
        show("HL TRENDING", trending_markets(markets))
    for title, metric, top in (
        (f"TOP {TOP_N} BY 24H CHANGE", CHANGE, True),
        (f"BOTTOM {TOP_N} BY 24H CHANGE", CHANGE, False),
        (f"TOP {TOP_N} BY 8H FUNDING", FUNDING, True),
        (f"BOTTOM {TOP_N} BY 8H FUNDING", FUNDING, False),
    ):
        show(title, rank_markets(universe, metric, top))
