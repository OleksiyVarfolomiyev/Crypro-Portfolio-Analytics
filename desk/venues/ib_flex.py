"""IB account history through IB's Flex Web Service: daily NAV and deposits, for time-weighted returns.

TWS's API has no account-value history; Flex does. One-time setup in IB's Client Portal:

1. Performance & Reports > Flex Queries > Activity Flex Query > Create.
   Sections: "Net Asset Value (NAV) in Base" and "Cash Transactions" (tick "Deposits &
   Withdrawals"). Delivery: format XML, period "Last 365 Calendar Days", and under
   "Breakout by Day" choose Yes. Save, and note the Query ID.
2. Performance & Reports > Flex Queries > Flex Web Service Configuration: enable it and
   generate a token (it is read-only: reports only, no trading).
3. Store the token: ``python -m desk.venues.ib_flex --store-token`` (Keychain), or set
   ``IB_FLEX_TOKEN``. Put the Query ID in accounts.json as ``"flex_query_id"`` on the IB entry.

Each refresh merges the statement into ``data/ib_nav_history_<account>.csv`` so history
keeps growing beyond IB's one-year window.
"""

from __future__ import annotations

import getpass
import os
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pandas as pd
import requests

from desk.accounts import Account

SEND_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
GET_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/GetStatement"
KEYCHAIN_SERVICE = "IB Flex Web Service"
KEYCHAIN_USER = "default"
USER_AGENT = "trading-desk/1.0 (read-only portfolio report)"
# IB answers 1019 while the statement is still being generated.
IN_PROGRESS_CODES = {"1019"}
POLL_ATTEMPTS = 12
POLL_SECONDS = 5
NAV_COLUMNS = ["date", "nav", "flows"]


class FlexNotConfigured(RuntimeError):
    """No Flex token or query ID is configured."""


def flex_token() -> str:
    if os.environ.get("IB_FLEX_TOKEN"):
        return os.environ["IB_FLEX_TOKEN"].strip()
    try:
        import keyring

        return keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_USER) or ""
    except Exception:  # no keyring or a locked keychain means no token, not a crash
        return ""


def store_flex_token() -> None:
    import keyring

    token = getpass.getpass("IB Flex Web Service token (input hidden): ").strip()
    if not token.isdigit():
        raise ValueError("A Flex token is a long string of digits.")
    keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_USER, token)
    print("Saved in the Keychain.")


def _get(url: str, params: dict, token: str) -> str:
    """GET that never lets the token appear in an error message."""
    try:
        response = requests.get(url, params=params, timeout=60, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"IB Flex request failed ({type(exc).__name__})") from None
    return response.text.replace(token, "<token>")


def fetch_statement(token: str, query_id: str, sleep=time.sleep) -> str:
    """Request a Flex statement and wait for IB to generate it; returns the statement XML."""
    reply = ElementTree.fromstring(_get(SEND_URL, {"t": token, "q": query_id, "v": 3}, token))
    if reply.findtext("Status") != "Success":
        raise RuntimeError(f"IB Flex refused the request: {reply.findtext('ErrorMessage') or 'unknown error'} "
                           f"(code {reply.findtext('ErrorCode')}).")
    reference, url = reply.findtext("ReferenceCode"), reply.findtext("Url") or GET_URL
    for _ in range(POLL_ATTEMPTS):
        text = _get(url, {"t": token, "q": reference, "v": 3}, token)
        root = ElementTree.fromstring(text)
        if root.tag != "FlexStatementResponse":
            return text
        if root.findtext("ErrorCode") not in IN_PROGRESS_CODES:
            raise RuntimeError(f"IB Flex statement failed: {root.findtext('ErrorMessage')} "
                               f"(code {root.findtext('ErrorCode')}).")
        sleep(POLL_SECONDS)
    raise RuntimeError("IB Flex statement was still being generated after a minute; try again shortly.")


def _date(text: str) -> pd.Timestamp:
    text = (text or "").split(";")[0].split(" ")[0].replace("-", "")
    return pd.Timestamp(text[:8], tz="UTC")


def parse_nav_history(xml_text: str) -> tuple[str, pd.DataFrame]:
    """Account ID and a daily frame of NAV (base currency) and net deposits/withdrawals."""
    root = ElementTree.fromstring(xml_text)
    statement = root.find(".//FlexStatement")
    account = statement.get("accountId", "") if statement is not None else ""
    nav = {}
    for row in root.iter("EquitySummaryByReportDateInBase"):
        if row.get("reportDate") and row.get("total") not in (None, ""):
            nav[_date(row.get("reportDate"))] = float(row.get("total"))
    flows = {}
    for row in root.iter("CashTransaction"):
        if "deposit" not in (row.get("type") or "").lower() and "withdraw" not in (row.get("type") or "").lower():
            continue
        when = row.get("reportDate") or row.get("dateTime") or row.get("settleDate")
        amount = float(row.get("amount") or 0) * float(row.get("fxRateToBase") or 1)
        flows[_date(when)] = flows.get(_date(when), 0.0) + amount
    frame = pd.DataFrame({"nav": pd.Series(nav, dtype=float)}).sort_index()
    frame["flows"] = pd.Series(flows, dtype=float).reindex(frame.index).fillna(0.0)
    frame.index.name = "date"
    return account, frame.reset_index()


def time_weighted_returns(history: pd.DataFrame) -> pd.Series:
    """Daily returns net of cash flows: (NAV - that day's deposits) / previous NAV - 1."""
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.drop_duplicates("date", keep="last").set_index("date").sort_index()
    previous = frame["nav"].shift(1)
    returns = (frame["nav"] - frame["flows"]) / previous.where(previous > 0) - 1
    return returns.dropna().rename("IB book")


def update_nav_history(account: Account, data_dir: Path | str) -> Path:
    """Fetch the configured Flex statement and merge it into ib_nav_history_<account>.csv."""
    token = flex_token()
    query_id = getattr(account, "flex_query_id", "")
    if not token or not query_id:
        raise FlexNotConfigured("no Flex token stored" if not token else "no flex_query_id in accounts.json")
    statement_account, fresh = parse_nav_history(fetch_statement(token, query_id))
    name = statement_account or account.id or "ib"
    path = Path(data_dir) / f"ib_nav_history_{name}.csv"
    if path.exists():
        saved = pd.read_csv(path)
        saved["date"] = pd.to_datetime(saved["date"], utc=True)
        fresh = pd.concat([saved, fresh], ignore_index=True).drop_duplicates("date", keep="last")
    fresh.sort_values("date").to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IB Flex Web Service history for the trading desk.")
    parser.add_argument("--store-token", action="store_true", help="Save a Flex Web Service token in the Keychain.")
    if parser.parse_args().store_token:
        store_flex_token()
