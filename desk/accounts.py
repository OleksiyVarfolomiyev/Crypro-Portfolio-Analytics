"""Desk account configuration: every venue account the desk reports on and risks together.

Accounts are listed in ``accounts.json`` at the repository root (kept out of git because
it names brokerage account numbers). ``accounts.example.json`` shows the format::

    [
      {"venue": "hl", "id": "0x...", "label": "HL 07f2"},
      {"venue": "ib", "id": "U1234567", "label": "IB", "port": 7496}
    ]

Each HL wallet is also read as an on-chain wallet on other EVM chains unless its
entry sets ``"onchain": false``. An IB ``id`` may be left empty to use the first account TWS manages. The file holds
public identifiers only; signing keys stay in the OS keychain.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACCOUNTS_FILE = ROOT / "accounts.json"
VENUES = ("hl", "ib")
# Older accounts files may use the long venue names.
VENUE_ALIASES = {"hyperliquid": "hl", "ibkr": "ib", "interactive brokers": "ib"}
IB_DEFAULT_HOST = "127.0.0.1"
# TWS live trading listens on 7496 by default; paper trading on 7497.
IB_DEFAULT_PORT = 7496
# Each concurrent API connection to TWS needs its own client ID.
IB_CLIENT_IDS = {"execution": 41, "overview": 42, "risk": 43, "analytics": 44}


@dataclass(frozen=True)
class Account:
    """One venue account: a HL wallet or an IB account."""

    venue: str
    id: str
    label: str = ""
    host: str = IB_DEFAULT_HOST
    port: int = IB_DEFAULT_PORT
    # HL wallets: also report the same address's holdings and activity on other chains.
    onchain: bool = True

    @property
    def name(self) -> str:
        """Short display name: the label, or a venue-specific abbreviation of the ID."""
        if self.label:
            return self.label
        if self.venue == "hl":
            return f"HL {self.id[-4:].lower()}"
        return f"IB {self.id}" if self.id else "IB"


def parse_accounts(items: list[dict]) -> list[Account]:
    """Validate raw account entries and return them as ``Account`` objects."""
    if not isinstance(items, list):
        raise ValueError("The accounts file must hold a JSON list of accounts.")
    accounts: list[Account] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Account entry {index} must be a JSON object.")
        venue = str(item.get("venue", "")).strip().lower()
        venue = VENUE_ALIASES.get(venue, venue)
        if venue not in VENUES:
            raise ValueError(f"Account entry {index} has unknown venue {venue!r}; use one of {VENUES}.")
        account_id = str(item.get("id", "")).strip()
        if venue == "hl" and not re.fullmatch(r"0x[0-9a-fA-F]{40}", account_id):
            raise ValueError(f"Account entry {index}: a HL id must be a 0x wallet address.")
        if venue == "ib" and account_id and not re.fullmatch(r"[A-Z0-9]{4,20}", account_id):
            raise ValueError(f"Account entry {index}: {account_id!r} is not an IB account number.")
        key = (venue, account_id.lower())
        if key in seen:
            raise ValueError(f"Account entry {index} repeats {venue} account {account_id!r}.")
        seen.add(key)
        port = int(item.get("port", IB_DEFAULT_PORT))
        if not 0 < port < 65536:
            raise ValueError(f"Account entry {index} has an invalid port {port}.")
        accounts.append(Account(
            venue=venue, id=account_id, label=str(item.get("label", "")).strip(),
            host=str(item.get("host", IB_DEFAULT_HOST)), port=port,
            onchain=bool(item.get("onchain", True)) and venue == "hl",
        ))
    return accounts


def load_accounts(path: Path | str | None = None) -> list[Account]:
    """Read the desk's accounts; return an empty list when no accounts file exists."""
    # DESK_ACCOUNTS_FILE lets tests and one-off runs point at a different file.
    path = Path(path) if path is not None else Path(os.environ.get("DESK_ACCOUNTS_FILE", ACCOUNTS_FILE))
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return parse_accounts(json.load(file))


def accounts_for(venue: str, accounts: list[Account] | None = None) -> list[Account]:
    """Return the configured accounts on one venue."""
    accounts = load_accounts() if accounts is None else accounts
    return [account for account in accounts if account.venue == venue]
