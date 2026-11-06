"""Desk-wide kill switch: halt new entries, then close every position on every venue.

Dry run by default: it shows what each venue would cancel and close. With --live and the
typed phrase it engages the shared KILL_SWITCH file (no executor opens new positions
while it exists), flattens every HL wallet in accounts.json, then closes the IB account.
A venue that cannot be reached is reported and the others still run. --clear-kill-switch
on any executor allows entries again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hl_trade  # noqa: E402
import ib_trade  # noqa: E402
from desk.accounts import accounts_for, load_accounts  # noqa: E402
from desk.venues import ib as ib_venue  # noqa: E402

PHRASE = "SHUT IT ALL DOWN"


def _hl_wallets(accounts) -> list[str]:
    return [account.id for account in accounts_for("hl", accounts)]


def show_plan(accounts) -> list[str]:
    """Print every venue's kill plan; return the venues that could not be read."""
    problems = []
    wallets = _hl_wallets(accounts)
    print("===== HL")
    if wallets:
        try:
            hl_trade.print_kill_switch_plan(wallets, hl_trade.get_dexes([]))
        except Exception as exc:  # one venue's failure must not hide the others
            problems.append(f"HL: {type(exc).__name__}: {exc}")
            print(f"UNREACHABLE: {exc}")
    else:
        print("No HL wallets configured.")
    for account in accounts_for("ib", accounts):
        print(f"===== {account.name}")
        try:
            ib = ib_venue.connect(account, "risk", readonly=True)
            try:
                ib_trade.print_kill_switch_plan(ib, ib_venue.account_id(ib, account))
            finally:
                ib.disconnect()
        except Exception as exc:
            problems.append(f"{account.name}: {exc}")
            print(f"UNREACHABLE: {exc}")
    return problems


def run(accounts) -> bool:
    """Engage the halt, then close every venue; return whether everything is confirmed flat."""
    ib_trade.KILL_SWITCH_FILE.write_text("Engaged by the desk-wide kill switch\n")
    print(f"New entries are halted on every venue ({ib_trade.KILL_SWITCH_FILE}).")
    results = {}
    wallets = _hl_wallets(accounts)
    if wallets:
        print("===== HL")
        try:
            results["HL"] = hl_trade.run_kill_switch(wallets, hl_trade.get_dexes([]))
        except Exception as exc:
            print(f"HL kill switch failed: {type(exc).__name__}: {exc}")
            results["HL"] = False
    for account in accounts_for("ib", accounts):
        print(f"===== {account.name}")
        try:
            ib = ib_venue.connect(account, "execution", readonly=False)
            try:
                results[account.name] = ib_trade.run_kill_switch(ib, ib_venue.account_id(ib, account))
            finally:
                ib.disconnect()
        except Exception as exc:
            print(f"{account.name} kill switch failed: {exc}")
            results[account.name] = False
    print("===== SUMMARY")
    for venue, flat in results.items():
        print(f"{venue}: {'flat' if flat else 'NOT confirmed flat: see above'}")
    return all(results.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help=f"Execute after typing {PHRASE}.")
    args = parser.parse_args(argv)
    accounts = load_accounts()
    if not accounts:
        print("No accounts configured in accounts.json.")
        return 2
    problems = show_plan(accounts)
    if not args.live:
        print("Dry run only. Nothing was cancelled or closed.")
        return 0
    if problems:
        print("Some venues could not be read; the kill switch still runs on the others.")
    if input(f"Type {PHRASE} to halt entries and close every position on every venue: ").strip() != PHRASE:
        print("Confirmation did not match; nothing was sent.")
        return 0
    return 0 if run(accounts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
