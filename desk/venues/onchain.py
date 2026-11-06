"""On-chain activity of the desk's HL wallets on EVM chains, outside HL.

Each HL wallet address is an ordinary EVM wallet that can also hold tokens on
Ethereum, Arbitrum (HL's bridge chain), Base and other chains. This module reads
those balances and transfers:

* Blockscout's public API (no key) for Ethereum, Optimism, Unichain, Scroll, zkSync, Ink.
* Etherscan's multichain API for Arbitrum and Polygon (free key; BNB Chain and Avalanche
  need a paid plan). Keep the key in the macOS Keychain (``--store-key``) or the
  ``ETHERSCAN_API_KEY`` environment variable.
* Base: Alchemy (free key; ``--store-alchemy-key`` or ``ALCHEMY_API_KEY``) gives every
  token balance and the full transfer history. Without it, Blockscout's PRO API is used on a
  paid plan, and otherwise Base's public RPC reads native ETH and the main Base tokens only.
* DefiLlama's public price API values tokens. Tokens it cannot price, which includes the
  scam airdrops that flood most wallets, are counted but never valued or listed.

Read-only: nothing here signs or sends a transaction.
"""

from __future__ import annotations

import getpass
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from desk.accounts import Account
from desk.venues.base import VenueAccount, VenuePosition

VENUE = "onchain"
ETHERSCAN_URL = "https://api.etherscan.io/v2/api"
LLAMA_URL = "https://coins.llama.fi/prices/current/"
KEYCHAIN_SERVICE = "Etherscan API"
BLOCKSCOUT_KEYCHAIN_SERVICE = "Blockscout PRO API"
ALCHEMY_KEYCHAIN_SERVICE = "Alchemy API"
# Chains read through Alchemy when an Alchemy key is stored, by Alchemy network name.
ALCHEMY_NETWORKS = {"base": "base-mainnet"}
ALCHEMY_PAGE_LIMIT = 20
KEYCHAIN_USER = "default"
BLOCKSCOUT_PRO_HOST = "api.blockscout.com"
BLOCKSCOUT_PRO_SPACING = 0.22  # free tier: 5 requests per second
# Public RPCs used for keyless balance reads (no history: their log queries are blocked).
PUBLIC_RPC = {"base": "https://mainnet.base.org"}
# Main tokens read by the keyless fallback; symbol and decimals are read on-chain.
WELL_KNOWN_TOKENS = {
    "base": (
        "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",  # USDC
        "0xd9aaec86b65d86f6a7b5b1b0c42ffa531710b6ca",  # USDbC
        "0x4200000000000000000000000000000000000006",  # WETH
        "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf",  # cbBTC
        "0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22",  # cbETH
        "0x50c5725949a6f0c72e6c4a641f24049a917db0cb",  # DAI
        "0x940181a94a35a4569e4529a3cdfb74e38fd98631",  # AERO
        "0x60a3e35cc302bfa44cb288bc5a4f316fdb1adb42",  # EURC
        "0xc1cba3fcea344f92d9239c08c0568f6f2f0ee452",  # wstETH
        "0xfde4c96c8593536e31f229ea8f37b2ada2699bb2",  # USDT
        "0xacfe6019ed1a7dc6f7b508c02d1b04ec88cc21bf",  # VVV (Venice)
        "0x321b7ff75154472b18edb199033ff4d116f340ff",  # sVVV: VVV staked with Venice
    ),
}
# Staking receipts priced as the token they represent one-for-one: (chain, receipt) -> token.
PRICED_AS = {
    ("base", "0x321b7ff75154472b18edb199033ff4d116f340ff"): "0xacfe6019ed1a7dc6f7b508c02d1b04ec88cc21bf",
}
# Unclaimed staking rewards: (chain, staking contract) -> (view selector taking the wallet,
# reward token). 0x31d7a262 is pendingRewards(address) on Venice's VVV staking contract.
STAKING_REWARDS = {
    ("base", "0x321b7ff75154472b18edb199033ff4d116f340ff"): ("0x31d7a262", "0xacfe6019ed1a7dc6f7b508c02d1b04ec88cc21bf"),
}
# Readable names for staking receipts in position tables.
STAKED_LABELS = {("base", "0x321b7ff75154472b18edb199033ff4d116f340ff"): "VVV staked on Venice"}
USER_AGENT = "trading-desk/1.0 (read-only portfolio report)"
NATIVE = "0x0000000000000000000000000000000000000000"
# DefiLlama confidence below this is treated as unpriced (thin or manipulated markets).
MIN_PRICE_CONFIDENCE = 0.8
# A single holding above this is assumed to be a mispriced spam token, not wealth.
MAX_PLAUSIBLE_HOLDING_USD = 10_000_000
DUST_USD = 1.0
BLOCKSCOUT_MAX_PAGES = 20
ETHERSCAN_CALL_SPACING = 0.25  # free tier: 5 calls per second
REQUEST_ATTEMPTS = 3
# Known contracts whose transfers are HL deposits/withdrawals, by chain.
KNOWN_COUNTERPARTIES = {
    ("arbitrum", "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7"): "HL bridge",
}


@dataclass(frozen=True)
class Chain:
    name: str
    chain_id: int
    source: str  # "blockscout", "blockscout_pro" (needs a key) or "etherscan"
    llama: str  # DefiLlama chain key
    native_symbol: str
    native_price_key: str
    host: str = ""


CHAINS = {
    chain.name: chain for chain in (
        Chain("ethereum", 1, "blockscout", "ethereum", "ETH", "coingecko:ethereum", "eth.blockscout.com"),
        Chain("optimism", 10, "blockscout", "optimism", "ETH", "coingecko:ethereum", "optimism.blockscout.com"),
        Chain("unichain", 130, "blockscout", "unichain", "ETH", "coingecko:ethereum", "unichain.blockscout.com"),
        Chain("scroll", 534352, "blockscout", "scroll", "ETH", "coingecko:ethereum", "scroll.blockscout.com"),
        Chain("zksync", 324, "blockscout", "era", "ETH", "coingecko:ethereum", "zksync.blockscout.com"),
        Chain("ink", 57073, "blockscout", "ink", "ETH", "coingecko:ethereum", "explorer.inkonchain.com"),
        Chain("arbitrum", 42161, "etherscan", "arbitrum", "ETH", "coingecko:ethereum"),
        Chain("base", 8453, "blockscout_pro", "base", "ETH", "coingecko:ethereum", f"{BLOCKSCOUT_PRO_HOST}/8453"),
        Chain("polygon", 137, "etherscan", "polygon", "POL", "coingecko:polygon-ecosystem-token"),
        Chain("bsc", 56, "etherscan", "bsc", "BNB", "coingecko:binancecoin"),
        Chain("avalanche", 43114, "etherscan", "avax", "AVAX", "coingecko:avalanche-2"),
    )
}
NAME_CACHE = Path(__file__).resolve().parents[2] / "data" / "api_cache" / "contract_names.json"
ACTIVITY_COLUMNS = ["chain", "time", "hash", "log_index", "direction", "symbol", "token", "amount",
                    "price_usd", "value_usd", "counterparty", "label"]


@dataclass(frozen=True)
class Holding:
    chain: str
    token: str  # contract address, or NATIVE
    symbol: str
    amount: float


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, headers: dict | None = None) -> requests.Response:
    """GET with a few retries on network errors and rate limits; 404 is returned, not raised."""
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=30,
                                    headers={"User-Agent": USER_AGENT, **(headers or {})})
        except (requests.Timeout, requests.ConnectionError):
            if attempt == REQUEST_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 429 and attempt < REQUEST_ATTEMPTS - 1:
            time.sleep(2 ** (attempt + 1))
            continue
        if response.status_code == 402:
            raise PaidPlanRequired(_paid_plan_reason(response))
        if response.status_code != 404:
            response.raise_for_status()
        return response
    raise RuntimeError("unreachable")


def _stored_key(env: str, service: str) -> str:
    """An API key from the environment or the Keychain, or '' if none is set."""
    if os.environ.get(env):
        return os.environ[env].strip()
    try:
        import keyring
    except ImportError:
        return ""
    try:
        return keyring.get_password(service, KEYCHAIN_USER) or ""
    except Exception:  # a locked or unavailable keychain means no key, not a crash
        return ""


def etherscan_key() -> str:
    return _stored_key("ETHERSCAN_API_KEY", KEYCHAIN_SERVICE)


def blockscout_key() -> str:
    return _stored_key("BLOCKSCOUT_API_KEY", BLOCKSCOUT_KEYCHAIN_SERVICE)


def alchemy_key() -> str:
    return _stored_key("ALCHEMY_API_KEY", ALCHEMY_KEYCHAIN_SERVICE)


def _store_key(service: str, label: str) -> None:
    """Prompt for an API key with hidden input and save it in the Keychain."""
    import keyring

    key = getpass.getpass(f"{label} API key (input hidden): ").strip()
    if not key or any(character.isspace() for character in key):
        raise ValueError(f"That does not look like a {label} API key.")
    keyring.set_password(service, KEYCHAIN_USER, key)
    print("Saved in the Keychain.")


def store_etherscan_key() -> None:
    _store_key(KEYCHAIN_SERVICE, "Etherscan")


def store_blockscout_key() -> None:
    _store_key(BLOCKSCOUT_KEYCHAIN_SERVICE, "Blockscout PRO")


def store_alchemy_key() -> None:
    _store_key(ALCHEMY_KEYCHAIN_SERVICE, "Alchemy")


def _blockscout_headers(chain: Chain) -> dict:
    """Headers a request to this Blockscout host needs: the PRO API key, sent as a bearer
    token so it never appears in a URL or an error message."""
    if chain.source != "blockscout_pro":
        return {}
    key = blockscout_key()
    if not key:
        raise MissingKey("needs a Blockscout PRO API key "
                         "(python -m desk.venues.onchain --store-blockscout-key)")
    time.sleep(BLOCKSCOUT_PRO_SPACING)
    return {"Authorization": f"Bearer {key}"}


def _paid_plan_reason(response: requests.Response) -> str:
    try:
        detail = response.json().get("error", "")
    except ValueError:
        detail = ""
    return f"needs a paid plan ({detail})" if detail else "needs a paid API plan"


def redact(text: str) -> str:
    """Remove stored API keys from text that may be printed or saved."""
    for key in (etherscan_key(), blockscout_key(), alchemy_key()):
        if key:
            text = text.replace(key, "<key>")
    return text


class PaidPlanRequired(RuntimeError):
    """Etherscan's free tier does not serve this chain."""


class MissingKey(RuntimeError):
    """The chain's data source needs an API key that is not stored."""


def _chain_problem(chain: Chain, exc: Exception) -> str:
    if isinstance(exc, (PaidPlanRequired, MissingKey)):
        return redact(f"{chain.name} skipped: {exc}")
    return redact(f"{chain.name} unavailable: {type(exc).__name__}: {exc}")


def _etherscan(chain: Chain, key: str, **params) -> list | str:
    time.sleep(ETHERSCAN_CALL_SPACING)
    data = _get(ETHERSCAN_URL, {"chainid": chain.chain_id, "apikey": key, **params}).json()
    result = data.get("result")
    if data.get("status") == "1":
        return result
    if isinstance(result, list) or (data.get("message") or "").startswith("No transactions"):
        return []
    if "Free API access is not supported" in str(result):
        raise PaidPlanRequired(f"needs a paid Etherscan plan (the free key does not cover {chain.name})")
    raise RuntimeError(f"Etherscan {chain.name}: {data.get('message')}: {result}")


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def price_key(chain: Chain, token: str) -> str:
    if token == NATIVE:
        return chain.native_price_key
    token = token.lower().removeprefix("rewards:")
    token = PRICED_AS.get((chain.name, token), token)
    return f"{chain.llama}:{token}"


def fetch_prices(keys: set[str]) -> dict[str, float]:
    """USD prices from DefiLlama for ``chain:address`` keys; unknown or doubtful ones are omitted."""
    prices: dict[str, float] = {}
    keys = sorted(keys)
    for start in range(0, len(keys), 60):
        chunk = keys[start:start + 60]
        coins = _get(LLAMA_URL + ",".join(chunk)).json().get("coins", {})
        for key, coin in coins.items():
            if float(coin.get("confidence") or 0) >= MIN_PRICE_CONFIDENCE and float(coin.get("price") or 0) > 0:
                prices[key.lower()] = float(coin["price"])
    return prices


# ---------------------------------------------------------------------------
# Blockscout chains
# ---------------------------------------------------------------------------

def blockscout_holdings(chain: Chain, wallet: str) -> list[Holding]:
    base = f"https://{chain.host}/api/v2/addresses/{wallet}"
    info = _get(base, headers=_blockscout_headers(chain))
    if info.status_code == 404:
        return []
    holdings = []
    native = int(info.json().get("coin_balance") or 0) / 1e18
    if native:
        holdings.append(Holding(chain.name, NATIVE, chain.native_symbol, native))
    for item in _get(base + "/token-balances", headers=_blockscout_headers(chain)).json():
        token = item.get("token") or {}
        if token.get("type") != "ERC-20" or token.get("decimals") in (None, ""):
            continue
        amount = int(item.get("value") or 0) / 10 ** int(token["decimals"])
        if amount:
            holdings.append(Holding(chain.name, token["address_hash"].lower(), token.get("symbol") or "?", amount))
    return holdings


def _paged(chain: Chain, url: str, params: dict, since: pd.Timestamp, max_pages: int = BLOCKSCOUT_MAX_PAGES):
    """Yield Blockscout items newest first until they are older than ``since``."""
    page = dict(params)
    for _ in range(max_pages):
        response = _get(url, page, headers=_blockscout_headers(chain))
        if response.status_code == 404:
            return
        data = response.json()
        for item in data.get("items", []):
            if pd.Timestamp(item["timestamp"]) < since:
                return
            yield item
        if not data.get("next_page_params"):
            return
        page = {**params, **data["next_page_params"]}


def _direction(wallet: str, sender: str, receiver: str) -> str:
    wallet = wallet.lower()
    if sender.lower() == wallet and receiver.lower() == wallet:
        return "self"
    return "out" if sender.lower() == wallet else "in"


def blockscout_activity(chain: Chain, wallet: str, since: pd.Timestamp) -> list[dict]:
    base = f"https://{chain.host}/api/v2/addresses/{wallet}"
    rows = []
    for item in _paged(chain, base + "/token-transfers", {"type": "ERC-20"}, since):
        token, total = item.get("token") or {}, item.get("total") or {}
        if total.get("decimals") in (None, "") or total.get("value") in (None, ""):
            continue
        sender, receiver = item["from"]["hash"], item["to"]["hash"]
        rows.append({
            "chain": chain.name, "time": pd.Timestamp(item["timestamp"]), "hash": item["transaction_hash"],
            "log_index": int(item.get("log_index") or 0), "direction": _direction(wallet, sender, receiver),
            "symbol": token.get("symbol") or "?", "token": token.get("address_hash", "").lower(),
            "amount": int(total["value"]) / 10 ** int(total["decimals"]),
            "counterparty": receiver if sender.lower() == wallet.lower() else sender,
        })
    for item in _paged(chain, base + "/transactions", {}, since):
        value = int(item.get("value") or 0)
        if not value or item.get("status") != "ok":
            continue
        sender, receiver = item["from"]["hash"], (item.get("to") or {}).get("hash") or ""
        rows.append({
            "chain": chain.name, "time": pd.Timestamp(item["timestamp"]), "hash": item["hash"], "log_index": -1,
            "direction": _direction(wallet, sender, receiver), "symbol": chain.native_symbol, "token": NATIVE,
            "amount": value / 1e18, "counterparty": receiver if sender.lower() == wallet.lower() else sender,
        })
    return rows


# ---------------------------------------------------------------------------
# Keyless fallback: balances through a public RPC
# ---------------------------------------------------------------------------

def _decode_string(result: str) -> str:
    raw = bytes.fromhex(result[2:]) if result and result != "0x" else b""
    if len(raw) >= 96:  # ABI-encoded dynamic string
        length = int.from_bytes(raw[32:64], "big")
        return raw[64:64 + length].decode(errors="replace")
    return raw.rstrip(b"\0").decode(errors="replace")  # bytes32 symbols


RPC_BATCH_SIZE = 8
RPC_BATCH_SPACING = 0.3


def _rpc_batch(url: str, calls: list[dict]) -> dict[int, str]:
    """Run JSON-RPC calls in small batches; calls the node refuses are retried.

    Public nodes answer an over-limit batch with a single error, and may refuse single
    calls inside a batch ("request limit reached"), so both are retried with backoff.
    Raises if any call still has no result, so a balance is never silently dropped.
    """
    results: dict[int, str] = {}
    pending = list(calls)
    for attempt in range(REQUEST_ATTEMPTS + 2):
        failed = []
        for start in range(0, len(pending), RPC_BATCH_SIZE):
            chunk = pending[start:start + RPC_BATCH_SIZE]
            if start:
                time.sleep(RPC_BATCH_SPACING)
            response = requests.post(url, json=chunk, timeout=30, headers={"User-Agent": USER_AGENT})
            replies = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
            if response.status_code != 200 or not isinstance(replies, list):
                failed.extend(chunk)
                continue
            answered = {reply.get("id"): reply for reply in replies}
            for call in chunk:
                reply = answered.get(call["id"], {})
                if "result" in reply:
                    results[call["id"]] = reply["result"]
                else:
                    failed.append(call)
        if not failed:
            return results
        pending = failed
        time.sleep(2 ** attempt)
    raise RuntimeError(f"public RPC refused {len(pending)} call(s) after retries (rate limited)")


def staking_reward_holdings(chain: Chain, wallet: str) -> list[Holding]:
    """Unclaimed staking rewards, read from each staking contract on this chain."""
    contracts = [(contract, selector, token) for (name, contract), (selector, token) in STAKING_REWARDS.items()
                 if name == chain.name]
    if not contracts or chain.name not in PUBLIC_RPC:
        return []
    padded = wallet.lower()[2:].rjust(64, "0")
    calls = [{"jsonrpc": "2.0", "id": index, "method": "eth_call",
              "params": [{"to": contract, "data": selector + padded}, "latest"]}
             for index, (contract, selector, _) in enumerate(contracts)]
    results = _rpc_batch(PUBLIC_RPC[chain.name], calls)
    holdings = []
    for index, (_, _, token) in enumerate(contracts):
        raw = results.get(index)
        amount = int(raw, 16) / 1e18 if raw and raw != "0x" else 0.0
        if amount:
            holdings.append(Holding(chain.name, f"rewards:{token}", "VVV", amount))
    return holdings


def rpc_holdings(chain: Chain, wallet: str) -> list[Holding]:
    """Native balance and the chain's well-known tokens through the chain's public RPC."""
    url = PUBLIC_RPC[chain.name]
    tokens = WELL_KNOWN_TOKENS.get(chain.name, ())
    padded = wallet.lower()[2:].rjust(64, "0")
    calls = [{"jsonrpc": "2.0", "id": 0, "method": "eth_getBalance", "params": [wallet, "latest"]}]
    for index, token in enumerate(tokens):
        for offset, data in enumerate(("0x70a08231" + padded, "0x95d89b41", "0x313ce567")):  # balanceOf, symbol, decimals
            calls.append({"jsonrpc": "2.0", "id": 1 + 3 * index + offset, "method": "eth_call",
                          "params": [{"to": token, "data": data}, "latest"]})
    results = _rpc_batch(url, calls)
    holdings = []
    native = int(results[0], 16) / 1e18
    if native:
        holdings.append(Holding(chain.name, NATIVE, chain.native_symbol, native))
    for index, token in enumerate(tokens):
        balance, symbol, decimals = (results.get(1 + 3 * index + offset) for offset in range(3))
        if not balance or balance == "0x" or not decimals or decimals == "0x":
            continue
        amount = int(balance, 16) / 10 ** int(decimals, 16)
        if amount:
            holdings.append(Holding(chain.name, token, _decode_string(symbol) or "?", amount))
    return holdings


# ---------------------------------------------------------------------------
# Alchemy chains (full balances and transfer history with a free key)
# ---------------------------------------------------------------------------

def uses_alchemy(chain: Chain) -> bool:
    return chain.name in ALCHEMY_NETWORKS and bool(alchemy_key())


def _alchemy(chain: Chain, calls: list[tuple[str, list]]) -> list:
    """Send JSON-RPC calls as one batch and return their results in order.

    The key goes in an Authorization header so it never appears in a URL.
    """
    url = f"https://{ALCHEMY_NETWORKS[chain.name]}.g.alchemy.com/v2"
    batch = [{"jsonrpc": "2.0", "id": index, "method": method, "params": params}
             for index, (method, params) in enumerate(calls)]
    headers = {"User-Agent": USER_AGENT, "Authorization": f"Bearer {alchemy_key()}"}
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = requests.post(url, json=batch, headers=headers, timeout=60)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == REQUEST_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 429 and attempt < REQUEST_ATTEMPTS - 1:
            time.sleep(2 ** (attempt + 1))
            continue
        break
    if response.status_code in (401, 403):
        detail = response.text
        try:
            body = response.json()
            body = body[0] if isinstance(body, list) and body else body
            detail = (body.get("error") or {}).get("message", detail) if isinstance(body, dict) else detail
        except ValueError:
            pass
        if "not enabled" in str(detail):
            raise MissingKey(f"the Alchemy app does not have {ALCHEMY_NETWORKS[chain.name]} enabled "
                             "(dashboard.alchemy.com > your app > Networks)")
        raise MissingKey(f"Alchemy rejected the key ({response.status_code}): {redact(str(detail)[:120])}")
    response.raise_for_status()
    replies = response.json()
    replies = replies if isinstance(replies, list) else [replies]
    by_id = {reply.get("id"): reply for reply in replies}
    results = []
    for index in range(len(calls)):
        reply = by_id.get(index, {})
        if "error" in reply:
            raise RuntimeError(f"Alchemy {calls[index][0]}: {redact(str(reply['error'])[:160])}")
        results.append(reply.get("result"))
    return results


def alchemy_holdings(chain: Chain, wallet: str) -> list[Holding]:
    """Native balance plus every ERC-20 balance Alchemy reports, with on-chain metadata."""
    native_hex, page = _alchemy(chain, [("eth_getBalance", [wallet, "latest"]),
                                        ("alchemy_getTokenBalances", [wallet, "erc20"])])
    balances = list(page.get("tokenBalances", []))
    for _ in range(ALCHEMY_PAGE_LIMIT):
        if not page.get("pageKey"):
            break
        (page,) = _alchemy(chain, [("alchemy_getTokenBalances", [wallet, "erc20", {"pageKey": page["pageKey"]}])])
        balances.extend(page.get("tokenBalances", []))
    held = [(item["contractAddress"].lower(), int(item["tokenBalance"], 16)) for item in balances
            if item.get("tokenBalance") and int(item["tokenBalance"], 16) > 0]
    holdings = []
    if int(native_hex, 16):
        holdings.append(Holding(chain.name, NATIVE, chain.native_symbol, int(native_hex, 16) / 1e18))
    for start in range(0, len(held), 50):
        chunk = held[start:start + 50]
        metadata = _alchemy(chain, [("alchemy_getTokenMetadata", [token]) for token, _ in chunk])
        for (token, raw), meta in zip(chunk, metadata):
            decimals = (meta or {}).get("decimals")
            if decimals is None:
                continue
            holdings.append(Holding(chain.name, token, (meta or {}).get("symbol") or "?", raw / 10 ** int(decimals)))
    return holdings


def _alchemy_row(chain: Chain, wallet: str, transfer: dict) -> dict | None:
    raw = transfer.get("rawContract") or {}
    native = transfer.get("category") == "external"
    if native:
        amount = float(transfer.get("value") or 0)
    elif raw.get("value") and raw.get("decimal"):
        amount = int(raw["value"], 16) / 10 ** int(raw["decimal"], 16)
    else:
        return None
    unique = str(transfer.get("uniqueId", ""))
    log_index = int(unique.split(":log:")[1]) if ":log:" in unique else -1
    sender, receiver = transfer.get("from") or "", transfer.get("to") or ""
    return {
        "chain": chain.name, "time": pd.Timestamp(transfer["metadata"]["blockTimestamp"]),
        "hash": transfer["hash"], "log_index": log_index, "direction": _direction(wallet, sender, receiver),
        "symbol": transfer.get("asset") or ("?" if not native else chain.native_symbol),
        "token": NATIVE if native else (raw.get("address") or "").lower(), "amount": amount,
        "counterparty": receiver if sender.lower() == wallet.lower() else sender,
    }


def alchemy_activity(chain: Chain, wallet: str, since: pd.Timestamp) -> list[dict]:
    """Native and ERC-20 transfers in both directions, newest first, back to ``since``."""
    rows = []
    for side in ("fromAddress", "toAddress"):
        query = {"fromBlock": "0x0", "toBlock": "latest", side: wallet, "category": ["external", "erc20"],
                 "withMetadata": True, "excludeZeroValue": True, "maxCount": "0x3e8", "order": "desc"}
        for _ in range(ALCHEMY_PAGE_LIMIT):
            (result,) = _alchemy(chain, [("alchemy_getAssetTransfers", [query])])
            done = False
            for transfer in result.get("transfers", []):
                row = _alchemy_row(chain, wallet, transfer)
                if row is None:
                    continue
                if row["time"] < since:
                    done = True
                    break
                rows.append(row)
            if done or not result.get("pageKey"):
                break
            query = {**query, "pageKey": result["pageKey"]}
    # A self-transfer appears in both directions; keep it once.
    unique = {(row["hash"], row["log_index"], row["token"]): row for row in rows}
    return list(unique.values())


# ---------------------------------------------------------------------------
# Etherscan chains
# ---------------------------------------------------------------------------

def etherscan_activity(chain: Chain, wallet: str, key: str, since: pd.Timestamp) -> list[dict]:
    """Recent ERC-20 transfers and native-value transactions (up to 1,000 of each)."""
    rows = []
    common = {"module": "account", "address": wallet, "page": 1, "offset": 1000, "sort": "desc"}
    for item in _etherscan(chain, key, action="tokentx", **common):
        stamp = pd.Timestamp(int(item["timeStamp"]), unit="s", tz="UTC")
        if stamp < since:
            break
        decimals = int(item.get("tokenDecimal") or 0)
        rows.append({
            "chain": chain.name, "time": stamp, "hash": item["hash"], "log_index": int(item.get("logIndex") or 0),
            "direction": _direction(wallet, item["from"], item["to"]), "symbol": item.get("tokenSymbol") or "?",
            "token": item["contractAddress"].lower(), "amount": int(item["value"]) / 10 ** decimals,
            "counterparty": item["to"] if item["from"].lower() == wallet.lower() else item["from"],
        })
    for item in _etherscan(chain, key, action="txlist", **common):
        stamp = pd.Timestamp(int(item["timeStamp"]), unit="s", tz="UTC")
        if stamp < since:
            break
        if not int(item.get("value") or 0) or item.get("isError") == "1":
            continue
        rows.append({
            "chain": chain.name, "time": stamp, "hash": item["hash"], "log_index": -1,
            "direction": _direction(wallet, item["from"], item["to"]), "symbol": chain.native_symbol,
            "token": NATIVE, "amount": int(item["value"]) / 1e18,
            "counterparty": item["to"] if item["from"].lower() == wallet.lower() else item["from"],
        })
    return rows


def etherscan_holdings(chain: Chain, wallet: str, key: str, priced_tokens: set[str]) -> list[Holding]:
    """Native balance plus the current balance of every priced token the wallet ever received."""
    holdings = []
    native = int(_etherscan(chain, key, module="account", action="balance", address=wallet, tag="latest")) / 1e18
    if native:
        holdings.append(Holding(chain.name, NATIVE, chain.native_symbol, native))
    history = _etherscan(chain, key, module="account", action="tokentx", address=wallet,
                         page=1, offset=10000, sort="asc")
    seen = {}
    for item in history:
        seen[item["contractAddress"].lower()] = (item.get("tokenSymbol") or "?", int(item.get("tokenDecimal") or 0))
    for token, (symbol, decimals) in seen.items():
        if price_key(chain, token) not in priced_tokens:
            continue  # spam and unpriced tokens are never worth a balance call
        raw = _etherscan(chain, key, module="account", action="tokenbalance", contractaddress=token,
                         address=wallet, tag="latest")
        amount = int(raw or 0) / 10 ** decimals
        if amount:
            holdings.append(Holding(chain.name, token, symbol, amount))
    return holdings


def _etherscan_token_keys(chain: Chain, wallet: str, key: str) -> set[str]:
    history = _etherscan(chain, key, module="account", action="tokentx", address=wallet,
                         page=1, offset=10000, sort="asc")
    return {price_key(chain, item["contractAddress"]) for item in history}


# ---------------------------------------------------------------------------
# Desk snapshot and activity
# ---------------------------------------------------------------------------

def wallet_snapshot(account: Account, chains: list[str] | None = None) -> tuple[VenueAccount, list[str]]:
    """Priced token holdings across chains as a desk account, plus skipped-chain notes."""
    wallet = account.id
    key = etherscan_key()
    notes, holdings, unpriced = [], [], 0
    pending_etherscan = []
    for chain in (CHAINS[name] for name in (chains or CHAINS)):
        try:
            if chain.source == "blockscout":
                holdings.extend(blockscout_holdings(chain, wallet))
            elif chain.source == "blockscout_pro":
                reason = "no Alchemy or Blockscout PRO API key"
                if uses_alchemy(chain):
                    try:
                        holdings.extend(alchemy_holdings(chain, wallet))
                        continue
                    except Exception as exc:  # fall back to the keyless balances below
                        reason = redact(f"Alchemy: {exc}")
                if blockscout_key() and reason.startswith("no "):
                    try:
                        holdings.extend(blockscout_holdings(chain, wallet))
                        continue
                    except PaidPlanRequired as exc:
                        reason = f"Blockscout {exc}"
                # Keyless fallback: native and well-known tokens from the public RPC.
                holdings.extend(rpc_holdings(chain, wallet))
                notes.append(f"{chain.name}: main tokens only, no transfer history ({reason})")
            elif not key:
                notes.append(f"{chain.name} skipped: no Etherscan API key "
                             "(python -m desk.venues.onchain --store-key)")
            else:
                pending_etherscan.append(chain)
        except Exception as exc:  # one chain failing never hides the others
            notes.append(_chain_problem(chain, exc))
    for chain in (CHAINS[name] for name in (chains or CHAINS)):
        try:
            holdings.extend(staking_reward_holdings(chain, wallet))
        except Exception as exc:  # rewards are a small extra; never fail the wallet over them
            notes.append(redact(f"{chain.name} staking rewards unavailable: {type(exc).__name__}: {exc}"))
    token_keys = {price_key(CHAINS[item.chain], item.token) for item in holdings}
    for chain in list(pending_etherscan):
        try:
            token_keys |= _etherscan_token_keys(chain, wallet, key) | {chain.native_price_key}
        except Exception as exc:
            notes.append(_chain_problem(chain, exc))
            pending_etherscan.remove(chain)  # report each failing chain once
    try:
        prices = fetch_prices(token_keys | {CHAINS[name].native_price_key for name in (chains or CHAINS)})
    except Exception as exc:
        prices = {}
        notes.append(f"token prices unavailable ({type(exc).__name__}: {exc}); holdings are unvalued")
    for chain in pending_etherscan:
        try:
            holdings.extend(etherscan_holdings(chain, wallet, key, set(prices)))
        except Exception as exc:
            notes.append(_chain_problem(chain, exc))
    positions = []
    for holding in holdings:
        price = prices.get(price_key(CHAINS[holding.chain], holding.token).lower())
        if price is None:
            unpriced += 1
            continue
        value = holding.amount * price
        if value > MAX_PLAUSIBLE_HOLDING_USD:
            notes.append(f"{holding.symbol} on {holding.chain} ignored: ${value:,.0f} is implausible (likely spam)")
            continue
        if value < DUST_USD and not holding.token.startswith("rewards:"):  # rewards always show
            continue
        positions.append(VenuePosition(
            venue=VENUE, account=wallet, asset_class="crypto spot",
            symbol=(f"{holding.symbol} ({STAKED_LABELS[(holding.chain, holding.token)]}, {holding.chain})"
                    if (holding.chain, holding.token) in STAKED_LABELS
                    else f"{holding.symbol} (unclaimed staking rewards, {holding.chain})"
                    if holding.token.startswith("rewards:") else f"{holding.symbol} ({holding.chain})"),
            quantity=holding.amount, avg_price=float("nan"), mark_price=price,
            unrealized_usd=float("nan"), ref=f"{holding.chain}:{holding.token}",
        ))
    if unpriced:
        notes.append(f"{unpriced} unpriced token balance(s) ignored (mostly spam airdrops)")
    equity = sum(position.notional_usd for position in positions)
    snapshot = VenueAccount(
        venue=VENUE, account=wallet, label=f"{account.name} wallet", equity_usd=equity, cash_usd=equity,
        margin_used_usd=0.0, available_usd=equity, unrealized_usd=float("nan"),
        positions=tuple(positions), notes=tuple(notes),
    )
    return snapshot, notes


def _load_names() -> dict[str, str]:
    try:
        return json.loads(NAME_CACHE.read_text())
    except (OSError, ValueError):
        return {}


def contract_name(chain: Chain, address: str, key: str, cache: dict[str, str]) -> str:
    """Verified contract name from the chain's explorer ('' for wallets and unverified code).

    Names never change once verified, so answers (including blanks) are cached on disk.
    """
    cache_key = f"{chain.name}:{address.lower()}"
    if cache_key in cache:
        return cache[cache_key]
    name = ""
    if chain.source == "blockscout_pro" and uses_alchemy(chain):
        pass  # no free explorer for this chain; leave it unnamed
    elif chain.source == "blockscout" or (chain.source == "blockscout_pro" and blockscout_key()):
        response = _get(f"https://{chain.host}/api/v2/addresses/{address}", headers=_blockscout_headers(chain))
        if response.status_code != 404:
            info = response.json()
            name = (info.get("name") or "") if info.get("is_contract") else ""
    elif key:
        result = _etherscan(chain, key, module="contract", action="getsourcecode", address=address)
        name = (result[0].get("ContractName") or "") if isinstance(result, list) and result else ""
    cache[cache_key] = name
    return name


def label_counterparties(frame: pd.DataFrame, desk_wallets: dict[str, str], min_usd: float = 1.0) -> list[str]:
    """Fill ``label`` with the HL bridge, desk wallets, or verified contract names."""
    notes, key = [], etherscan_key()
    cache = _load_names()
    labels = []
    for chain_name, party, value, label in zip(frame["chain"], frame["counterparty"], frame["value_usd"], frame["label"]):
        party = str(party).lower()
        if label:
            labels.append(label)
        elif party in desk_wallets:
            labels.append(f"Desk wallet {desk_wallets[party]}")
        elif party == NATIVE:
            labels.append("Mint/burn")
        elif abs(value) >= min_usd:
            try:
                name = contract_name(CHAINS[chain_name], party, key, cache)
            except Exception as exc:  # labels are cosmetic; never fail the report over one
                notes.append(redact(f"could not name {party} on {chain_name}: {type(exc).__name__}"))
                name = ""
            labels.append(name)
        else:
            labels.append("")
    frame["label"] = labels
    try:
        NAME_CACHE.parent.mkdir(parents=True, exist_ok=True)
        NAME_CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True))
    except OSError:
        pass
    return notes[:3]


def wallet_activity(account: Account, since: pd.Timestamp, chains: list[str] | None = None,
                    desk_wallets: dict[str, str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Transfers on every chain since ``since``, valued at current prices; spam omitted."""
    wallet, key = account.id, etherscan_key()
    rows, notes = [], []
    for chain in (CHAINS[name] for name in (chains or CHAINS)):
        try:
            if uses_alchemy(chain):
                rows.extend(alchemy_activity(chain, wallet, since))
            elif chain.source == "blockscout" or (chain.source == "blockscout_pro" and blockscout_key()):
                rows.extend(blockscout_activity(chain, wallet, since))
            elif chain.source == "blockscout_pro":
                notes.append(f"activity: {chain.name} transfer history needs a free Alchemy API key "
                             "(python -m desk.venues.onchain --store-alchemy-key)")
            elif key and chain.source == "etherscan":
                rows.extend(etherscan_activity(chain, wallet, key, since))
        except Exception as exc:
            notes.append(f"activity: {_chain_problem(chain, exc)}")
    frame = pd.DataFrame(rows, columns=[column for column in ACTIVITY_COLUMNS
                                        if column not in {"price_usd", "value_usd", "label"}])
    if frame.empty:
        return pd.DataFrame(columns=ACTIVITY_COLUMNS), notes
    frame["price_key"] = [price_key(CHAINS[chain], token) for chain, token in zip(frame["chain"], frame["token"])]
    try:
        prices = fetch_prices(set(frame["price_key"]))
    except Exception as exc:
        prices = {}
        notes.append(f"token prices unavailable ({type(exc).__name__}: {exc})")
    frame["price_usd"] = frame["price_key"].str.lower().map(prices)
    spam = frame["price_usd"].isna()
    if spam.any():
        notes.append(f"{int(spam.sum())} transfer(s) of unpriced tokens omitted (mostly spam airdrops)")
    frame = frame[~spam].drop(columns="price_key")
    frame["value_usd"] = frame["amount"] * frame["price_usd"]
    frame["label"] = [KNOWN_COUNTERPARTIES.get((chain, str(party).lower()), "")
                      for chain, party in zip(frame["chain"], frame["counterparty"])]
    notes += label_counterparties(frame, {wallet.lower(): name for wallet, name in (desk_wallets or {}).items()
                                          if wallet.lower() != account.id.lower()})
    return frame[ACTIVITY_COLUMNS].sort_values("time", ascending=False).reset_index(drop=True), notes


def update_activity_file(account: Account, data_dir: Path, lookback_days: int = 365,
                         chains: list[str] | None = None,
                         desk_wallets: dict[str, str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Fetch new transfers and merge them into ``onchain_activity_<wallet>.csv``.

    The first run reads ``lookback_days``; later runs start a day before the newest saved
    transfer. Values are re-priced at current prices on every run.
    """
    path = Path(data_dir) / f"onchain_activity_{account.id.lower()}.csv"
    saved = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=ACTIVITY_COLUMNS)
    if not saved.empty:
        saved["time"] = pd.to_datetime(saved["time"], utc=True, format="mixed")
        since = saved["time"].max() - pd.Timedelta(days=1)
    else:
        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    fresh, notes = wallet_activity(account, since, chains, desk_wallets)
    frames = [frame for frame in (saved, fresh) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=ACTIVITY_COLUMNS), notes
    merged = pd.concat(frames, ignore_index=True)
    merged["hash"] = merged["hash"].str.lower()
    merged = merged.drop_duplicates(["chain", "hash", "log_index", "token"], keep="last")
    merged = merged.sort_values("time", ascending=False).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged, notes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="On-chain wallet reader for the trading desk.")
    parser.add_argument("--store-key", action="store_true", help="Save an Etherscan API key in the Keychain.")
    parser.add_argument("--store-alchemy-key", action="store_true",
                        help="Save an Alchemy API key (free at alchemy.com) in the Keychain.")
    parser.add_argument("--store-blockscout-key", action="store_true",
                        help="Save a Blockscout PRO API key (free at dev.blockscout.com) in the Keychain.")
    args = parser.parse_args()
    if args.store_key:
        store_etherscan_key()
    if args.store_blockscout_key:
        store_blockscout_key()
    if args.store_alchemy_key:
        store_alchemy_key()
