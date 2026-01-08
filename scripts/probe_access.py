#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.card import open_card

from src.card import (
    CardError,
    VERIFY_TARGET_CSC0,
    VERIFY_TARGET_CSC1,
    VERIFY_TARGET_CSC2,
    ADDR_ISSUER_BASE,
    APP_WORD_BASE,
    APP_WORD_LEN,
)


CSC0_HEX = "00112233"
CSC1_HEX = "44556677"
CSC2_HEX = "89aabbcc"


def _parse_hex_code4(s: str) -> bytes:
    v = (s or "").strip().lower().replace("0x", "").replace(" ", "")
    b = bytes.fromhex(v)
    if len(b) != 4:
        raise ValueError("code4_must_be_4_bytes_hex")
    return b


def _candidate_codes() -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    for name, v in [("CSC0", CSC0_HEX), ("CSC1", CSC1_HEX), ("CSC2", CSC2_HEX)]:
        if v and v.strip():
            out.append((name, _parse_hex_code4(v)))
    return out


def _range_words(start: int, count: int) -> List[int]:
    return [start + i for i in range(count)]


def _fmt_addr(a: int) -> str:
    return f"0x{a:02X}"


def _safe_read_word(s, addr: int) -> Tuple[bool, Optional[bytes], Optional[str]]:
    try:
        data = s.read_word(addr)
        return True, data, None
    except Exception as e:
        return False, None, str(e)


def _safe_update_same_word(s, addr: int) -> Tuple[bool, Optional[str]]:
    ok, data, err = _safe_read_word(s, addr)
    if not ok or data is None:
        return False, f"read_failed_before_update: {err}"
    try:
        s.update_word(addr, data)
        return True, None
    except Exception as e:
        return False, str(e)


@dataclass
class VerifyOutcome:
    target_name: str
    target_p2: int
    ok: bool
    matched_source: Optional[str] = None
    sw_error: Optional[str] = None


def _try_verify_targets(s, codes: List[Tuple[str, bytes]]) -> List[VerifyOutcome]:
    targets = [
        ("CSC0", VERIFY_TARGET_CSC0),
        ("CSC1", VERIFY_TARGET_CSC1),
        ("CSC2", VERIFY_TARGET_CSC2),
    ]

    outcomes: List[VerifyOutcome] = []

    for name, p2 in targets:
        matched = None
        last_err = None
        for src_name, c in codes:
            try:
                s.verify(p2, c)
                matched = src_name
                last_err = None
                break
            except Exception as e:
                last_err = str(e)
                continue

        outcomes.append(
            VerifyOutcome(
                target_name=name,
                target_p2=p2,
                ok=(matched is not None),
                matched_source=matched,
                sw_error=None if matched is not None else last_err,
            )
        )
    return outcomes


def _print_block(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe card access conditions (read/write/verify) for CartePuce")
    parser.add_argument("--timeout", type=int, default=10, help="Card wait timeout (seconds)")
    parser.add_argument(
        "--write-test",
        action="store_true",
        help="Perform UPDATE probes (update same data) to infer write permissions (sends UPDATE APDU).",
    )
    parser.add_argument(
        "--write-test-scope",
        choices=["app", "issuer", "range"],
        default="app",
        help="Which addresses to test UPDATE on: app(0x10..), issuer(0x01..), range(custom)",
    )
    parser.add_argument(
        "--range",
        default=None,
        help="Custom address range for probing, format: START:END (hex), e.g. 00:3F. Used for read scan; also for write-test-scope=range.",
    )
    args = parser.parse_args()

    try:
        s = open_card(args.timeout)
    except Exception as e:
        print(f"[ERROR] cannot open card: {e}")
        return 2

    _print_block("CARD BASIC INFO")
    print(f"ATR: {s.atr_hex}")

    issuer_addrs = _range_words(ADDR_ISSUER_BASE, 4)
    print(f"ISSUER_SN addresses: {_fmt_addr(issuer_addrs[0])}..{_fmt_addr(issuer_addrs[-1])}")

    if args.range:
        try:
            a, b = args.range.split(":")
            start = int(a, 16)
            end = int(b, 16)
            if not (0 <= start <= 0xFF and 0 <= end <= 0xFF and start <= end):
                raise ValueError("range_out_of_bounds")
        except Exception as e:
            raise SystemExit(f"bad --range '{args.range}': {e}")
        scan_addrs = list(range(start, end + 1))
        scan_label = f"custom_range({_fmt_addr(start)}..{_fmt_addr(end)})"
    else:
        scan_addrs = sorted(set(issuer_addrs + _range_words(APP_WORD_BASE, APP_WORD_LEN)))
        scan_label = "default(issuer + app_record)"

    _print_block(f"READ PROBE: {scan_label}")
    read_ok: List[int] = []
    read_fail: List[int] = []
    for addr in scan_addrs:
        ok, data, err = _safe_read_word(s, addr)
        if ok and data is not None:
            read_ok.append(addr)
            print(f"[READ OK ] addr={_fmt_addr(addr)} data={data.hex()}")
        else:
            read_fail.append(addr)
            print(f"[READ DENY] addr={_fmt_addr(addr)} err={err}")

    codes = _candidate_codes()

    _print_block("VERIFY PROBE (CSC0/CSC1/CSC2)")
    if not codes:
        print("No CSC codes configured in code.")
        return 1

    outcomes = _try_verify_targets(s, codes)

    verified: Dict[str, bool] = {}
    for o in outcomes:
        verified[o.target_name] = o.ok
        if o.ok:
            print(f"[VERIFY OK ] target={o.target_name} P2=0x{o.target_p2:02X} matched={o.matched_source}")
        else:
            print(f"[VERIFY NO ] target={o.target_name} P2=0x{o.target_p2:02X} err={o.sw_error}")

    if args.write_test:
        _print_block("WRITE PROBE (UPDATE same-word)")

        if args.write_test_scope == "app":
            w_addrs = _range_words(APP_WORD_BASE, APP_WORD_LEN)
            w_label = f"app_record({_fmt_addr(w_addrs[0])}..{_fmt_addr(w_addrs[-1])})"
        elif args.write_test_scope == "issuer":
            w_addrs = issuer_addrs
            w_label = f"issuer_sn({_fmt_addr(w_addrs[0])}..{_fmt_addr(w_addrs[-1])})"
        else:
            if not args.range:
                raise SystemExit("write-test-scope=range requires --range START:END")
            w_addrs = scan_addrs
            w_label = f"custom_range({_fmt_addr(w_addrs[0])}..{_fmt_addr(w_addrs[-1])})"

        print(f"Scope: {w_label}")

        print("Phase 1: UPDATE with current session state")
        for addr in w_addrs:
            ok, err = _safe_update_same_word(s, addr)
            if ok:
                print(f"[UPD OK ] addr={_fmt_addr(addr)}")
            else:
                print(f"[UPD NO ] addr={_fmt_addr(addr)} err={err}")

        _print_block("WRITE PROBE Phase 2: unlock (CSC1 -> CSC0 -> CSC2) then UPDATE again")
        unlock_targets = [("CSC1", VERIFY_TARGET_CSC1), ("CSC0", VERIFY_TARGET_CSC0), ("CSC2", VERIFY_TARGET_CSC2)]
        unlocked = False
        last_err = None
        for name, p2 in unlock_targets:
            for src_name, c in codes:
                try:
                    s.verify(p2, c)
                    print(f"[UNLOCK OK] via {name} (P2=0x{p2:02X}) using {src_name}")
                    unlocked = True
                    break
                except Exception as e:
                    last_err = str(e)
            if unlocked:
                break

        if not unlocked:
            print(f"[UNLOCK NO] last_err={last_err}")
        else:
            for addr in w_addrs:
                ok, err = _safe_update_same_word(s, addr)
                if ok:
                    print(f"[UPD OK ] addr={_fmt_addr(addr)}")
                else:
                    print(f"[UPD NO ] addr={_fmt_addr(addr)} err={err}")

    _print_block("INFERENCE SUMMARY (best-effort)")
    print("Named regions (based on current project address usage):")
    print(f"  - ISSUER_SN: {_fmt_addr(ADDR_ISSUER_BASE)}..{_fmt_addr(ADDR_ISSUER_BASE+3)}")
    print(f"  - APP_RECORD: {_fmt_addr(APP_WORD_BASE)}..{_fmt_addr(APP_WORD_BASE+APP_WORD_LEN-1)}")

    issuer_can_read = all(a in read_ok for a in issuer_addrs)
    app_can_read = all(a in read_ok for a in _range_words(APP_WORD_BASE, APP_WORD_LEN))

    print("\nObserved READ capability in this run:")
    print(f"  - ISSUER_SN readable: {issuer_can_read}")
    print(f"  - APP_RECORD readable: {app_can_read}")

    print("\nObserved VERIFY capability in this run:")
    for k in ["CSC0", "CSC1", "CSC2"]:
        print(f"  - {k}: {verified.get(k, False)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
