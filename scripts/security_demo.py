#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

from src.config import load_config
from src.db import connect, init_db, upsert_user, upsert_biometric, log_auth, get_user_by_card
from src.security import (
    pbkdf2_hash_pin,
    verify_pin,
    NonceReplayProtector,
    generate_nonce,
    new_card_key,
    key_id_from_key,
    build_hmac_message,
    compute_hmac_tag,
    verify_hmac_tag,
    sha256_file_hex,
    verify_file_sha256,
    build_audit_context,
    encode_audit_context,
    compact_reason,
)


def _print(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _ensure_template(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Create a deterministic dummy template file (not an actual face)
    with open(path, "wb") as f:
        f.write(b"DEMO_TEMPLATE_V1\n" + b"\x00" * 1024)


def demo_password_hashing() -> None:
    _print("1) Password hashing demo (PBKDF2 + optional pepper compatibility)")

    pin = "1234"
    salt, h = pbkdf2_hash_pin(pin)
    ok = verify_pin(pin, salt, h)
    bad = verify_pin("9999", salt, h)
    print(f"verify_pin(correct) = {ok} (expected True)")
    print(f"verify_pin(wrong)   = {bad} (expected False)")

    print("\n[pepper] If you set CARTEPUCE_PASSWORD_PEPPER and re-run, verify_pin remains backward-compatible.")
    print("Example:")
    print('  export CARTEPUCE_PASSWORD_PEPPER="my-pepper"')
    print("  python3 scripts/security_demo.py --only password")


def demo_template_integrity(tmp_path: str) -> None:
    _print("2) Template integrity demo (detect template replacement)")

    _ensure_template(tmp_path)
    expected = sha256_file_hex(tmp_path)
    ok = verify_file_sha256(tmp_path, expected)
    print(f"verify_file_sha256(original) = {ok} (expected True)")

    # Tamper file
    with open(tmp_path, "ab") as f:
        f.write(b"\nTAMPER\n")

    bad = verify_file_sha256(tmp_path, expected)
    print(f"verify_file_sha256(tampered)  = {bad} (expected False)")


def demo_replay_protection(card_id: str) -> None:
    _print("3) Replay protection demo (nonce re-use detection)")

    rp = NonceReplayProtector(ttl_seconds=30, max_entries=128)
    nonce = generate_nonce(16)

    d1 = rp.check_and_remember(card_id, nonce)
    d2 = rp.check_and_remember(card_id, nonce)  # replay immediately
    print(f"first use : ok={d1.ok}, reason={d1.reason} (expected ok)")
    print(f"replay    : ok={d2.ok}, reason={d2.reason} (expected replay_nonce_seen)")

    print("\n(wait TTL expiration demo)")
    time.sleep(2)
    print("You can re-run with shorter TTL to show expiry behavior if needed.")


def demo_hmac_challenge_response(card_id: str) -> None:
    _print("4) HMAC challenge-response demo (host-side secret, constant-time verify)")

    k_card = new_card_key(32)
    key_id = key_id_from_key(k_card)
    nonce = generate_nonce(16)
    counter = 1
    context = "demo-login@terminal-1"

    msg = build_hmac_message(card_id, nonce, counter, context)
    tag = compute_hmac_tag(k_card, msg, tag_len=16)

    ok = verify_hmac_tag(k_card, msg, tag)
    print(f"key_id={key_id}")
    print(f"tag(ok)={ok} (expected True)")

    # Attack: modify nonce or context
    msg2 = build_hmac_message(card_id, generate_nonce(16), counter, context)
    bad = verify_hmac_tag(k_card, msg2, tag)
    print(f"tag(after message change)={bad} (expected False)")

    # Show audit context packing
    ctx = build_audit_context(nonce=nonce, counter=counter, key_id=key_id, tag=tag, context=context)
    ctx_json = encode_audit_context(ctx)
    reason = compact_reason("hmac_demo", ctx_json)
    print(f"audit_reason_field_example:\n  {reason}")


def demo_auth_logs(db_path: str) -> None:
    _print("5) Audit log demo (write synthetic auth logs + query)")

    conn = connect(db_path)
    init_db(conn)

    # synthetic log entries
    log_auth(conn, "deadbeef" * 4, "3B 02 53 01", "alice", True, 0.42, "ALLOW", "ok")
    log_auth(conn, "deadbeef" * 4, "3B 02 53 01", "alice", False, None, "DENY", "bad_pin")

    rows = conn.execute(
        "SELECT ts, card_id, user_id, pwd_ok, bio_score, decision, reason FROM auth_logs ORDER BY id DESC LIMIT 5"
    ).fetchall()
    for r in rows:
        print(dict(r))


def demo_db_offline_attack_model(db_path: str) -> None:
    _print("6) DB compromise model demo (offline guessing cost + pepper)")

    print("This demo explains the threat model rather than brute-forcing.")
    print("- pwd_hash is PBKDF2-HMAC-SHA256 with 200k iterations (costly offline).")
    print("- If CARTEPUCE_PASSWORD_PEPPER is set, DB-only compromise is insufficient to verify guesses.")
    print(f"DB path: {db_path}")
    print("Check your users table stores only salt/hash, not plaintext.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Security demo runner for CartePuce project")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--template", default="data/templates/demo_template.bin", help="Template file for integrity demo")
    parser.add_argument("--card-id", default="f22c9c89dce6ca8876dcc20421af84d3", help="Demo card_id (16B hex -> 32 chars)")
    parser.add_argument(
        "--only",
        choices=["password", "template", "replay", "hmac", "logs", "dbmodel", "all"],
        default="all",
        help="Run only one section",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.only in ("password", "all"):
        demo_password_hashing()

    if args.only in ("template", "all"):
        demo_template_integrity(args.template)

    if args.only in ("replay", "all"):
        demo_replay_protection(args.card_id)

    if args.only in ("hmac", "all"):
        demo_hmac_challenge_response(args.card_id)

    if args.only in ("logs", "all"):
        demo_auth_logs(cfg.db_path)

    if args.only in ("dbmodel", "all"):
        demo_db_offline_attack_model(cfg.db_path)

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
