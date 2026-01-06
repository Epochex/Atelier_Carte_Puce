import argparse
from typing import Optional

from src.card import CardSession


def _print_app_record(r) -> None:
    if r is None:
        print("app_record: None")
        return
    print("app_record:")
    print(f"  card_uid  = {r.card_uid}")
    print(f"  user_hash8 = {r.user_hash8}")
    print(f"  tpl_hash8  = {r.tpl_hash8}")


def main():
    parser = argparse.ArgumentParser(description="Print detailed card information")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout seconds when waiting for a card")
    parser.add_argument("--simulate-card-id", dest="sim_card_id", help="Simulated card_id (for offline testing)")
    parser.add_argument("--simulate-atr", dest="sim_atr", help="Simulated ATR hex string (for offline testing)")
    args = parser.parse_args()

    # If simulation args are provided, print them and exit
    if args.sim_card_id or args.sim_atr:
        print("Simulated card info:")
        print(f"  card_id = {args.sim_card_id}")
        print(f"  atr_hex = {args.sim_atr}")
        return

    try:
        s = CardSession(timeout_seconds=args.timeout)
    except Exception as e:
        print(f"No card found or error: {e}")
        return

    # ATR
    print(f"ATR: {s.atr_hex}")

    # Issuer SN
    try:
        issuer = s.get_issuer_sn()
        print(f"Issuer SN (raw): {issuer.hex()}")
    except Exception as e:
        print(f"Could not read issuer SN: {e}")

    # UID derived from issuer
    try:
        uid_from_issuer = s.uid_from_issuer()
        print(f"UID from issuer: {uid_from_issuer}")
    except Exception as e:
        print(f"Could not compute uid from issuer: {e}")

    # Stored app record UID
    card_id = s.get_uid()
    print(f"Stored card_id (app record): {card_id}")

    # App record details
    try:
        app = s.read_app_record()
        _print_app_record(app)
    except Exception as e:
        print(f"Could not read app record: {e}")


if __name__ == "__main__":
    main()
