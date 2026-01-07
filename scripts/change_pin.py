import argparse
import time
import getpass

from src.config import load_config
from src.db import (
    connect,
    init_db,
    get_user_by_card,
    get_user_by_id,
    update_user_pin,
    is_locked,
    record_pin_failure,
    clear_auth_state,
)
from src.security import verify_pin, pbkdf2_hash_pin
from src.card import open_card


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default=None)
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    conn = connect(cfg.db_path)
    init_db(conn)

    if args.user_id:
        user = get_user_by_id(conn, args.user_id)
        if not user:
            raise SystemExit("unknown_user")
        card_id = user["card_id"]
    else:
        s = open_card(10)
        card_id = s.get_uid()
        if not card_id:
            raise SystemExit("no_card")
        user = get_user_by_card(conn, card_id)
        if not user:
            raise SystemExit("unknown_card")

    user_id = user["user_id"]
    now = int(time.time())
    locked, _ = is_locked(conn, user_id, now_epoch=now)
    if locked:
        raise SystemExit("locked_out")

    old_pin = getpass.getpass("Enter current PIN/mot de passe: ").strip()
    if not verify_pin(old_pin, user["pwd_salt"], user["pwd_hash"]):
        record_pin_failure(
            conn,
            user_id,
            now_epoch=now,
            max_attempts=cfg.auth.max_pin_attempts,
            lockout_seconds=cfg.auth.lockout_seconds,
        )
        raise SystemExit("bad_pin")

    new_pin1 = getpass.getpass("Enter new PIN/mot de passe: ").strip()
    new_pin2 = getpass.getpass("Re-enter new PIN/mot de passe: ").strip()
    if not new_pin1 or new_pin1 != new_pin2:
        raise SystemExit("pin_mismatch")
