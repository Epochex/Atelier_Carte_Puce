from src.config import load_config
from src.db import connect, init_db
from src.auth_flow import run_auth_flow
from src.card import open_card


def main():
    cfg = load_config("config.yaml")
    conn = connect(cfg.db_path)
    init_db(conn)

    s = open_card(10)
    card_uid = s.get_uid_or_none()
    if not card_uid:
        print("DENY  user=None bio_score=None reason=unknown_card")
        return

    print(f"[card] detected card_id={card_uid} atr={s.atr_hex}")

    pin = input("Enter PIN/mot de passe: ").strip()

    result = run_auth_flow(cfg, conn, card_id=card_uid, pin=pin)

    if result.decision == "ALLOW":
        print(f"ALLOW user={result.user_id} bio_score={result.bio_score:.3f} reason={result.reason}")
    else:
        print(f"DENY  user={result.user_id} bio_score={result.bio_score} reason={result.reason}")


if __name__ == "__main__":
    main()
