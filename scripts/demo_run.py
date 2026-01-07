import getpass

from src.config import load_config
from src.db import connect, init_db
from src.auth_flow import run_auth_flow
from src.card import open_card


def main():
    cfg = load_config("config.yaml")
    conn = connect(cfg.db_path)
    init_db(conn)

    s = open_card(10)
    card_uid = s.get_uid()
    if not card_uid:
        print("[ACCESS DENIED] user=None bio_score=None reason=unknown_card")
        return

    rec = s.read_app_record()
    user_hash8 = rec.user_hash8 if rec else None
    tpl_hash8 = rec.tpl_hash8 if rec else None

    print(f"[DETECTED] card_id={card_uid} atr={s.atr_hex}")

    pin = getpass.getpass("Enter PIN/mot de passe: ").strip()


    result = run_auth_flow(
        cfg,
        conn,
        card_id=card_uid,
        card_atr=s.atr_hex,
        pin=pin,
        card_user_hash8=user_hash8,
        card_tpl_hash8=tpl_hash8,
    )

    if result.decision == "ALLOW":
        if result.bio_score is None:
            print(f"[ACCESS GRANTED] user={result.user_id} bio_score=None reason={result.reason}")
        else:
            print(f"[ACCESS GRANTED] user={result.user_id} bio_score={result.bio_score:.3f} reason={result.reason}")
    else:
        print(f"[ACCESS DENIED] user={result.user_id} bio_score={result.bio_score} reason={result.reason}")


if __name__ == "__main__":
    main()
