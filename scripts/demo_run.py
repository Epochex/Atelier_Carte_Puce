import argparse
from src.config import load_config
from src.db import connect, init_db
from src.card import get_card_id
from src.auth_flow import run_auth_flow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-card", default=None, help="use this as card_id (skip real card)")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    conn = connect(cfg.db_path)
    init_db(conn)

    # Insertion de la carte
    card = get_card_id(simulate_card=args.simulate_card, timeout_seconds=60)
    print(f"[card] detected card_id = {card.card_id}")

    # PIN/mot de passe
    pin = input("Enter PIN/mot de passe: ").strip()

    # capture -> comparaison -> DB -> autorisation/refus
    result = run_auth_flow(cfg, conn, card_id=card.card_id, pin=pin)

    if result.decision == "ALLOW":
        print(f"ALLOW user={result.user_id} bio_score={result.bio_score:.3f} reason={result.reason}")
    else:
        print(f"DENY  user={result.user_id} bio_score={result.bio_score} reason={result.reason}")

if __name__ == "__main__":
    main()
