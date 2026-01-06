from src.card import get_card_id

def main():
    info = get_card_id(simulate_card=None, timeout_seconds=30)
    print(f"card_id(ATR) = {info.card_id}")

if __name__ == "__main__":
    main()
