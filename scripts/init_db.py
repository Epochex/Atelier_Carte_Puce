from src.config import load_config
from src.db import connect, init_db

def main():
    cfg = load_config("config.yaml")
    conn = connect(cfg.db_path)
    init_db(conn)
    print(f"DB initialized: {cfg.db_path}")

if __name__ == "__main__":
    main()
