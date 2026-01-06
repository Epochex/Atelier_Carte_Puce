from __future__ import annotations
import os
import sqlite3
from typing import Optional, Tuple, Any, Dict


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
  user_id      TEXT PRIMARY KEY,
  card_id      TEXT UNIQUE NOT NULL,
  pwd_salt     BLOB NOT NULL,
  pwd_hash     BLOB NOT NULL,
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS biometrics (
  user_id         TEXT PRIMARY KEY,
  template_path   TEXT NOT NULL,
  template_sha256 TEXT NOT NULL,
  algo            TEXT NOT NULL,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS auth_logs (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        TEXT NOT NULL DEFAULT (datetime('now')),
  card_id   TEXT,
  user_id   TEXT,
  pwd_ok    INTEGER,
  bio_score REAL,
  decision  TEXT,
  reason    TEXT
);
"""


def ensure_parent_dir(db_path: str) -> None:
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def connect(db_path: str) -> sqlite3.Connection:
    ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def upsert_user(conn: sqlite3.Connection, user_id: str, card_id: str, pwd_salt: bytes, pwd_hash: bytes) -> None:
    conn.execute(
        """
        INSERT INTO users(user_id, card_id, pwd_salt, pwd_hash)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          card_id=excluded.card_id,
          pwd_salt=excluded.pwd_salt,
          pwd_hash=excluded.pwd_hash
        """,
        (user_id, card_id, pwd_salt, pwd_hash),
    )
    conn.commit()


def upsert_biometric(conn: sqlite3.Connection, user_id: str, template_path: str, template_sha256: str, algo: str) -> None:
    conn.execute(
        """
        INSERT INTO biometrics(user_id, template_path, template_sha256, algo)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          template_path=excluded.template_path,
          template_sha256=excluded.template_sha256,
          algo=excluded.algo
        """,
        (user_id, template_path, template_sha256, algo),
    )
    conn.commit()


def get_user_by_card(conn: sqlite3.Connection, card_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT u.user_id, u.card_id, u.pwd_salt, u.pwd_hash,
               b.template_path, b.template_sha256, b.algo
        FROM users u
        LEFT JOIN biometrics b ON b.user_id = u.user_id
        WHERE u.card_id = ?
        """,
        (card_id,),
    ).fetchone()
    return dict(row) if row else None


def log_auth(conn: sqlite3.Connection, card_id: str, user_id: Optional[str], pwd_ok: bool,
             bio_score: Optional[float], decision: str, reason: str) -> None:
    conn.execute(
        """
        INSERT INTO auth_logs(card_id, user_id, pwd_ok, bio_score, decision, reason)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        (card_id, user_id, int(pwd_ok), bio_score, decision, reason),
    )
    conn.commit()
