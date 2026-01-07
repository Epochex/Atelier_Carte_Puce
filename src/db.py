from __future__ import annotations

import os
import sqlite3
import time
from typing import Optional, Any, Dict, Tuple


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
  user_id      TEXT PRIMARY KEY,
  card_id      TEXT UNIQUE NOT NULL,
  card_atr     TEXT,
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

CREATE TABLE IF NOT EXISTS auth_state (
  user_id            TEXT PRIMARY KEY,
  fail_count         INTEGER NOT NULL DEFAULT 0,
  locked_until_epoch INTEGER,
  last_fail_epoch    INTEGER,
  updated_at         TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS auth_logs (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        TEXT NOT NULL DEFAULT (datetime('now')),
  card_id   TEXT,
  card_atr  TEXT,
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
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, ddl: str) -> None:
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
        conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    _ensure_column(conn, "users", "card_atr", "card_atr TEXT")
    _ensure_column(conn, "auth_logs", "card_atr", "card_atr TEXT")


def upsert_user(
    conn: sqlite3.Connection,
    user_id: str,
    card_id: str,
    pwd_salt: bytes,
    pwd_hash: bytes,
    card_atr: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO users(user_id, card_id, card_atr, pwd_salt, pwd_hash)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          card_id=excluded.card_id,
          card_atr=excluded.card_atr,
          pwd_salt=excluded.pwd_salt,
          pwd_hash=excluded.pwd_hash
        """,
        (user_id, card_id, card_atr, pwd_salt, pwd_hash),
    )
    conn.commit()


def update_user_pin(conn: sqlite3.Connection, user_id: str, pwd_salt: bytes, pwd_hash: bytes) -> None:
    conn.execute(
        """
        UPDATE users SET pwd_salt=?, pwd_hash=? WHERE user_id=?
        """,
        (pwd_salt, pwd_hash, user_id),
    )
    conn.commit()


def upsert_biometric(
    conn: sqlite3.Connection,
    user_id: str,
    template_path: str,
    template_sha256: str,
    algo: str,
) -> None:
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
        SELECT u.user_id, u.card_id, u.card_atr, u.pwd_salt, u.pwd_hash,
               b.template_path, b.template_sha256, b.algo
        FROM users u
        LEFT JOIN biometrics b ON b.user_id = u.user_id
        WHERE u.card_id = ?
        """,
        (card_id,),
    ).fetchone()
    return dict(row) if row else None


def get_user_by_id(conn: sqlite3.Connection, user_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT u.user_id, u.card_id, u.card_atr, u.pwd_salt, u.pwd_hash,
               b.template_path, b.template_sha256, b.algo
        FROM users u
        LEFT JOIN biometrics b ON b.user_id = u.user_id
        WHERE u.user_id = ?
        """,
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def _ensure_auth_state(conn: sqlite3.Connection, user_id: str) -> None:
    conn.execute(
        """
        INSERT INTO auth_state(user_id, fail_count, locked_until_epoch, last_fail_epoch)
        VALUES(?, 0, NULL, NULL)
        ON CONFLICT(user_id) DO NOTHING
        """,
        (user_id,),
    )
    conn.commit()


def get_auth_state(conn: sqlite3.Connection, user_id: str) -> Dict[str, Any]:
    _ensure_auth_state(conn, user_id)
    row = conn.execute(
        """
        SELECT user_id, fail_count, locked_until_epoch, last_fail_epoch
        FROM auth_state WHERE user_id=?
        """,
        (user_id,),
    ).fetchone()
    return dict(row) if row else {"user_id": user_id, "fail_count": 0, "locked_until_epoch": None, "last_fail_epoch": None}


def is_locked(conn: sqlite3.Connection, user_id: str, now_epoch: Optional[int] = None) -> Tuple[bool, Optional[int]]:
    st = get_auth_state(conn, user_id)
    locked_until = st.get("locked_until_epoch")
    if locked_until is None:
        return False, None
    if now_epoch is None:
        now_epoch = int(time.time())
    return now_epoch < int(locked_until), int(locked_until)


def record_pin_failure(
    conn: sqlite3.Connection,
    user_id: str,
    now_epoch: int,
    max_attempts: int,
    lockout_seconds: int,
) -> Dict[str, Any]:
    st = get_auth_state(conn, user_id)
    fail_count = int(st.get("fail_count") or 0) + 1
    locked_until = st.get("locked_until_epoch")
    if max_attempts > 0 and fail_count >= max_attempts:
        locked_until = int(now_epoch + max(1, lockout_seconds))
    conn.execute(
        """
        UPDATE auth_state
        SET fail_count=?, locked_until_epoch=?, last_fail_epoch=?, updated_at=datetime('now')
        WHERE user_id=?
        """,
        (fail_count, locked_until, int(now_epoch), user_id),
    )
    conn.commit()
    return {"user_id": user_id, "fail_count": fail_count, "locked_until_epoch": locked_until}


def clear_auth_state(conn: sqlite3.Connection, user_id: str) -> None:
    _ensure_auth_state(conn, user_id)
    conn.execute(
        """
        UPDATE auth_state
        SET fail_count=0, locked_until_epoch=NULL, last_fail_epoch=NULL, updated_at=datetime('now')
        WHERE user_id=?
        """,
        (user_id,),
    )
    conn.commit()


def log_auth(
    conn: sqlite3.Connection,
    card_id: str,
    card_atr: Optional[str],
    user_id: Optional[str],
    pwd_ok: bool,
    bio_score: Optional[float],
    decision: str,
    reason: str,
) -> None:
    conn.execute(
        """
        INSERT INTO auth_logs(card_id, card_atr, user_id, pwd_ok, bio_score, decision, reason)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (card_id, card_atr, user_id, int(pwd_ok), bio_score, decision, reason),
    )
    conn.commit()
