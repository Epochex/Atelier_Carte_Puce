from __future__ import annotations

import os
import time
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple, List

from smartcard.CardType import ATRCardType
from smartcard.CardRequest import CardRequest
from smartcard.CardConnection import CardConnection
from smartcard.Exceptions import CardConnectionException


class CardError(RuntimeError):
    pass


GCM_ATR = [0x3B, 0x02, 0x53, 0x01]
GCM_ATR_MASK = [0xFF, 0xFF, 0xFF, 0x00]
GCM_CARD_TYPE = ATRCardType(GCM_ATR, GCM_ATR_MASK)

INS_READ = 0xBE
INS_UPDATE = 0xDE
INS_VERIFY = 0x20

ADDR_ISSUER_BASE = 0x01
ADDR_USER1_BASE = 0x10

APP_WORD_BASE = ADDR_USER1_BASE
APP_WORD_LEN = 10

MAGIC = b"CP01"

VERIFY_TARGET_CSC0 = 0x07
VERIFY_TARGET_CSC1 = 0x39
VERIFY_TARGET_CSC2 = 0x3B


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _chunks4(b: bytes):
    if len(b) % 4 != 0:
        raise CardError("record_length_not_multiple_of_4")
    return [b[i : i + 4] for i in range(0, len(b), 4)]


def _env_code4(name: str) -> Optional[bytes]:
    v = os.getenv(name, "").strip()
    if not v:
        return None
    v = v.lower().replace("0x", "").replace(" ", "")
    try:
        b = bytes.fromhex(v)
    except ValueError:
        raise CardError(f"bad_env_hex:{name}")
    if len(b) != 4:
        raise CardError(f"bad_env_len:{name}")
    return b


def _candidate_codes() -> List[bytes]:
    out: List[bytes] = []
    for k in ["CARTEPUCE_CSC1_HEX", "CARTEPUCE_CSC0_HEX", "CARTEPUCE_CSC2_HEX"]:
        b = _env_code4(k)
        if b:
            out.append(b)
    common = [
        b"\x11\x11\x11\x11",
        b"\x22\x22\x22\x22",
        b"\xAA\xAA\xAA\xAA",
        b"\x00\x00\x00\x00",
        b"\xFF\xFF\xFF\xFF",
    ]
    for c in common:
        if c not in out:
            out.append(c)
    return out


@dataclass
class AppRecord:
    card_uid: str
    user_hash8: str
    tpl_hash8: str


class CardSession:
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds
        self.cla = 0x80
        self._svc = None
        self._conn = None
        self.atr_hex = ""
        self._acquire()

    @property
    def conn(self):
        if self._conn is None:
            raise CardError("no_connection")
        return self._conn

    def _acquire(self) -> None:
        req = CardRequest(timeout=self.timeout_seconds, cardType=GCM_CARD_TYPE)
        svc = req.waitforcard()
        conn = svc.connection
        conn.connect(protocol=CardConnection.T0_protocol)
        atr = bytes(conn.getATR())
        self._svc = svc
        self._conn = conn
        self.atr_hex = " ".join(f"{x:02X}" for x in atr)

    def _reacquire(self) -> None:
        last = None
        for _ in range(3):
            try:
                self._acquire()
                return
            except Exception as e:
                last = e
                time.sleep(0.2)
        raise CardError(f"reacquire_failed:{last}")

    def _transmit(self, apdu) -> Tuple[bytes, int, int]:
        try:
            data, sw1, sw2 = self.conn.transmit(apdu)
            return bytes(data), sw1, sw2
        except CardConnectionException:
            self._reacquire()
            data, sw1, sw2 = self.conn.transmit(apdu)
            return bytes(data), sw1, sw2

    def read_word(self, addr: int) -> bytes:
        apdu = [self.cla, INS_READ, 0x00, addr & 0xFF, 0x04]
        data, sw1, sw2 = self._transmit(apdu)
        if (sw1, sw2) != (0x90, 0x00):
            raise CardError(f"read_word_sw={sw1:02X}{sw2:02X}")
        if len(data) != 4:
            raise CardError("read_word_len")
        return data

    def update_word(self, addr: int, word4: bytes) -> None:
        if len(word4) != 4:
            raise CardError("update_word_len")
        apdu = [self.cla, INS_UPDATE, 0x00, addr & 0xFF, 0x04] + list(word4)
        _, sw1, sw2 = self._transmit(apdu)
        if (sw1, sw2) != (0x90, 0x00):
            raise CardError(f"update_word_sw={sw1:02X}{sw2:02X}")

    def verify(self, target: int, code4: bytes) -> None:
        if len(code4) != 4:
            raise CardError("verify_len")
        apdu = [0x00, INS_VERIFY, 0x00, target & 0xFF, 0x04] + list(code4)
        _, sw1, sw2 = self._transmit(apdu)
        if (sw1, sw2) != (0x90, 0x00):
            raise CardError(f"verify_sw={sw1:02X}{sw2:02X}")

    def get_issuer_sn(self) -> bytes:
        out = b""
        for a in range(ADDR_ISSUER_BASE, ADDR_ISSUER_BASE + 4):
            out += self.read_word(a)
        return out

    def uid_from_issuer(self) -> str:
        issuer = self.get_issuer_sn()
        raw = issuer + bytes.fromhex(self.atr_hex.replace(" ", ""))
        return _sha256(raw)[:16].hex()

    def read_app_record(self) -> Optional[AppRecord]:
        raw = b""
        try:
            for i in range(APP_WORD_LEN):
                raw += self.read_word(APP_WORD_BASE + i)
        except Exception:
            return None

        if raw[:4] != MAGIC:
            return None

        card_uid = raw[4:20].hex()
        user_hash8 = raw[20:28].hex()
        tpl_hash8 = raw[28:36].hex()
        return AppRecord(card_uid=card_uid, user_hash8=user_hash8, tpl_hash8=tpl_hash8)

    def _build_record(self, card_uid16: bytes, user_id: str, tpl_sha256_hex: str) -> bytes:
        user_hash8 = _sha256(user_id.encode("utf-8"))[:8]
        tpl_hash8 = bytes.fromhex(tpl_sha256_hex)[:8]
        tail = b"\x00\x00\x00\x00"
        return MAGIC + card_uid16 + user_hash8 + tpl_hash8 + tail

    def _try_unlock_user1(self) -> None:
        codes = _candidate_codes()
        targets = [VERIFY_TARGET_CSC1, VERIFY_TARGET_CSC0, VERIFY_TARGET_CSC2]
        last_err = None
        for t in targets:
            for c in codes:
                try:
                    self.verify(t, c)
                    return
                except Exception as e:
                    last_err = e
                    continue
        raise CardError(f"unlock_failed:{last_err}")

    def write_app_record(self, user_id: str, tpl_sha256_hex: str, card_uid16: Optional[bytes] = None) -> str:
        if card_uid16 is None:
            card_uid16 = os.urandom(16)

        rec = self._build_record(card_uid16, user_id, tpl_sha256_hex)
        words = _chunks4(rec)

        self._try_unlock_user1()
        for i, w in enumerate(words):
            self.update_word(APP_WORD_BASE + i, w)

        return card_uid16.hex()

    def provision_or_load_uid(self, user_id: str, tpl_sha256_hex: str) -> Tuple[str, bool]:
        r = self.read_app_record()
        expected_user_hash8 = _sha256(user_id.encode("utf-8"))[:8].hex()
        expected_tpl_hash8 = bytes.fromhex(tpl_sha256_hex)[:8].hex()

        if r is None:
            uid = self.write_app_record(user_id=user_id, tpl_sha256_hex=tpl_sha256_hex, card_uid16=None)
            return uid, True

        if (r.user_hash8.lower() == expected_user_hash8.lower()) and (r.tpl_hash8.lower() == expected_tpl_hash8.lower()):
            return r.card_uid, False

        uid16 = bytes.fromhex(r.card_uid)
        uid = self.write_app_record(user_id=user_id, tpl_sha256_hex=tpl_sha256_hex, card_uid16=uid16)
        return uid, True

    def get_uid(self) -> Optional[str]:
        r = self.read_app_record()
        return None if r is None else r.card_uid


def open_card(timeout_seconds: int = 10) -> CardSession:
    return CardSession(timeout_seconds=timeout_seconds)
