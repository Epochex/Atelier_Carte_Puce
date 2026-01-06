from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class CardInfo:
    card_id: str  # 这里先用 ATR 字符串作为 card_id（稳定且易演示）


def try_read_card_atr(timeout_seconds: int = 30) -> Optional[CardInfo]:
    """
    尝试通过 PC/SC 读取插入卡的 ATR，返回 card_id=ATR_HEX。
    若未安装 pyscard 或读卡失败，返回 None。
    """
    try:
        from smartcard.System import readers
        from smartcard.CardRequest import CardRequest
        from smartcard.CardType import AnyCardType
    except Exception:
        return None

    rlist = readers()
    if not rlist:
        return None

    try:
        cr = CardRequest(timeout=timeout_seconds, cardType=AnyCardType())
        card_service = cr.waitforcard()
        conn = card_service.connection
        conn.connect()
        atr = conn.getATR()
        atr_hex = " ".join(f"{b:02X}" for b in atr)
        return CardInfo(card_id=atr_hex)
    except Exception:
        return None


def get_card_id(simulate_card: Optional[str], timeout_seconds: int = 30) -> CardInfo:
    """
    优先 simulate；否则尝试读 ATR；仍失败则抛异常。
    """
    if simulate_card:
        return CardInfo(card_id=simulate_card.strip())

    info = try_read_card_atr(timeout_seconds=timeout_seconds)
    if info is None:
        raise RuntimeError(
            "无法读取智能卡（pyscard 未安装或 PC/SC 读取失败）。"
            "你可以使用 --simulate-card <CARD_ID> 继续跑通流程。"
        )
    return info
