import logging
from dataclasses import dataclass
from enum import Enum, unique
from typing import Final

from ._invocation import parse_invocation_stream

_LOGGER = logging.getLogger(__name__)

_BUTTON_EVENT_MIN: Final[int] = 29  # FunctionButtonEvent0
_BUTTON_EVENT_MAX: Final[int] = 36  # FunctionButtonEvent7
_INPUT_EVENT_MIN: Final[int] = 64  # FunctionNotifyInput0
_INPUT_EVENT_MAX: Final[int] = 71  # FunctionNotifyInput7

_TARGET_TYPE_BUTTON: Final[int] = 0x06
_TARGET_TYPE_INPUT: Final[int] = 0x12


@unique
class ButtonEventType(Enum):
    PRESS = 0x01
    RELEASE = 0x02
    HOLD = 0x09
    RELEASE_AFTER_HOLD = 0x0C
    UNKNOWN = 0xFFFF


@dataclass(frozen=True, repr=True)
class SwitchEvent:
    button_event_index: int  # 0-based index from protocol (opcode - base)
    button: int  # 1-based label = button_event_index + 1
    unit_id: int
    target_type: int  # 0x06 = button stream, 0x12 = input stream
    event: ButtonEventType
    flags: int
    extra_data: bytes


def parseSwitchEvents(
    data: bytes, packet_seq: int, raw_packet: bytes | None = None
) -> list[SwitchEvent]:
    """Parse decrypted type-7 packet payload as INVOCATION frames and emit switch events."""

    frames = parse_invocation_stream(data, logger=_LOGGER)
    events: list[SwitchEvent] = []

    for frame in frames:
        target_type = frame.target & 0xFF
        unit_id = frame.target >> 8

        if (
            target_type == _TARGET_TYPE_BUTTON
            and _BUTTON_EVENT_MIN <= frame.opcode <= _BUTTON_EVENT_MAX
        ):
            # Button stream: press/release encoded in bit 1 of origin low byte.
            # Confirmed on PTM215B captures: is_release = (origin & 0x02) != 0
            button_event_index = frame.opcode - _BUTTON_EVENT_MIN
            is_release = bool(frame.origin & 0x02)
            event = ButtonEventType.RELEASE if is_release else ButtonEventType.PRESS
            events.append(
                SwitchEvent(
                    button_event_index=button_event_index,
                    button=button_event_index + 1,
                    unit_id=unit_id,
                    target_type=target_type,
                    event=event,
                    flags=frame.flags,
                    extra_data=frame.payload,
                )
            )

        elif (
            target_type == _TARGET_TYPE_INPUT
            and _INPUT_EVENT_MIN <= frame.opcode <= _INPUT_EVENT_MAX
        ):
            # Input stream: payload[0] is the event type directly (matches ButtonEventType values).
            # Confirmed on PTM215B: 0x02=RELEASE, 0x09=HOLD, 0x0C=RELEASE_AFTER_HOLD
            if not frame.payload:
                _LOGGER.debug("Input stream frame with empty payload, skipping.")
                continue
            button_event_index = frame.opcode - _INPUT_EVENT_MIN
            try:
                event = ButtonEventType(frame.payload[0])
            except ValueError:
                _LOGGER.debug(
                    "Unknown input event code 0x%02x in input stream frame.",
                    frame.payload[0],
                )
                event = ButtonEventType.UNKNOWN
            events.append(
                SwitchEvent(
                    button_event_index=button_event_index,
                    button=button_event_index + 1,
                    unit_id=unit_id,
                    target_type=target_type,
                    event=event,
                    flags=frame.flags,
                    extra_data=frame.payload[1:],
                )
            )

        else:
            _LOGGER.debug(
                "Ignoring INVOCATION frame: opcode=0x%02x target_type=0x%02x.",
                frame.opcode,
                target_type,
            )

    if not events:
        _LOGGER.debug("No switch events found in packet #%s.", packet_seq)

    return events
