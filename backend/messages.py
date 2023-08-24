import enum
import struct
from uuid import UUID
from typing import Optional


class Type(enum.IntEnum):
    SESSION_ID = 1
    PROGRESS = 2
    IMAGE = 3


def build_message(message_type: Type, data: bytes):
    if data:
        header = struct.pack(">ii", message_type, len(data))
        return header + data
    else:
        return struct.pack(">ii", message_type, 0)


def build_session_id(session_id: UUID):
    return build_message(Type.SESSION_ID, struct.pack(">16s", session_id.bytes))


def build_progress(generator_id: Optional[UUID], amount: int):
    uuid = generator_id or UUID(int=0)
    return build_message(Type.PROGRESS, struct.pack(">16si", uuid.bytes, amount))


def build_image(generator_id: UUID, image_data: bytes):
    uuid = generator_id or UUID(int=0)
    return build_message(
        Type.IMAGE,
        struct.pack(f">16s{len(image_data)}s", uuid.bytes, image_data),
    )
