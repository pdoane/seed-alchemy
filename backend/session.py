from asyncio import Future
from dataclasses import dataclass

import janus


@dataclass
class Session:
    queue: janus.Queue
    cancel: bool
    tasks: list[Future]


class CancelException(Exception):
    pass
