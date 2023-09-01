import argparse
import asyncio
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Optional
from uuid import UUID, uuid4

import janus
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic.json import ENCODERS_BY_TYPE
from websockets.exceptions import ConnectionClosedError

from . import config, messages, utils
from .models import (
    CancelRequest,
    ImageRequest,
    MoveRequest,
    PathRequest,
    ProcessRequest,
    PromptGenRequest,
)
from .session import Session

# Configuration
parser = argparse.ArgumentParser(description="Seed Alchemy Server")
parser.add_argument("--root", type=str, help="Root directory path")
args = parser.parse_args()
config.load_settings(args.root)
print(config.settings)

ENCODERS_BY_TYPE[bytes] = lambda bytes_obj: repr(bytes_obj)

# Globals
lock: asyncio.Lock = None
executor = ThreadPoolExecutor(max_workers=1)
sessions: dict[UUID, Session] = {}

# Fast API server
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


def background_task(session: Optional[Session], *args: Any):
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(executor, *args)
    if session:
        session.tasks.append(task)
    return task


@lru_cache(maxsize=1)
def controlnet_processor():
    from .control_net import ControlNetProcessor

    return ControlNetProcessor()


@lru_cache(maxsize=1)
def image_generator():
    from .image_generator import ImageGenerator

    return ImageGenerator(controlnet_processor())


@lru_cache(maxsize=1)
def preview_processor():
    from .image_generator import PreviewProcessor

    return PreviewProcessor(controlnet_processor())


@lru_cache(maxsize=1)
def prompt_generator():
    from .prompt_generator import PromptGenerator

    return PromptGenerator()


@app.on_event("startup")
async def startup_event():
    global lock
    lock = asyncio.Lock()


@app.post("/api/v1/cancel")
async def post_cancel(req: CancelRequest):
    session = sessions.get(req.session_id)
    if session:
        session.cancel = True
    return


@app.post("/api/v1/sd-generate")
async def post_sd_generate(req: ImageRequest, generator=Depends(image_generator)):
    async with lock:
        session = sessions.get(req.session_id) if req.session_id else None
        return await background_task(session, generator, req, session)


@app.post("/api/v1/controlnet-process")
async def post_control_net_process(req: ProcessRequest, processor=Depends(preview_processor)):
    async with lock:
        return await background_task(None, processor, req)


@app.post("/api/v1/prompt-generate")
async def post_prompt_generate(req: PromptGenRequest, generator=Depends(prompt_generator)):
    async with lock:
        print("prompt_generate", req)
        return await background_task(None, generator, req)


@app.post("/api/v1/image-interrogate")
async def post_image_interrogate():
    pass


@app.post("/api/v1/image-delete")
async def post_image_delete(req: PathRequest):
    image_full_path = config.get_image_path(req.user, req.path)
    if os.path.exists(image_full_path):
        import send2trash

        send2trash.send2trash(image_full_path)

    thumbnail_full_path = config.get_thumbnail_path(req.user, req.path)
    try:
        os.remove(thumbnail_full_path)
    except FileNotFoundError:
        pass
    return


@app.post("/api/v1/image-move")
async def post_image_move(req: MoveRequest):
    src_full_path = config.get_image_path(req.user, req.src_path)
    if not os.path.exists(src_full_path):
        return

    output_path = config.generate_output_path(req.user, req.dst_collection)
    dst_full_path = config.get_image_path(req.user, output_path)

    shutil.move(src_full_path, dst_full_path)
    return output_path


@app.post("/api/v1/reveal")
async def post_reveal(req: PathRequest) -> None:
    full_path = config.get_image_path(req.user, req.path)
    if os.path.exists(full_path):
        if sys.platform == "darwin":
            from AppKit import NSURL, NSWorkspace

            url = NSURL.fileURLWithPath_(full_path)
            NSWorkspace.sharedWorkspace().activateFileViewerSelectingURLs_([url])
        elif sys.platform == "win32":
            import subprocess

            subprocess.run(["explorer", "/select,", full_path])


@app.get("/api/v1/users")
async def get_users():
    return config.settings.users


@app.get("/api/v1/models")
async def get_models():
    return sorted([(info.type, info.base, key) for key, info in config.models.items()])


@app.get("/api/v1/schedulers")
async def get_schedulers():
    from . import scheduler_registry

    return sorted(scheduler_registry.DICT.keys())


@app.get("/api/v1/control_net_processors")
async def get_control_net_processors():
    from . import control_net_registry

    return sorted(control_net_registry.processors.keys())


@app.get("/api/v1/settings/{user}")
async def get_settings(user: str):
    full_path = config.get_settings_path(user)

    if os.path.exists(full_path):
        return FileResponse(full_path)
    else:
        return {}


@app.put("/api/v1/settings/{user}")
async def put_settings(user: str, request: Request):
    body = await request.body()

    full_path = config.get_settings_path(user)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w") as file:
        file.write(body.decode())
    return


@app.get("/api/v1/collections/{user}")
async def get_collections(user: str):
    full_path = config.get_images_path(user)

    list = []
    if os.path.exists(full_path):
        list = sorted(
            [
                entry
                for entry in os.listdir(full_path)
                if entry[0] != "." and os.path.isdir(os.path.join(full_path, entry))
            ]
        )
    if not list:
        list = ["outputs"]

    return list


@app.get("/api/v1/images/{user}/{collection}")
async def get_images(user: str, collection: str):
    full_path = config.get_image_path(user, collection)
    if not os.path.exists(full_path):
        return []

    list = sorted(
        [
            os.path.join(collection, file)
            for file in os.listdir(full_path)
            if file.lower().endswith((".webp", ".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]
    )
    list.reverse()
    return list


@app.get("/api/v1/metadata/{user}/{path:path}")
async def get_metadata(user: str, path: str):
    full_path = config.get_image_path(user, path)
    if not os.path.exists(full_path):
        return {}

    with Image.open(full_path) as image:
        return image.info


@app.post("/api/v1/upload")
async def upload_image(image: UploadFile = File(...), user: str = Form(...), collection: str = Form(...)):
    output_path = config.generate_output_path(user, collection)
    full_path = config.get_image_path(user, output_path)
    with open(full_path, "wb") as dst:
        shutil.copyfileobj(image.file, dst)

    return utils.normalize_path(output_path)


@app.get("/images/{user}/{path:path}")
async def get_image(user: str, path: str):
    full_path = config.get_image_path(user, path)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404)

    response = FileResponse(full_path)
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


@app.get("/thumbnails/{user}/{path:path}")
async def get_thumbnail(user: str, path: str):
    thumbnail_full_path = config.get_thumbnail_path(user, path)
    if not os.path.exists(thumbnail_full_path) or not os.path.isfile(thumbnail_full_path):
        image_full_path = config.get_image_path(user, path)
        if not os.path.exists(image_full_path) or not os.path.isfile(image_full_path):
            raise HTTPException(status_code=404)

        with Image.open(image_full_path) as image:
            thumbnail = utils.create_thumbnail(image, 256)

            os.makedirs(os.path.dirname(thumbnail_full_path), exist_ok=True)
            thumbnail.save(thumbnail_full_path, bitmap_format="webp")

    response = FileResponse(thumbnail_full_path)
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


async def websocket_reader(websocket: WebSocket):
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("Websocket disconnected")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    reader_task = asyncio.create_task(websocket_reader(websocket))

    session_id = uuid4()
    queue = janus.Queue()
    queue_task = None
    session = Session(queue, False, [])
    sessions[session_id] = session

    try:
        await websocket.send_bytes(messages.build_session_id(session_id))

        while not reader_task.done():
            queue_task = asyncio.create_task(queue.async_q.get())
            done, _ = await asyncio.wait([queue_task, reader_task], return_when=asyncio.FIRST_COMPLETED)
            if queue_task in done:
                message = await queue_task
                await websocket.send_bytes(message)

    except (WebSocketDisconnect, ConnectionClosedError):
        print("Websocket disconnected")

    session.cancel = True
    for task in session.tasks:
        await task
    if queue_task:
        queue_task.cancel()
    queue.close()
    await queue.wait_closed()
    sessions.pop(session_id)


if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
