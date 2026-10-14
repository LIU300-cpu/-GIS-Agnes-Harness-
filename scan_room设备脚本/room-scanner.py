"""Room scanner with CLIP-based keyframe selection.

Usage:
    cd E:\\agnes-harness\\packages\\mhs\\python
    uv run --with opencv-python --with pillow --with torch --with transformers \\
        python examples\\room-scanner.py ws://127.0.0.1:4180
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import asyncio
import io
import math
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel

from agnes_mhs import Device

W, H = 480, 360
JPEG_QUALITY = 75
SCAN_SECONDS = 30
N_KEYFRAMES = 9

dev = Device(
    "room-scanner-01", "camera", name="Room Scanner (phone)",
    state={"scanning": {"type": "boolean"}, "frames": {"type": "integer", "min": 0}},
)
dev.update(scanning=False, frames=0)

view = dev.source(
    "view", "image", "After scan_room() holds a 3x3 CLIP-selected panorama",
    hz=1, mime="image/jpeg", encoding="rgb",
    size=[1440, 1080], fov_deg=[60, 45],
    mount={"xyz": [0, 0, 1.5], "rpy": [0, 0, 0]},
)

print("[scanner] loading CLIP...", flush=True)
_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model.eval()
print("[scanner] CLIP ready", flush=True)


def open_cam():
    for idx in [1, 0, 2, 3]:
        c = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            ok, _ = c.read()
            if ok:
                print(f"[scanner] camera opened at index {idx}", flush=True)
                return c
            c.release()
    return None


def clip_embeds(frames_bgr):
    """Return L2-normalized CLIP image embeddings for a list of BGR frames."""
    pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    inputs = _clip_proc(images=pil_imgs, return_tensors="pt")
    with torch.no_grad():
        v = _clip_model.vision_model(pixel_values=inputs["pixel_values"])
        feat = _clip_model.visual_projection(v.pooler_output)
    return feat / feat.norm(dim=-1, keepdim=True)


def pick_diverse(embeds, k=9):
    """Greedy: pick frames whose embedding is farthest from already-picked ones."""
    n = embeds.shape[0]
    if n <= k:
        return list(range(n))
    picked = [0]
    while len(picked) < k:
        last = picked[-1]
        # distance from every unpicked frame to the closest picked frame
        best_i, best_min_dist = None, -1.0
        for i in range(n):
            if i in picked:
                continue
            min_d = min(float(1.0 - torch.dot(embeds[i], embeds[j])) for j in picked)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_i = i
        picked.append(best_i)
    return sorted(picked)


def stitch(frames, idxs):
    canvas = Image.new("RGB", (W * 3, H * 3), (20, 20, 30))
    draw = ImageDraw.Draw(canvas)
    for slot, i in enumerate(idxs[:9]):
        r, c = divmod(slot, 3)
        f = frames[i]
        canvas.paste(Image.fromarray(cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2RGB)), (c * W, r * H))
        draw.text((c * W + 5, r * H + 5), f"#{i}", fill=(255, 80, 80))
    return canvas


state = {"scanning": False, "frozen": False, "frames": [], "start": 0}
cap = None


async def stream_view():
    global cap
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (40, 40, 50)).save(buf, format="JPEG", quality=60)
    view.send({"w": 320, "h": 240}, buf.getvalue())
    print("[scanner] placeholder sent", flush=True)
    while True:
        if view.wants() or state["scanning"]:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = await asyncio.to_thread(open_cam)
                if cap is None:
                    await asyncio.sleep(2)
                    continue
            ok, frame = await asyncio.to_thread(cap.read)
            if ok:
                now = time.time()
                if state["scanning"]:
                    state["frames"].append(frame)
                    dev.update(scanning=True, frames=len(state["frames"]))
                    if now - state["start"] >= SCAN_SECONDS:
                        print(f"[scanner] got {len(state['frames'])} frames, running CLIP...", flush=True)
                        embeds = await asyncio.to_thread(clip_embeds, state["frames"])
                        idxs = await asyncio.to_thread(pick_diverse, embeds, N_KEYFRAMES)
                        print(f"[scanner] picked: {idxs}", flush=True)
                        grid = stitch(state["frames"], idxs)
                        buf = io.BytesIO()
                        grid.save(buf, format="JPEG", quality=JPEG_QUALITY)
                        view.send({"w": grid.width, "h": grid.height}, buf.getvalue())
                        state["scanning"] = False
                        state["frozen"] = True
                        state["frames"] = []
                        dev.update(scanning=False, frames=0)
                elif not state["frozen"] and view.wants():
                    small = cv2.resize(frame, (W, H))
                    buf = io.BytesIO()
                    Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG", quality=JPEG_QUALITY)
                    view.send({"w": W, "h": H}, buf.getvalue())
            else:
                if cap is not None:
                    cap.release()
                    cap = None
        await asyncio.sleep(0.1)


@dev.tool("Pan around the room for 30s; CLIP picks the 9 most distinctive frames and stitches them into a 3x3 panorama. Then read_device.", timeout=60)
async def scan_room(call):
    state["scanning"] = True
    state["frozen"] = False
    state["frames"] = []
    state["start"] = time.time()
    dev.update(scanning=True, frames=0)
    return f"scanning started - slowly pan the phone 360 degrees for {SCAN_SECONDS}s."


async def main(url):
    asyncio.create_task(stream_view())
    try:
        await dev.run(url)
    finally:
        if cap is not None:
            cap.release()


import sys
asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:4180"))
