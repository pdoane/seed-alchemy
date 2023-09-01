import { KeyboardEvent, PointerEvent, WheelEvent, useEffect, useRef } from "react";
import { subscribe } from "valtio";
import { subscribeKey } from "valtio/utils";
import { CanvasSceneRender } from "./CanvasSceneRender";
import { CanvasElementState, CanvasStrokeState } from "./schema";
import { stateCanvas, stateSession, stateSystem } from "./store";
import { Vec2 } from "./vec2";

export let canvasSceneRender: CanvasSceneRender | null = null;

export const CanvasScene = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null!);
  const requestRef = useRef<number>();
  const captureElementRef = useRef<CanvasElementState | null>(null);
  const panningRef = useRef<boolean>(false);
  const startMousePosRef = useRef<Vec2 | null>(null);
  const startElementPosRef = useRef<Vec2 | null>(null);

  function pointToScene(p: Vec2): Vec2 {
    return Vec2.create(
      p.x / stateCanvas.scale - stateCanvas.translate.x,
      p.y / stateCanvas.scale - stateCanvas.translate.y
    );
  }

  function eventToScene(e: PointerEvent | WheelEvent): Vec2 {
    const rect = e.currentTarget.getBoundingClientRect();
    return pointToScene(Vec2.create(e.clientX - rect.left, e.clientY - rect.top));
  }

  function contains(element: CanvasElementState, p: Vec2): boolean {
    return (
      p.x >= element.x && p.y >= element.y && p.x <= element.x + element.width && p.y <= element.y + element.height
    );
  }

  function closestElement(scenePos: Vec2): CanvasElementState | null {
    for (const element of stateCanvas.elements) {
      if (contains(element, scenePos)) {
        return element;
      }
    }
    return null;
  }

  function handlePointerDown(e: PointerEvent<HTMLCanvasElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    const scenePos = eventToScene(e);

    const element = closestElement(scenePos);

    startMousePosRef.current = scenePos;

    if (stateCanvas.tool === "select") {
      if (element) {
        stateCanvas.selectedId = element.id;
        captureElementRef.current = element;
        startElementPosRef.current = Vec2.create(element.x, element.y);
      } else {
        stateCanvas.selectedId = null;
        panningRef.current = true;
      }
    } else if (stateCanvas.tool === "brush" || stateCanvas.tool === "eraser") {
      const stroke = new CanvasStrokeState();
      stroke.tool = stateCanvas.tool;
      stroke.segments.push(scenePos.x, scenePos.y);
      stateCanvas.strokes.push(stroke);
    }
  }

  function handlePointerMove(e: PointerEvent<HTMLCanvasElement>) {
    const scenePos = eventToScene(e);

    if (captureElementRef.current) {
      e.preventDefault();

      const newPosX = startElementPosRef.current!.x + scenePos.x - startMousePosRef.current!.x;
      const newPosY = startElementPosRef.current!.y + scenePos.y - startMousePosRef.current!.y;

      const snappedPosX = Math.round(newPosX / 8) * 8;
      const snappedPosY = Math.round(newPosY / 8) * 8;

      captureElementRef.current.x = snappedPosX;
      captureElementRef.current.y = snappedPosY;
    } else if (panningRef.current) {
      e.preventDefault();

      stateCanvas.translate.x += scenePos.x - startMousePosRef.current!.x;
      stateCanvas.translate.y += scenePos.y - startMousePosRef.current!.y;
    } else if (stateCanvas.tool === "select") {
      const element = closestElement(scenePos);
      stateCanvas.hoveredId = element ? element.id : null;
    } else if (stateCanvas.tool === "brush" || stateCanvas.tool === "eraser") {
      if (startMousePosRef.current) {
        const stroke = stateCanvas.strokes[stateCanvas.strokes.length - 1];
        stroke.segments.push(scenePos.x, scenePos.y);
      }
    }

    if (stateCanvas.tool !== "select") {
      stateCanvas.cursorPos = eventToScene(e);
    }
  }

  function handlePointerUp(e: PointerEvent<HTMLCanvasElement>) {
    e.currentTarget.releasePointerCapture(e.pointerId);
    captureElementRef.current = null;
    panningRef.current = false;
    startMousePosRef.current = null;
    startElementPosRef.current = null;
  }

  function handleWheel(e: WheelEvent<HTMLCanvasElement>): void {
    const scaleRate = 1.005;
    const scaleDelta = Math.pow(e.deltaY > 0 ? scaleRate : 1 / scaleRate, Math.min(32, Math.abs(e.deltaY)));

    const oldPos = eventToScene(e);
    stateCanvas.scale *= scaleDelta;
    const newPos = eventToScene(e);
    stateCanvas.translate.x += newPos.x - oldPos.x;
    stateCanvas.translate.y += newPos.y - oldPos.y;

    if (stateCanvas.tool !== "select") {
      stateCanvas.cursorPos = eventToScene(e);
    }
  }

  function handleMouseLeave() {
    stateCanvas.cursorPos = null;
  }

  function handleKeyDown(e: KeyboardEvent<HTMLCanvasElement>): void {
    if (e.key === "Backspace") {
      if (stateCanvas.selectedId) {
        const index = stateCanvas.elements.findIndex((element) => element.id == stateCanvas.selectedId);
        if (index != -1) stateCanvas.elements.splice(index, 1);
      }
    }
  }

  function clearTextureCache() {
    canvasSceneRender?.clearTextureCache();
  }

  function requestRender() {
    if (requestRef.current === undefined) {
      requestRef.current = requestAnimationFrame(animate);
    }
  }

  function setPreviewUrl() {
    canvasSceneRender?.setPreviewUrl(stateSession.previewUrl);
  }

  function animate(_: number) {
    canvasSceneRender?.render();
    requestRef.current = undefined;
  }

  useEffect(() => {
    canvasSceneRender = new CanvasSceneRender(canvasRef.current, requestRender);
    const unsubscribe1 = subscribe(stateCanvas, requestRender);
    const unsubscribe2 = subscribeKey(stateSystem, "user", () => {
      clearTextureCache();
      requestRender();
    });
    const unsubscribe3 = subscribeKey(stateSession, "previewUrl", setPreviewUrl);

    requestRender();
    setPreviewUrl();

    return () => {
      unsubscribe3();
      unsubscribe2();
      unsubscribe1();
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
        requestRef.current = undefined;
      }
      canvasSceneRender?.cleanup();
    };
  }, []);

  return (
    <div className="relative w-full h-full flex-shrink overflow-hidden">
      <canvas
        ref={canvasRef}
        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
        tabIndex={0}
        width={2048}
        height={2048}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerMove={handlePointerMove}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
        onKeyDown={handleKeyDown}
      />
    </div>
  );
};
