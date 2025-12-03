from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np
import cv2

from core.event_bus import EventBus

try:
    import tkinter as tk
except ImportError:
    tk = None


class StatusOverlay:
    """
    Always-on-top status pill showing current mode/intent.
    Runs in its own thread; non-fatal if Tk is unavailable.
    """

    def __init__(self, bus: EventBus, poll_ms: int = 120, window_title: str = "NovaControl Status") -> None:
        self.bus = bus
        self.poll_ms = poll_ms
        self.window_title = window_title
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._root: Optional[tk.Tk] = None
        self._label: Optional[tk.Label] = None
        self._use_cv = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        target = self._run_tk if tk is not None else self._run_cv
        if tk is None:
            self._use_cv = True
            logging.warning("Tkinter not available; falling back to OpenCV status overlay.")
        logging.info("Starting status overlay (%s)", "cv2" if self._use_cv else "tk")
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
        if self._use_cv:
            try:
                cv2.destroyWindow(self.window_title)
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # Tkinter overlay ------------------------------------------------------
    def _run_tk(self) -> None:
        if tk is None:
            return
        self._root = tk.Tk()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        try:
            self._root.wm_attributes("-toolwindow", True)
        except Exception:
            pass
        try:
            self._root.attributes("-alpha", 0.78)
        except Exception:
            pass
        self._root.configure(bg="#000000")

        self._label = tk.Label(
            self._root,
            text="SAFE",
            fg="#FFFFFF",
            bg="#000000",
            font=("Segoe UI", 11, "bold"),
            padx=14,
            pady=8,
        )
        self._label.pack()
        self._place_top_center()
        self._tick()
        self._root.mainloop()

    def _tick(self) -> None:
        if self._stop.is_set() or not self._root:
            return
        text = self._status_text()
        if self._label:
            self._label.config(text=text)
        self._place_top_center()
        self._root.after(self.poll_ms, self._tick)

    def _status_text(self) -> str:
        mode = "SAFE"
        intent = None
        latest_mode = self.bus.latest("mode")
        if latest_mode:
            mode = latest_mode.name
        latest_intent = self.bus.latest("intent")
        if latest_intent:
            intent = latest_intent.name

        if intent:
            return f"{mode} • {intent.replace('_', ' ').title()}"
        return mode

    def _place_top_center(self) -> None:
        if not self._root:
            return
        try:
            self._root.update_idletasks()
            w = self._root.winfo_width()
            h = self._root.winfo_height()
            screen_w = self._root.winfo_screenwidth()
            x = max(0, (screen_w - w) // 2)
            y = 10
            self._root.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    # OpenCV overlay fallback ----------------------------------------------
    def _run_cv(self) -> None:
        self._use_cv = True
        cv2.namedWindow(self.window_title, cv2.WINDOW_AUTOSIZE)
        try:
            cv2.setWindowProperty(self.window_title, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        while not self._stop.is_set():
            img = self._render_cv_image()
            cv2.imshow(self.window_title, img)
            self._center_cv_window(img)
            cv2.waitKey(1)
            time.sleep(self.poll_ms / 1000.0)

        try:
            cv2.destroyWindow(self.window_title)
        except Exception:
            pass

    def _render_cv_image(self) -> np.ndarray:
        text = self._status_text()
        padding = 12
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.6, 2)
        w = text_w + padding * 2
        h = text_h + padding * 2
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.putText(img, text, (padding, h - padding + 2), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def _center_cv_window(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        screen_w, screen_h = self._screen_size()
        x = max(0, (screen_w - w) // 2)
        y = 10
        try:
            cv2.moveWindow(self.window_title, x, y)
        except Exception:
            pass
        self._ensure_topmost(x, y)

    def _screen_size(self) -> tuple[int, int]:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            if tk:
                try:
                    root = tk.Tk()
                    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
                    root.destroy()
                    return w, h
                except Exception:
                    pass
        return (1920, 1080)

    def _ensure_topmost(self, x: int, y: int) -> None:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, self.window_title)
            if hwnd:
                SWP_NOSIZE = 0x0001
                SWP_NOACTIVATE = 0x0010
                HWND_TOPMOST = -1
                user32.SetWindowPos(hwnd, HWND_TOPMOST, x, y, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE)
                user32.ShowWindow(hwnd, 5)
        except Exception:
            pass
