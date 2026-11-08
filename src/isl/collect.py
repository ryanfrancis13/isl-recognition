import argparse
import time
import cv2
import numpy as np
from isl import core

STREAM_SEC = 8.0
IDLE_SEC = 20.0
REP_SEC = 2.0
REPS = 8
PREP_SEC = 2.0


def draw_hud(view, title, sub, color=(0, 255, 0), progress=0.0):
    w = view.shape[1]
    cv2.rectangle(view, (0, 0), (w, 76), (20, 20, 20), -1)
    cv2.putText(view, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(view, sub, (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    if progress > 0:
        cv2.rectangle(view, (0, 72), (int(w * min(progress, 1.0)), 76), color, -1)


def capture_span(cap, holistic, duration, title, sub, color):
    frames = []
    t0 = time.perf_counter()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            return None, "quit"
        feat, res = core.process_frame(holistic, frame)
        core.draw_skeleton(frame, res)
        view = cv2.flip(frame, 1)

        elapsed = time.perf_counter() - t0
        if elapsed >= duration:
            break
        frames.append(feat)
        draw_hud(view, title, f"{sub} ({duration - elapsed:.1f}s)", color, elapsed / duration)
        cv2.imshow("ISL Collect", view)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("r")):
            return None, ("quit" if k == ord("q") else "redo")
    return np.array(frames, dtype=np.float32), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sign", type=str)
    args = ap.parse_args()

    sdir = core.RAW_DIR / f"session_{args.session}"
    queue = [args.sign] if args.sign else (core.PILOT_SIGNS if args.pilot else list(core.SIGNS))
    cap = cv2.VideoCapture(0)
    idx = 0

    with core.make_holistic() as holistic:
        while idx < len(queue) and cap.isOpened():
            sign = queue[idx]
            mode = core.SIGNS[sign]
            out_dir = sdir / sign
            target = 1 if mode == "stream" else REPS

            if not args.sign and out_dir.exists() and len(list(out_dir.glob("*.npy"))) >= target:
                idx += 1
                continue

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    return
                _, res = core.process_frame(holistic, frame)
                core.draw_skeleton(frame, res)
                view = cv2.flip(frame, 1)
                draw_hud(
                    view,
                    f"[{idx + 1}/{len(queue)}] {sign} ({mode})",
                    "SPACE: start | N: skip | B: back | Q: quit",
                    (0, 255, 255),
                )
                cv2.imshow("ISL Collect", view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(" "):
                    break
                if k == ord("n") or (k == ord("b") and idx > 0):
                    idx += 1 if k == ord("n") else -1
                    break
                if k == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            if idx >= len(queue) or queue[idx] != sign:
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            for rep in range(target):
                prep = PREP_SEC if rep == 0 else 1.5
                _, st = capture_span(cap, holistic, prep, f"Prep: {sign} ({rep + 1}/{target})", "R: redo | Q: quit", (0, 200, 255))
                if st != "ok":
                    if st == "quit":
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    break

                dur = (IDLE_SEC if sign == "IDLE" else STREAM_SEC) if mode == "stream" else REP_SEC
                clip, st = capture_span(cap, holistic, dur, f"REC: {sign} ({rep + 1}/{target})", "R: redo | Q: quit", (0, 0, 255))
                if st != "ok" or len(clip) < 10:
                    if st == "quit":
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    break

                np.save(out_dir / f"{mode}_{rep:02d}_{int(time.time())}.npy", clip)
                print(f"Saved {sign} ({rep + 1}/{target}) -> {len(clip)} frames")
            else:
                idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
