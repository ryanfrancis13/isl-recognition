from isl import core  # must import first to set TF log env vars

from collections import deque
import json
import time
import cv2
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")


def main():
    model = tf.keras.models.load_model(core.ARTIFACTS_DIR / "model.keras")
    signs = json.loads((core.ARTIFACTS_DIR / "labels.json").read_text(encoding="utf-8"))

    buf = deque(maxlen=20)
    ema = None
    step = 0
    label, conf = "Waiting...", 0.0
    prev_t = time.perf_counter()
    fps = 0.0

    cap = cv2.VideoCapture(0)
    with core.make_holistic() as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            feat, res = core.process_frame(holistic, frame)
            core.draw_skeleton(frame, res)
            view = cv2.flip(frame, 1)
            buf.append(feat)
            step += 1

            if len(buf) == buf.maxlen and step % 2 == 0:
                arr = np.array(buf, dtype=np.float32)
                if (arr[:, 132:134].sum(axis=1) > 0).mean() < 0.30:
                    label, conf, ema = "IDLE", 1.0, None
                else:
                    win = core.to_window(arr)[None, ...]
                    probs = model(win, training=False).numpy()[0]
                    ema = probs if ema is None else 0.55 * probs + 0.45 * ema
                    idx = int(np.argmax(ema))
                    conf = float(ema[idx])
                    if conf >= 0.75:
                        label = signs[idx]

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
            prev_t = now

            w = view.shape[1]
            cv2.rectangle(view, (0, 0), (w, 56), (20, 20, 20), -1)
            cv2.putText(
                view,
                f"Sign: {label} ({conf:.2f}) | FPS: {fps:.1f}",
                (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if label not in ("IDLE", "Uncertain", "Waiting...") else (180, 180, 180),
                2,
            )
            cv2.imshow("ISL Live", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
