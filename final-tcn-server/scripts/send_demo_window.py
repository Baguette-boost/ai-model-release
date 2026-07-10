"""Send a 50-sample demo IMU window to the final TCN server."""

from __future__ import annotations

import json
import urllib.request


URL = "http://127.0.0.1:8010/predict-window"


def main() -> None:
    samples = []
    for index in range(50):
        samples.append(
            {
                "device": "esp32-1",
                "timestamp": f"2026-07-10T10:00:{index:02d}",
                "t_ms": index * 40,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "ax": 0.02,
                "ay": -0.86,
                "az": -0.06,
                "wx": -2.0,
                "wy": 1.1,
                "wz": -0.4,
                "lat": 36.621853,
                "lng": 127.426337,
            }
        )
    payload = {
        "device": "esp32-1",
        "personId": "demo-user",
        "forward_to_backend": False,
        "samples": samples,
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
