"""Locust 压测脚本（对应 docs/06-测试计划.md §4 性能测试）。

启动后端后运行：
  locust -f scripts/locustfile.py --host http://127.0.0.1:8000 \
         --headless -u 20 -r 5 -t 30s --only-summary

也可打开 Web UI（去掉 --headless）在 http://localhost:8089 观察。
"""
from __future__ import annotations

from locust import HttpUser, between, task

PRED_BODY = {
    "station_id": "1001A", "pollutant": "pm25",
    "steps": [1, 3, 6, 12, 24], "as_of": "2024-12-31T23:00:00",
}
QNET = "pollutant=pm25&from=2022-01-01&to=2022-06-30&surrogates=100"


class PlatformUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict(self):
        self.client.post("/api/v1/predictions", json=PRED_BODY, name="POST /predictions")

    @task(3)
    def grading(self):
        self.client.get("/api/v1/grading?station_id=1001A&pollutant=pm25", name="GET /grading")

    @task(2)
    def networks(self):
        self.client.get(f"/api/v1/networks?{QNET}", name="GET /networks")

    @task(2)
    def key_stations(self):
        self.client.get(f"/api/v1/key-stations?top=10&{QNET}", name="GET /key-stations")
