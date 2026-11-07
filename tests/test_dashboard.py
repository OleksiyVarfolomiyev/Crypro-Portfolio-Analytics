import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd

from desk.dashboard import json_safe, make_handler, table


class PayloadTests(unittest.TestCase):
    def test_json_safe_turns_nan_and_numpy_values_into_plain_json(self):
        value = {"a": np.float64("nan"), "b": np.int64(3), "c": [float("inf"), 1.5],
                 "d": pd.Timestamp("2026-11-07", tz="UTC"), "e": pd.NaT, "f": np.bool_(True)}
        cleaned = json_safe(value)
        self.assertEqual(cleaned, {"a": None, "b": 3, "c": [None, 1.5], "d": "2026-11-07T00:00:00+00:00",
                                   "e": None, "f": True})
        json.dumps(cleaned, allow_nan=False)

    def test_table_splits_the_total_row_and_keeps_formats(self):
        frame = pd.DataFrame({"Account": ["A", "B", "Total"], "Value": [1.0, 2.0, 3.0]})
        spec = table(frame, {"Value": "usd"}, total_last=True)
        self.assertEqual([column["fmt"] for column in spec["columns"]], ["text", "usd"])
        self.assertEqual(len(spec["rows"]), 2)
        self.assertEqual(spec["total"]["Value"], 3.0)
        self.assertEqual(table(pd.DataFrame(), {})["rows"], [])


class _StubDashboard:
    def __init__(self):
        self.wake = threading.Event()

    def state_json(self):
        return b'{"status": {}, "data": null}'


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.dashboard = _StubDashboard()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), None)
        self.port = self.server.server_address[1]
        self.server.RequestHandlerClass = make_handler(self.dashboard, self.port)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, host=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.putrequest(method, path, skip_host=True)
        connection.putheader("Host", host or f"127.0.0.1:{self.port}")
        for key, value in (headers or {}).items():
            connection.putheader(key, value)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, response.read()

    def test_serves_state_and_page_to_localhost(self):
        status, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"], None)
        status, body = self.request("GET", "/", host=f"localhost:{self.port}")
        self.assertEqual(status, 200)
        self.assertIn(b"Trading desk", body)

    def test_rejects_other_host_names(self):
        status, _ = self.request("GET", "/api/state", host=f"evil.example:{self.port}")
        self.assertEqual(status, 403)

    def test_refresh_wakes_the_loop_and_checks_origin(self):
        status, _ = self.request("POST", "/api/refresh", headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertFalse(self.dashboard.wake.is_set())
        status, _ = self.request("POST", "/api/refresh", headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)
        self.assertTrue(self.dashboard.wake.is_set())


if __name__ == "__main__":
    unittest.main()
