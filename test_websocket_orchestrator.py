"""
Mindfully AI - Automated Integration Test for main.py WebSocket Server
=====================================================================
File: test_websocket_orchestrator.py
"""

import sys
import json
import asyncio
import unittest
import websockets

from main import MindfullyOrchestrator


class TestWebSocketOrchestrator(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_orchestrator_payload_broadcast(self):
        port = 8766  # Use separate port to avoid collision
        orchestrator = MindfullyOrchestrator(
            host="localhost",
            port=port,
            mock=True,
            step_size=64,
            duration=8.0,  # Run for 8 seconds
        )

        # Run orchestrator server in a background asyncio task
        server_task = asyncio.create_task(orchestrator.run_server())

        # Allow server a brief moment to start listening
        await asyncio.sleep(1.0)

        received_payloads = []

        try:
            async with websockets.connect(f"ws://localhost:{port}") as ws:
                # 1. First message should be handshake
                handshake_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                handshake = json.loads(handshake_raw)
                print(f"\n[Test Client] Handshake received: {handshake}")
                self.assertIn("type", handshake)

                # 2. Wait for CRAVING_SCORE broadcast messages (stream fills 4s buffer then infers)
                print("[Test Client] Listening for CRAVING_SCORE broadcasts...")
                while len(received_payloads) < 2:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    msg = json.loads(raw_msg)
                    print(f"[Test Client] Message received: {msg}")
                    if msg.get("type") == "CRAVING_SCORE":
                        received_payloads.append(msg)

        finally:
            orchestrator.stop_event.set()
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

        # Assertions on received payloads
        self.assertGreaterEqual(len(received_payloads), 2)
        for p in received_payloads:
            self.assertEqual(p["type"], "CRAVING_SCORE")
            self.assertIn("score", p)
            self.assertIn("level", p)
            self.assertIsInstance(p["score"], float)
            self.assertIsInstance(p["level"], str)
            self.assertGreaterEqual(p["score"], 0.0)
            self.assertLessEqual(p["score"], 1.0)
            self.assertIn(p["level"], ["LOW", "MODERATE", "HIGH", "CRITICAL"])
            print(f"[Test Passed] Validated payload schema: {p}")


if __name__ == "__main__":
    unittest.main()

