"""
Mindfully AI - Core Orchestrator & Real-Time WebSocket Server
============================================================
File: main.py
Reference: project_architecture.md

Objective:
----------
Orchestrates the real-time EEG ingestion and edge inference pipeline. Runs an
asynchronous WebSocket server (via `websockets`) listening on ws://localhost:8765.
Ingests live EEG data via LSL using `run_eeg_receiver` from `/ml_pipeline`, buffers
into a 4-second sliding window ring buffer (at 256 Hz), executes quantized inference
via `mindfully_edge_v1.tflite`, and broadcasts real-time JSON payloads:
    {'type': 'CRAVING_SCORE', 'score': float, 'level': string}
to all connected WebSocket clients (such as the Mindfully React Native mobile app).
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import threading
from datetime import datetime
from typing import Set, Optional, Dict, Any

# Ensure workspace root and ml_pipeline are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
ML_PIPELINE_DIR = os.path.join(BASE_DIR, "ml_pipeline")
if ML_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, ML_PIPELINE_DIR)

import websockets
from ml_pipeline.eeg_receiver import run_eeg_receiver

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MindfullyOrchestrator")


class MindfullyOrchestrator:
    """
    Coordinates LSL EEG reception, TFLite inference in a worker thread, and
    broadcasting to async WebSocket clients.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        model_path: Optional[str] = None,
        stream_name: Optional[str] = None,
        stream_type: str = "EEG",
        window_size: int = 1024,
        step_size: int = 64,
        timeout: float = 10.0,
        mock: bool = False,
        threshold: float = 0.70,
        duration: Optional[float] = None,
    ):
        self.host = host
        self.port = port
        self.model_path = model_path or os.path.join(
            BASE_DIR, "edge_models", "mindfully_edge_v1.tflite"
        )
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.window_size = window_size
        self.step_size = step_size
        self.timeout = timeout
        self.mock = mock
        self.threshold = threshold
        self.duration = duration

        # Connected WebSocket clients
        self.connected_clients: Set[Any] = set()

        # Inter-thread synchronization and async messaging
        self.stop_event = threading.Event()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.broadcast_queue: Optional[asyncio.Queue] = None
        self.receiver_thread: Optional[threading.Thread] = None

        # Statistics
        self.total_broadcasts = 0
        self.latest_payload: Optional[Dict[str, Any]] = None

    def on_prediction_callback(self, data: Dict[str, Any]):
        """
        Thread-safe callback triggered by run_eeg_receiver upon every inference.
        Dispatches payload to the asyncio broadcast queue.
        """
        substance_score = float(data.get("score", 0.0))
        level = str(data.get("level", "LOW"))

        # Build user-requested JSON payload schema
        payload = {
            "type": "CRAVING_SCORE",
            "score": round(substance_score, 4),
            "level": level,
        }
        self.latest_payload = payload

        if self.loop is not None and self.broadcast_queue is not None:
            self.loop.call_soon_threadsafe(self.broadcast_queue.put_nowait, payload)

    async def client_handler(self, websocket: Any):
        """
        Handles individual WebSocket client connections.
        Registers the client, sends an initial handshake, and listens for client disconnects.
        """
        remote_address = getattr(websocket, "remote_address", "client")
        self.connected_clients.add(websocket)
        logger.info(f"WebSocket client connected from: {remote_address} (Total clients: {len(self.connected_clients)})")

        # Send initial handshake / status message
        handshake = {
            "type": "CONNECTION_ESTABLISHED",
            "message": "Connected to Mindfully EEG Orchestrator",
            "ws_url": f"ws://{self.host}:{self.port}",
            "model": "mindfully_edge_v1.tflite",
        }
        try:
            await websocket.send(json.dumps(handshake))
            # If an inference has already been computed, send it immediately
            if self.latest_payload is not None:
                await websocket.send(json.dumps(self.latest_payload))
        except Exception as e:
            logger.warning(f"Error sending handshake to {remote_address}: {e}")

        try:
            # Keep connection open until client closes it or disconnects
            async for message in websocket:
                # Optionally process incoming messages from mobile app (e.g. ping/ack)
                try:
                    data = json.loads(message)
                    if data.get("type") == "PING":
                        await websocket.send(json.dumps({"type": "PONG", "timestamp": time.time()}))
                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected: {remote_address} (Remaining clients: {len(self.connected_clients)})")

    async def broadcast_worker(self):
        """
        Asynchronously pulls prediction payloads from the queue and broadcasts
        to all connected WebSocket clients using websockets.broadcast.
        """
        assert self.broadcast_queue is not None
        while not self.stop_event.is_set():
            try:
                payload = await self.broadcast_queue.get()
                payload_str = json.dumps(payload)

                if self.connected_clients:
                    websockets.broadcast(self.connected_clients, payload_str)
                    self.total_broadcasts += 1
                    logger.info(
                        f"[WS Broadcast] {payload['type']} -> score: {payload['score']:.4f} "
                        f"({payload['level']}) to {len(self.connected_clients)} client(s)"
                    )

                self.broadcast_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as err:
                logger.error(f"Error in broadcast worker: {err}")

    def _start_eeg_receiver_thread(self):
        """
        Runs the blocking LSL EEG ingestion and edge inference loop in a background thread.
        """
        logger.info(f"Starting EEG Receiver thread (Mock={self.mock}, Model={os.path.basename(self.model_path)})...")
        try:
            run_eeg_receiver(
                model_path=self.model_path,
                stream_name=self.stream_name,
                stream_type=self.stream_type,
                window_size=self.window_size,
                step_size=self.step_size,
                timeout=self.timeout,
                duration=self.duration,
                mock=self.mock,
                jitai_threshold=self.threshold,
                on_prediction=self.on_prediction_callback,
                stop_event=self.stop_event,
            )
        except Exception as e:
            logger.error(f"EEG Receiver encountered an error: {e}", exc_info=True)
        finally:
            logger.info("EEG Receiver worker thread exited.")

    async def run_server(self):
        """
        Main asynchronous server entry point: starts WebSocket server and broadcast worker.
        """
        self.loop = asyncio.get_running_loop()
        self.broadcast_queue = asyncio.Queue()

        # Start EEG receiver in background worker thread
        self.receiver_thread = threading.Thread(
            target=self._start_eeg_receiver_thread,
            daemon=True,
            name="MindfullyEEGReceiverThread",
        )
        self.receiver_thread.start()

        # Start async broadcast dispatcher task
        broadcast_task = asyncio.create_task(self.broadcast_worker(), name="BroadcastWorker")

        logger.info("=" * 75)
        logger.info(" MINDFULLY AI - WEBSOCKET ORCHESTRATOR ACTIVE")
        logger.info(f" Listening on : ws://{self.host}:{self.port}")
        logger.info(f" Model Target : {self.model_path}")
        logger.info("=" * 75)

        # Start the websockets server
        async with websockets.serve(self.client_handler, self.host, self.port):
            try:
                # If a duration was specified, wait for it; otherwise run indefinitely
                if self.duration:
                    await asyncio.sleep(self.duration)
                else:
                    await asyncio.Future()  # run forever
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.info("Shutdown signal received.")
            finally:
                logger.info("Stopping orchestrator and shutting down connections...")
                self.stop_event.set()
                broadcast_task.cancel()
                try:
                    await broadcast_task
                except asyncio.CancelledError:
                    pass

                # Close connected clients
                if self.connected_clients:
                    close_tasks = [client.close() for client in list(self.connected_clients)]
                    await asyncio.gather(*close_tasks, return_exceptions=True)

                if self.receiver_thread and self.receiver_thread.is_alive():
                    self.receiver_thread.join(timeout=2.0)

                logger.info("Orchestrator shutdown complete.")


def parse_arguments():
    default_model = os.path.join(BASE_DIR, "edge_models", "mindfully_edge_v1.tflite")

    parser = argparse.ArgumentParser(
        description="Mindfully AI - WebSocket EEG Receiver Orchestrator"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="WebSocket server host binding (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=default_model,
        help=f"Path to .tflite edge model file (default: {default_model})",
    )
    parser.add_argument(
        "--stream-name",
        type=str,
        default=None,
        help="Optional LSL stream name to match.",
    )
    parser.add_argument(
        "--stream-type",
        type=str,
        default="EEG",
        help="LSL stream type property (default: EEG).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1024,
        help="Ring buffer window size in samples (default: 1024 = 4.0s at 256 Hz).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=64,
        help="Inference hop size in samples (default: 64 = 0.25s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="LSL stream discovery timeout in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Substance urge JITAI trigger threshold (default: 0.70).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration limit in seconds for automated testing.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Launch an internal simulated 256 Hz EEG stream for testing.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    orchestrator = MindfullyOrchestrator(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        stream_name=args.stream_name,
        stream_type=args.stream_type,
        window_size=args.window_size,
        step_size=args.step_size,
        timeout=args.timeout,
        mock=args.mock,
        threshold=args.threshold,
        duration=args.duration,
    )

    try:
        asyncio.run(orchestrator.run_server())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user (Ctrl+C). Exiting.")


if __name__ == "__main__":
    main()

