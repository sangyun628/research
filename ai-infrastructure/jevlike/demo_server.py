"""Local-only playground backed by the cloned Jevlike implementation."""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import torch

from jevlike.data import ChoiceExample, synthetic_example
from jevlike.model import load_checkpoint, select_device
from jevlike.train import move


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / ".repos/jevlike/runs/research/synthetic.pt"
PAGE = Path(__file__).with_name("demo.html")
MAX_BODY = 64 * 1024


class Playground:
    def __init__(self, checkpoint: Path, device_name: str):
        self.device = select_device(device_name)
        self.model, self.collator, self.config = load_checkpoint(checkpoint, self.device)
        self.model.eval()
        self.lock = threading.Lock()
        self.tiny = self.config["encoder"] == "tiny"
        self.default_demo = checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve()
        self.info = {
            "model": "TinyScorer" if self.tiny else self.config["hf_model"],
            "encoder": self.config["encoder"],
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "checkpoint": checkpoint.name,
            "context_limit": self.config["context_tokens"],
            "option_limit": self.config["option_tokens"],
            "length_unit": "bytes" if self.tiny else "tokens",
            "default_demo": self.default_demo,
            "training_description": (
                "영어 색상·동물 배지 맞추기 · 학습 2,000개 · 8 epochs"
                if self.default_demo else "사용자 지정 체크포인트 · 학습 범위를 직접 확인하세요."
            ),
        }
        sample_path = DEFAULT_CHECKPOINT.parent / "synthetic/test.jsonl"
        self.examples = []
        if sample_path.exists():
            self.examples = [json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()]
        # Warm up once; requests reuse this loaded model.
        self.predict(self.example())

    def example(self):
        if self.examples:
            return random.choice(self.examples)
        row = synthetic_example(random.randrange(1_000_000))
        return {"context": row.context, "options": list(row.options), "label": row.label}

    def lengths(self, texts):
        if self.tiny:
            return [len(text.encode("utf-8")) for text in texts]
        return [len(ids) for ids in self.collator.tokenizer(texts, add_special_tokens=True)["input_ids"]]

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def predict(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("JSON 객체에 context와 options를 넣어 주세요.")
        context, options = payload.get("context"), payload.get("options")
        if not isinstance(context, str) or not context.strip():
            raise ValueError("문맥을 입력해 주세요.")
        if not isinstance(options, list) or not 2 <= len(options) <= 16:
            raise ValueError("선택지는 2~16개여야 합니다.")
        if any(not isinstance(option, str) or not option.strip() for option in options):
            raise ValueError("빈 선택지를 제거하거나 내용을 입력해 주세요.")
        if len(set(options)) != len(options):
            raise ValueError("중복된 선택지는 사용할 수 없습니다.")
        if len(context) > 8192 or any(len(option) > 2048 for option in options):
            raise ValueError("입력이 너무 깁니다. 문맥 8,192자, 선택지당 2,048자 이내로 입력해 주세요.")

        with self.lock, torch.inference_mode():
            lengths = self.lengths([context, *options])
            warnings = []
            limits = [self.info["context_limit"], *([self.info["option_limit"]] * len(options))]
            for index, (length, limit) in enumerate(zip(lengths, limits)):
                if length > limit:
                    name = "문맥" if index == 0 else f"선택지 {index}"
                    warnings.append(f"{name}: {length} {self.info['length_unit']} 중 앞 {limit}까지만 모델에 전달됩니다.")
            self.synchronize()
            started = time.perf_counter_ns()
            batch = move(self.collator([ChoiceExample(context, tuple(options), 0)]), self.device)
            probabilities = self.model(batch).softmax(-1)[0, :len(options)].cpu().tolist()
            self.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        winner = max(range(len(options)), key=probabilities.__getitem__)
        return {
            "prediction": {"index": winner, "option": options[winner], "probability": probabilities[winner]},
            "scores": [{"index": index, "option": option, "probability": probability}
                       for index, (option, probability) in enumerate(zip(options, probabilities))],
            "inference_ms": elapsed_ms,
            "lengths": {"context": lengths[0], "options": lengths[1:], "unit": self.info["length_unit"]},
            "warnings": warnings,
        }


def handler_for(playground, port):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{host}" for host in allowed_hosts}

    class Handler(BaseHTTPRequestHandler):
        server_version = "JevlikeLocalDemo/1.0"

        def send_body(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, payload):
            self.send_body(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def check_host(self):
            if self.headers.get("Host") not in allowed_hosts:
                self.send_json(403, {"error": "로컬 주소로 접속해 주세요."})
                return False
            return True

        def do_GET(self):
            if not self.check_host():
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.send_body(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self.send_json(200, playground.info)
            elif path == "/api/example":
                self.send_json(200, playground.example())
            elif path == "/favicon.ico":
                self.send_body(204, b"", "image/x-icon")
            else:
                self.send_json(404, {"error": "존재하지 않는 경로입니다."})

        def do_POST(self):
            if not self.check_host():
                return
            origin = self.headers.get("Origin")
            if origin and origin not in allowed_origins:
                self.send_json(403, {"error": "로컬 데모 화면에서 요청해 주세요."})
                return
            if urlsplit(self.path).path != "/api/predict":
                self.send_json(404, {"error": "존재하지 않는 경로입니다."})
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "Content-Type은 application/json이어야 합니다."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    self.send_json(413, {"error": "요청 본문은 64 KiB 이하여야 합니다."})
                    return
                payload = json.loads(self.rfile.read(length))
                self.send_json(200, playground.predict(payload))
            except (ValueError, UnicodeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception as error:
                self.log_error("Inference failed: %s", error)
                self.send_json(500, {"error": "모델 실행 중 오류가 발생했습니다. 서버 로그를 확인해 주세요."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. See local-demo.md for training commands.")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    playground = Playground(args.checkpoint, args.device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(playground, args.port))
    print(f"Jevlike ready: http://127.0.0.1:{args.port} | {playground.info['model']} | {playground.device}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
