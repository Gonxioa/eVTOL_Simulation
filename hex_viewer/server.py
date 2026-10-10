"""A localhost-only, read-only replay server with no third-party dependencies."""
import argparse
import gzip
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from .data import ReplayDataset

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent/'static'


def make_handler(dataset):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == '/api/catalog':
                    self.send_json(dataset.catalog())
                elif parsed.path == '/api/record':
                    query = parse_qs(parsed.query)
                    group = int(query.get('group', ['0'])[0]); label = int(query.get('label', ['-1'])[0])
                    self.send_json(dataset.record(group, label))
                elif parsed.path in ['/', '/viewer.js', '/core.js', '/viewer.css']:
                    name = 'index.html' if parsed.path == '/' else parsed.path[1:]
                    content_type = 'text/html' if name.endswith('.html') else 'text/css' if name.endswith('.css') else 'text/javascript'
                    self.send_body((STATIC/name).read_bytes(), content_type+'; charset=utf-8')
                elif parsed.path == '/favicon.ico':
                    self.send_body(b'', 'image/x-icon', 204)
                else:
                    self.send_json({'error': '不存在的地址'}, 404)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.send_json({'error': str(exc)}, 400)

        def send_json(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
            self.send_body(data, 'application/json; charset=utf-8', status)

        def send_body(self, data, content_type, status=200):
            compressed = 'gzip' in self.headers.get('Accept-Encoding', '') and len(data) > 4096
            if compressed: data = gzip.compress(data)
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if compressed: self.send_header('Content-Encoding', 'gzip')
            self.end_headers(); self.wfile.write(data)

        def log_message(self, format, *args):
            if len(args) < 2 or str(args[1]) not in ['200', '204']:
                super().log_message(format, *args)
    return Handler


def main():
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='回放现有八组六旋翼任务；不仿真、不训练、不修改CSV。')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'output_hex_next_20261009/batch2/dataset_v5')
    parser.add_argument('--predictions-dir', type=Path, default=None)
    parser.add_argument('--group', type=int, default=6)
    parser.add_argument('--label', type=int, default=0)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-open', action='store_true', help='不自动打开默认浏览器')
    parser.add_argument('--check', action='store_true', help='校验全部记录、时间和哈希后退出')
    args = parser.parse_args()
    prediction_dir = args.predictions_dir or args.data_dir.parent/'diagnostics/imu_only_epoch30/predictions'
    try:
        dataset = ReplayDataset(args.data_dir, prediction_dir)
        if args.check:
            ends = []; count = 0; early = 0
            for group, label in sorted(dataset.entries):
                record = dataset.record(group, label); count += 1
                early += record['meta']['termination'] != 'completed'; ends.append(record['meta']['actual_end_s'])
            print(json.dumps({'checked': count, 'early_endings': early, 'min_end_s': min(ends),
                              'max_end_s': max(ends)}, ensure_ascii=False))
            return
        if (args.group, args.label) not in dataset.entries:
            raise ValueError('初始组别/标签不存在，请用--group 1…8、--label 0…18。')
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(dataset))
    except (ValueError, OSError) as exc:
        parser.exit(1, f'无法启动回放：{exc}\n')
    url = f'http://127.0.0.1:{server.server_port}/?group={args.group}&label={args.label}'
    print(f'任务回放：{url}\n记录数量：{len(dataset.entries)}。按 Ctrl+C 停止。', flush=True)
    if not args.no_open:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
