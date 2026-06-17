from __future__ import annotations

import argparse
import os
import re
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeRequestHandler(SimpleHTTPRequestHandler):
    _range: tuple[int, int] | None = None

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        stat = os.fstat(source.fileno())
        size = stat.st_size
        content_type = self.guess_type(path)
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"^bytes=(\d*)-(\d*)$", range_header.strip())
            if match:
                start_text, end_text = match.groups()
                if start_text == "" and end_text:
                    suffix_len = int(end_text)
                    start = max(0, size - suffix_len)
                    end = size - 1
                else:
                    start = int(start_text or "0")
                    end = int(end_text) if end_text else size - 1
                end = min(end, size - 1)
                if start < 0 or start >= size or end < start:
                    source.close()
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return None

                self._range = (start, end)
                source.seek(start)
                self.send_response(206)
                self.send_header("Content-type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
                self.end_headers()
                return source

        self._range = None
        self.send_response(200)
        self.send_header("Content-type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.end_headers()
        return source

    def copyfile(self, source, outputfile):
        selected_range = self._range
        if selected_range is None:
            shutil.copyfileobj(source, outputfile)
            return
        remaining = selected_range[1] - selected_range[0] + 1
        while remaining > 0:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    handler = lambda *handler_args, **handler_kwargs: RangeRequestHandler(
        *handler_args,
        directory=args.directory,
        **handler_kwargs,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {args.directory} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
