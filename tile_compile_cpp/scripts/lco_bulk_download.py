#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import subprocess


def _http_get_json(url: str, timeout_s: int) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _http_post_json(url: str, payload: dict, timeout_s: int) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    header_variants: list[dict[str, str]] = [
        # Some servers reject unknown/strict Accept headers with 406.
        {"Content-Type": "application/json", "Accept": "*/*"},
        # Fallback: no Accept at all.
        {"Content-Type": "application/json"},
        # Fallback: explicitly accept octet-stream.
        {"Content-Type": "application/json", "Accept": "application/octet-stream"},
    ]

    last_err: Exception | None = None
    for headers in header_variants:
        try:
            req = urllib.request.Request(url, data=body, method="POST", headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last_err = e
            # If it's not a content negotiation issue, don't keep retrying.
            if int(getattr(e, "code", 0) or 0) != 406:
                raise
        except Exception as e:
            last_err = e
            raise

    if last_err is not None:
        raise last_err
    raise RuntimeError("POST failed")


def _download_file(url: str, out_path: str, timeout_s: int, retries: int) -> None:
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.0 + attempt)
                continue
            raise
    if last_err is not None:
        raise last_err


def _frames_query_url(
    object_name: str,
    n: int,
    rlevel: int,
    obstype: str,
    public: bool,
    extra_params: list[str],
) -> str:
    params: list[tuple[str, str]] = [
        ("OBJECT", object_name),
        ("OBSTYPE", obstype),
        ("RLEVEL", str(rlevel)),
        ("limit", str(n)),
    ]
    if public:
        params.append(("public", "true"))

    for kv in extra_params:
        if "=" not in kv:
            raise ValueError(f"Bad --param '{kv}', expected KEY=VALUE")
        k, v = kv.split("=", 1)
        params.append((k, v))

    qs = urllib.parse.urlencode(params)
    return f"https://archive-api.lco.global/frames/?{qs}"


def _sanitize_component(s: str) -> str:
    s = s.strip()
    if not s:
        return "unknown"
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in s)


def _extract_zip(zip_path: str, out_dir: str) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target_path = os.path.join(out_dir, info.filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target_path)
    return extracted


def _try_funpack(path_fz: str) -> bool:
    funpack = shutil.which("funpack")
    if not funpack:
        return False
    try:
        subprocess.run([funpack, "-O", path_fz[:-3], path_fz], check=True)
        return True
    except Exception:
        return False


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Download public Las Cumbres Observatory (LCO) frames as raw FITS (.fits.fz) "
            "using the archive-api.lco.global endpoint."
        )
    )

    ap.add_argument("--object", required=True, help="Target name, e.g. M42, M31, IC434")
    ap.add_argument("--n", type=int, default=50, help="Number of frames to fetch")
    ap.add_argument(
        "--out",
        default="./lco_download",
        help="Output directory (will be created)",
    )
    ap.add_argument(
        "--rlevel",
        type=int,
        default=0,
        help="Reduction level: 0=raw, 91=processed",
    )
    ap.add_argument(
        "--obstype",
        default="EXPOSE",
        help="OBSTYPE, usually EXPOSE (avoid GUIDE)",
    )
    ap.add_argument(
        "--public",
        action="store_true",
        default=True,
        help="Only public data (default: true)",
    )
    ap.add_argument(
        "--no-public",
        dest="public",
        action="store_false",
        help="Allow non-public (will likely require auth, so usually not useful)",
    )
    ap.add_argument(
        "--zip",
        action="store_true",
        help="Download as a single zip via /frames/zip/ (requires collecting frame_ids)",
    )
    ap.add_argument(
        "--uncompress",
        action="store_true",
        help="When using --zip, request uncompressed .fits (bigger) instead of .fits.fz",
    )
    ap.add_argument(
        "--filters",
        default="",
        help="Comma-separated filter list to download (e.g. rp,gp,ip). If set, downloads per filter.",
    )
    ap.add_argument(
        "--extract",
        action="store_true",
        help="If --zip is used, extract the zip after download into the target directory.",
    )
    ap.add_argument(
        "--funpack",
        action="store_true",
        help="After download/extract, convert *.fits.fz to *.fits using system 'funpack' if available.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per file",
    )
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        help=(
            "Additional query parameter(s) for the frames endpoint, e.g. --param FILTER=rp "
            "(can be specified multiple times)"
        ),
    )

    args = ap.parse_args(argv)

    object_name = args.object
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    filters: list[str] = []
    if isinstance(args.filters, str) and args.filters.strip():
        filters = [f.strip() for f in args.filters.split(",") if f.strip()]

    target_base_dir = os.path.join(out_dir, _sanitize_component(object_name))
    os.makedirs(target_base_dir, exist_ok=True)

    def run_for_filter(filter_value: str | None) -> int:
        extra_params = list(args.param)
        suffix = ""
        target_dir = target_base_dir
        if filter_value:
            extra_params.append(f"FILTER={filter_value}")
            suffix = f"_{_sanitize_component(filter_value)}"
            target_dir = os.path.join(target_base_dir, _sanitize_component(filter_value))
            os.makedirs(target_dir, exist_ok=True)

        query_url = _frames_query_url(
            object_name=object_name,
            n=args.n,
            rlevel=args.rlevel,
            obstype=args.obstype,
            public=args.public,
            extra_params=extra_params,
        )

        print(f"Query: {query_url}")
        data = _http_get_json(query_url, timeout_s=args.timeout)

        results = data.get("results", [])
        if not results:
            print(
                "No frames returned. Try removing filters or adjust --param / --obstype / --rlevel.",
                file=sys.stderr,
            )
            return 2

        if args.zip:
            frame_ids = [int(r["id"]) for r in results if "id" in r]
            zip_payload = {"frame_ids": frame_ids, "uncompress": bool(args.uncompress)}
            zip_url = "https://archive-api.lco.global/frames/zip/"
            print(f"Requesting zip for {len(frame_ids)} frames...")
            zip_bytes = _http_post_json(zip_url, zip_payload, timeout_s=args.timeout)
            zip_name = (
                f"{_sanitize_component(object_name)}{suffix}_n{len(frame_ids)}_rlevel{args.rlevel}_{args.obstype}"
            )
            if args.uncompress:
                zip_name += "_uncompressed"
            zip_path = os.path.join(target_dir, zip_name + ".zip")
            with open(zip_path, "wb") as f:
                f.write(zip_bytes)
            print(f"Wrote: {zip_path}")

            extracted_files: list[str] = []
            if args.extract:
                print(f"Extracting zip to: {target_dir}")
                extracted_files = _extract_zip(zip_path, target_dir)
                print(f"Extracted {len(extracted_files)} files")

            if args.funpack:
                funpacked = 0
                candidates = extracted_files if extracted_files else []
                if not candidates:
                    candidates = [os.path.join(target_dir, r.get("filename", "")) for r in results]
                for p in candidates:
                    if p.endswith(".fits.fz") and os.path.exists(p):
                        if _try_funpack(p):
                            funpacked += 1
                if funpacked == 0:
                    if not shutil.which("funpack"):
                        print("funpack not found in PATH; leaving .fits.fz as-is", file=sys.stderr)
            return 0

        ok = 0
        downloaded_paths: list[str] = []
        for idx, r in enumerate(results, start=1):
            url = r.get("url")
            filename = r.get("filename")
            if not url or not filename:
                continue

            out_path = os.path.join(target_dir, filename)
            downloaded_paths.append(out_path)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"[{idx}/{len(results)}] skip existing {filename}")
                ok += 1
                continue

            print(f"[{idx}/{len(results)}] download {filename}")
            try:
                _download_file(url, out_path, timeout_s=args.timeout, retries=args.retries)
                ok += 1
            except Exception as e:
                print(f"Failed: {filename}: {e}", file=sys.stderr)

        if ok == 0:
            return 3

        if args.funpack:
            funpacked = 0
            for p in downloaded_paths:
                if p.endswith(".fits.fz") and os.path.exists(p):
                    if _try_funpack(p):
                        funpacked += 1
            if funpacked == 0:
                if not shutil.which("funpack"):
                    print("funpack not found in PATH; leaving .fits.fz as-is", file=sys.stderr)

        print(f"Done. Downloaded {ok}/{len(results)}")
        return 0

    if filters:
        rc = 0
        for f in filters:
            rcf = run_for_filter(f)
            if rcf != 0:
                rc = rcf
        return rc

    return run_for_filter(None)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
