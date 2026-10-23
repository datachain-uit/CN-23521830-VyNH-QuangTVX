"""Read-only Kaggle dataset tree inventory for reproducible input packaging.

This helper exists for the CQ V3 M0 handoff.  It lists remote files and sizes
without downloading dataset contents, follows pagination, and never modifies a
Kaggle dataset.  The resulting JSON is evidence for the exact subset selected
for a local/Kaggle run bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_kaggle_sdk(vendor_dir: Path | None) -> tuple[Any, Any]:
    if vendor_dir is not None:
        sys.path.insert(0, str(vendor_dir.resolve()))

    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import (
        ApiListTreeDatasetFilesRequest,
    )

    return build_kaggle_client, ApiListTreeDatasetFilesRequest


def _list_directory(
    client: Any,
    request_type: Any,
    *,
    owner: str,
    dataset: str,
    version: int,
    path: str,
) -> dict[str, list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        request = request_type()
        request.owner_slug = owner
        request.dataset_slug = dataset
        request.dataset_version_number = version
        request.path = path
        request.page_size = 200
        if page_token:
            request.page_token = page_token

        response = client.datasets.dataset_api_client.list_tree_dataset_files(request)
        payload = response.to_dict()
        files.extend(payload.get("files", []))
        directories.extend(payload.get("directories", []))
        page_token = payload.get("nextPageToken") or payload.get("next_page_token")
        if not page_token:
            break

    return {"files": files, "directories": directories}


def inventory_tree(
    *,
    owner: str,
    dataset: str,
    version: int,
    root: str,
    vendor_dir: Path | None,
) -> dict[str, Any]:
    build_client, request_type = _load_kaggle_sdk(vendor_dir)
    client = build_client()
    pending = [root.strip("/")]
    visited: set[str] = set()
    files: list[dict[str, Any]] = []

    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        listing = _list_directory(
            client,
            request_type,
            owner=owner,
            dataset=dataset,
            version=version,
            path=current,
        )
        files.extend(listing["files"])
        pending.extend(item["relativeUrl"] for item in listing["directories"])

    files.sort(key=lambda item: item["relativeUrl"])
    return {
        "dataset": f"{owner}/{dataset}",
        "version": version,
        "root": root,
        "file_count": len(files),
        "total_bytes": sum(int(item.get("totalBytes", 0)) for item in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="w4ngggg/phase-features-v3")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--root", default="phase_features_v3/course_random_split")
    parser.add_argument("--vendor-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    owner, dataset = args.dataset.split("/", maxsplit=1)
    payload = inventory_tree(
        owner=owner,
        dataset=dataset,
        version=args.version,
        root=args.root,
        vendor_dir=args.vendor_dir,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
