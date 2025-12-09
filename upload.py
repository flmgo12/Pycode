import os
import math
import argparse
from pathlib import Path

import requests

GITHUB_API = "https://api.github.com"
GITHUB_UPLOADS = "https://uploads.github.com"


def create_release(owner, repo, tag_name, release_name, token, body=""):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/releases"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "tag_name": tag_name,
        "name": release_name,
        "body": body,
        "draft": False,
        "prerelease": False,
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code >= 300:
        raise RuntimeError(
            f"创建 release 失败: {resp.status_code} {resp.text}"
        )
    data = resp.json()
    return data["id"], data["html_url"]


def upload_asset(owner, repo, release_id, file_path: Path, token: str):
    url = f"{GITHUB_UPLOADS}/repos/{owner}/{repo}/releases/{release_id}/assets"
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/octet-stream",
        "Accept": "application/vnd.github+json",
    }
    params = {"name": file_path.name}

    with file_path.open("rb") as f:
        resp = requests.post(url, headers=headers, params=params, data=f)
    if resp.status_code >= 300:
        raise RuntimeError(
            f"上传 {file_path.name} 失败: {resp.status_code} {resp.text}"
        )
    print(f"  已上传: {file_path.name}")


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def main():
    parser = argparse.ArgumentParser(
        description="按每 10 个 .bin 文件创建一个 GitHub Release 并上传"
    )
    parser.add_argument(
        "--owner",
        required=True,
        help="GitHub 仓库 owner（例如：yourname）",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub 仓库名（例如：your-repo）",
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="包含 1.bin, 2.bin ... 的目录（默认：当前目录）",
    )
    parser.add_argument(
        "--pattern",
        default="*.bin",
        help="文件匹配模式（默认：*.bin）",
    )
    parser.add_argument(
        "--per_release",
        type=int,
        default=10,
        help="每个 release 上传多少个文件（默认 10）",
    )
    parser.add_argument(
        "--tag_prefix",
        default="bin-chunk",
        help="tag 前缀（默认：bin-chunk）",
    )
    parser.add_argument(
        "--release_prefix",
        default="BIN Chunk",
        help="release 名称前缀（默认：BIN Chunk）",
    )
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "未找到 GITHUB_TOKEN 环境变量，请先设置 GitHub Personal Access Token。"
        )

    base_dir = Path(args.dir).resolve()
    files = sorted(
        base_dir.glob(args.pattern),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
    )

    if not files:
        raise RuntimeError(f"目录 {base_dir} 下没有匹配到文件：{args.pattern}")

    print(f"在 {base_dir} 下共找到 {len(files)} 个文件，准备每 {args.per_release} 个打一个 release")

    groups = list(chunk_list(files, args.per_release))
    total_releases = len(groups)

    for idx, group in enumerate(groups, start=1):
        tag_name = f"{args.tag_prefix}-part-{idx:03d}"
        release_name = f"{args.release_prefix} #{idx:03d}"
        body = f"自动上传的分片，第 {idx}/{total_releases} 批，共 {len(group)} 个文件。"

        print(f"\n=== 创建 Release {idx}/{total_releases} ===")
        print(f"tag: {tag_name}, name: {release_name}")

        release_id, release_url = create_release(
            owner=args.owner,
            repo=args.repo,
            tag_name=tag_name,
            release_name=release_name,
            token=token,
            body=body,
        )
        print(f"Release 已创建: {release_url}")

        for f in group:
            upload_asset(args.owner, args.repo, release_id, f, token)

    print("\n全部完成！")


if __name__ == "__main__":
    main()
