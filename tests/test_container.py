"""Container files (DECISIONS.md §10). Read as text, without a Docker daemon;
the image itself was checked by hand, see §10."""

from __future__ import annotations

import re
import tomllib

from tests.support import ROOT

COMPOSE: str = (ROOT / "compose.yaml").read_text(encoding="utf-8")
DOCKERFILE: str = (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_image_tag_follows_the_package_version() -> None:
    version: str = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert f"image: teach-by-showing:{version}" in COMPOSE


def test_the_archive_commands_name_the_current_version() -> None:
    """§10: the README's archive commands are bumped with the version."""
    version: str = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    saved: list[str] = re.findall(r"docker save teach-by-showing:(\S+) ", readme)
    archives: list[str] = re.findall(r"teach-by-showing-(\S+)\.tar\.zst", readme)
    assert saved == [version] and archives and set(archives) == {version}


def test_port_is_published_to_localhost_only() -> None:
    ports: list[str] = re.findall(r'^\s*- "([^"]*:8081)"', COMPOSE, flags=re.M)
    assert ports and all(p.startswith("127.0.0.1:") for p in ports)


def test_crashed_server_is_restarted() -> None:
    assert re.search(r"^\s*restart: unless-stopped$", COMPOSE, flags=re.M)


def test_no_devices_gpu_or_host_network() -> None:
    settings: str = "\n".join(line.split("#", 1)[0] for line in COMPOSE.splitlines())
    for word in ("devices:", "nvidia", "network_mode", "privileged", "ipc:"):
        assert word not in settings, word


def test_base_image_is_pinned_by_digest() -> None:
    assert re.search(
        r"^ARG PYTHON_IMAGE=python:3\.12-slim@sha256:[0-9a-f]{64}$", DOCKERFILE, re.M
    )
    assert "--require-hashes" in DOCKERFILE


def test_build_context_is_an_allow_list() -> None:
    lines: list[str] = [
        line
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert lines[0] == "*"
