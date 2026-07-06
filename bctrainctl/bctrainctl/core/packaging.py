"""Project packaging helpers."""

from __future__ import annotations

import fnmatch
import os
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import pathspec

from bctrainctl.core.models import PackageResult
from bctrainctl.utils.errors import BCTrainctlError
from bctrainctl.utils.hashing import sha256_file
from bctrainctl.utils.paths import ensure_app_dirs
from bctrainctl.utils.time import compact_utc_timestamp

# Excluded by default. These are gitignore-style patterns evaluated together
# with the user-supplied `exclude` rules and the project `.gitignore`.
DEFAULT_EXCLUDES = [
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    "outputs",
    "wandb",
]

# Directories we never descend into, regardless of include rules. Re-including a
# path from inside these is never meaningful and walking them is expensive.
HARD_PRUNE_NAMES = {".git", "__pycache__", ".venv", "node_modules"}


class GitignoreFilter:
    """Decide which paths to keep, with Git-compatible ignore semantics.

    Layers, in increasing priority:
    1. `DEFAULT_EXCLUDES` + the user `exclude` rules + the project `.gitignore`
       (negations inside `.gitignore` are honored).
    2. `gitignore_include` rules, which re-include paths even if step 1 ignored
       them — this is intentionally more permissive than Git, so a path can be
       kept even when its parent directory was excluded.
    """

    def __init__(
        self,
        *,
        project_path: Path,
        exclude: list[str],
        use_gitignore: bool,
        gitignore_include: list[str],
    ) -> None:
        ignore_lines = [*DEFAULT_EXCLUDES, *exclude]
        if use_gitignore:
            ignore_lines += _read_gitignore_lines(project_path)
        self._ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", ignore_lines)

        self._include_lines = [line.strip() for line in gitignore_include if line.strip()]
        self._include_spec = (
            pathspec.PathSpec.from_lines("gitwildmatch", self._include_lines)
            if self._include_lines
            else None
        )

        # Re-include rules that may resurrect paths inside an ignored directory:
        # explicit `gitignore_include` plus any `!` negations from the ignore
        # layers. These drive directory pruning so an except rule takes effect
        # consistently regardless of which directory it lives under.
        self._reinclude_lines = list(self._include_lines)
        self._reinclude_lines += [
            line[1:].strip() for line in ignore_lines if line.startswith("!") and line[1:].strip()
        ]

    def is_excluded(self, relative_path: Path) -> bool:
        rel = relative_path.as_posix()
        if rel in ("", "."):
            return False
        if not self._ignore_spec.match_file(rel):
            return False
        if self._include_spec is not None and self._include_spec.match_file(rel):
            return False
        return True

    def should_prune_dir(self, relative_path: Path) -> bool:
        """Whether `os.walk` can safely skip descending into this directory."""

        if relative_path.name in HARD_PRUNE_NAMES:
            return True
        rel = relative_path.as_posix()
        if not (self._ignore_spec.match_file(rel) or self._ignore_spec.match_file(f"{rel}/")):
            return False
        # Directory is ignored. Keep walking only if a re-include rule could
        # match something inside it.
        if self._include_spec is not None and (
            self._include_spec.match_file(rel) or self._include_spec.match_file(f"{rel}/")
        ):
            return False
        if _include_descends_into(self._reinclude_lines, rel):
            return False
        return True


def package_extra_upload(*, local_path: Path, label: str) -> PackageResult:
    """Package a single extra file or directory into a tar.gz archive.

    Directories are archived so their *contents* land directly under the
    extraction target; single files keep their basename. Used by
    `spec.extra_uploads` to ship small datasets or test fixtures alongside code.
    """

    if not local_path.exists():
        raise BCTrainctlError(
            f"Extra upload path does not exist: {local_path}",
            hint="Check `spec.extra_uploads[].local` and run from the expected directory.",
        )

    paths = ensure_app_dirs()
    timestamp = compact_utc_timestamp()
    safe_label = label.replace("/", "-").replace(" ", "-")
    final_path = paths.packages_dir / f"extra-{safe_label}-{timestamp}.tar.gz"

    try:
        with NamedTemporaryFile(
            suffix=".tar.gz",
            prefix="bctrainctl-extra-",
            dir=paths.packages_dir,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        file_count = 0
        with tarfile.open(tmp_path, mode="w:gz") as archive:
            if local_path.is_dir():
                for root, _dirs, files in os.walk(local_path, topdown=True):
                    root_rel = Path(root).relative_to(local_path)
                    for filename in files:
                        relative_path = root_rel / filename
                        archive.add(
                            Path(root) / filename, arcname=relative_path.as_posix()
                        )
                        file_count += 1
            else:
                archive.add(local_path, arcname=local_path.name)
                file_count = 1

        tmp_path.replace(final_path)
        package_hash = sha256_file(final_path)
        return PackageResult(
            package_path=final_path,
            sha256=package_hash,
            file_count=file_count,
            size_bytes=final_path.stat().st_size,
        )
    except Exception as exc:  # noqa: BLE001
        raise BCTrainctlError(
            f"Failed to package extra upload from {local_path}: {exc}",
            hint="Check file permissions and available disk space.",
        ) from exc


def _read_gitignore_lines(project_path: Path) -> list[str]:
    """Read raw, comment-stripped `.gitignore` lines (negations preserved)."""

    gitignore_path = project_path / ".gitignore"
    if not gitignore_path.is_file():
        return []
    lines: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def _include_descends_into(include_lines: list[str], dir_rel: str) -> bool:
    """Whether some include rule could match a path inside `dir_rel`."""

    dir_parts = dir_rel.split("/")
    for raw in include_lines:
        pattern = raw[1:] if raw.startswith("!") else raw
        pattern = pattern.strip("/")
        if not pattern:
            continue
        # Patterns that can match at any depth force a descent.
        if "/" not in pattern or pattern == "**" or pattern.startswith("**/"):
            return True
        pattern_parts = pattern.split("/")
        prefix_matches = True
        for index, dir_seg in enumerate(dir_parts):
            if index >= len(pattern_parts):
                break  # directory is an ancestor of the pattern path
            if not _segment_matches(pattern_parts[index], dir_seg):
                prefix_matches = False
                break
        if prefix_matches:
            return True
    return False


def _segment_matches(pattern_segment: str, name: str) -> bool:
    return pattern_segment == name or fnmatch.fnmatch(name, pattern_segment)


def package_project(
    *,
    project_path: Path,
    job_name: str,
    exclude: list[str],
    use_gitignore: bool = False,
    gitignore_include: list[str] | None = None,
) -> PackageResult:
    """Create a tar.gz archive from a local project directory."""

    paths = ensure_app_dirs()
    timestamp = compact_utc_timestamp()
    safe_job_name = job_name.replace("/", "-").replace(" ", "-")
    final_path = paths.packages_dir / f"{safe_job_name}-{timestamp}.tar.gz"

    file_filter = GitignoreFilter(
        project_path=project_path,
        exclude=exclude,
        use_gitignore=use_gitignore,
        gitignore_include=gitignore_include or [],
    )

    try:
        with NamedTemporaryFile(
            suffix=".tar.gz",
            prefix="bctrainctl-",
            dir=paths.packages_dir,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        file_count = 0
        with tarfile.open(tmp_path, mode="w:gz") as archive:
            for root, dirs, files in os.walk(project_path, topdown=True):
                root_path = Path(root)
                root_rel = root_path.relative_to(project_path)
                dirs[:] = [
                    item
                    for item in dirs
                    if not file_filter.should_prune_dir(root_rel / item)
                ]
                for filename in files:
                    relative_path = root_rel / filename
                    if file_filter.is_excluded(relative_path):
                        continue
                    archive.add(project_path / relative_path, arcname=relative_path.as_posix())
                    file_count += 1

        tmp_path.replace(final_path)
        package_hash = sha256_file(final_path)
        return PackageResult(
            package_path=final_path,
            sha256=package_hash,
            file_count=file_count,
            size_bytes=final_path.stat().st_size,
        )
    except Exception as exc:  # noqa: BLE001
        raise BCTrainctlError(
            f"Failed to package project from {project_path}: {exc}",
            hint="Check file permissions, exclude rules, and available disk space.",
        ) from exc
