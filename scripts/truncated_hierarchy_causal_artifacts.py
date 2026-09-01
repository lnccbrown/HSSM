"""Immutable, hash-bound artifacts for the #1282 causal experiment.

The causal runner treats a cell JSON as a commit marker.  Every artifact it
references is therefore written first, without replacement, and the exact bytes
are bound by SHA-256.  JSON is restricted to the deterministic finite subset so
that a digest identifies one unambiguous payload on every supported Python.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

SHA256_HEX_LENGTH = 64


class CausalArtifactError(RuntimeError):
    """Raised when immutable artifact publication or verification fails."""


def _validate_json_value(value: Any, path: str = "$") -> None:
    """Reject values outside canonical finite JSON before serialization."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CausalArtifactError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CausalArtifactError(f"{path} contains a non-string key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise CausalArtifactError(
        f"{path} contains unsupported JSON value {type(value).__name__}"
    )


def canonical_json_bytes(payload: Any) -> bytes:
    """Encode one finite JSON value in the experiment's canonical form."""
    _validate_json_value(payload)
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_constant(token: str) -> None:
    raise CausalArtifactError(f"JSON contains forbidden constant {token!r}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CausalArtifactError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def decode_canonical_json(data: bytes) -> Any:
    """Decode strict JSON and require its bytes to already be canonical."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CausalArtifactError("JSON artifact is not valid UTF-8") from error
    try:
        payload = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise CausalArtifactError("artifact is not strict JSON") from error
    if canonical_json_bytes(payload) != data:
        raise CausalArtifactError("JSON artifact is not in canonical byte form")
    return payload


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a regular file without loading it into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CausalArtifactError(f"cannot hash artifact {path}") from error
    return digest.hexdigest()


def validate_sha256(value: str) -> str:
    """Validate and normalize a lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CausalArtifactError("SHA-256 must be 64 lowercase hexadecimal digits")
    return value


def validate_relative_path(value: str) -> str:
    """Return one normalized artifact-relative POSIX path.

    Absolute paths, empty components, dot components, platform separators, and
    traversal are rejected.  The resulting path is safe to resolve below a run
    root and stable when recorded in JSON on any platform.
    """
    if not isinstance(value, str) or not value:
        raise CausalArtifactError("artifact path must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise CausalArtifactError("artifact path must use safe POSIX components")
    pure = PurePosixPath(value)
    if pure.is_absolute() or str(pure) != value:
        raise CausalArtifactError("artifact path must be normalized and relative")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise CausalArtifactError("artifact path contains an unsafe component")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A run-relative immutable artifact bound to its exact bytes."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        """Validate the relative path, exact digest, and byte count."""
        object.__setattr__(self, "path", validate_relative_path(self.path))
        object.__setattr__(self, "sha256", validate_sha256(self.sha256))
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise CausalArtifactError("artifact size_bytes must be non-negative")

    def as_dict(self) -> dict[str, str | int]:
        """Return the canonical result-record representation."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactRef:
        """Parse an exact artifact reference without accepting extra fields."""
        if set(value) != {"path", "sha256", "size_bytes"}:
            raise CausalArtifactError("artifact reference has unexpected fields")
        return cls(
            path=value["path"],
            sha256=value["sha256"],
            size_bytes=value["size_bytes"],
        )


class ArtifactStore:
    """Publish and verify immutable files below one explicit run root."""

    def __init__(self, root: str | Path) -> None:
        root_path = Path(root)
        if not root_path.is_absolute():
            raise CausalArtifactError("artifact root must be absolute")
        self.root = root_path.resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        """Resolve an artifact path while preventing symlink escape."""
        relative = validate_relative_path(relative_path)
        candidate = self.root.joinpath(*PurePosixPath(relative).parts)
        parent = candidate.parent
        parent.mkdir(parents=True, exist_ok=True)
        try:
            resolved_parent = parent.resolve(strict=True)
            resolved_parent.relative_to(self.root)
        except (OSError, ValueError) as error:
            raise CausalArtifactError("artifact path escapes the run root") from error
        if candidate.is_symlink():
            raise CausalArtifactError("artifact destination may not be a symlink")
        return candidate

    def write_bytes(
        self,
        relative_path: str,
        data: bytes,
        *,
        expected_sha256: str | None = None,
    ) -> ArtifactRef:
        """Publish bytes atomically and refuse to replace any existing path."""
        if not isinstance(data, bytes):
            raise TypeError("artifact payload must be bytes")
        digest = sha256_bytes(data)
        if expected_sha256 is not None and digest != validate_sha256(expected_sha256):
            raise CausalArtifactError("artifact bytes do not match expected SHA-256")
        destination = self.resolve(relative_path)
        temporary: Path | None = None
        descriptor: int | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = None
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # link(2) is the portable no-clobber atomic publication primitive.
            os.link(temporary, destination)
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except FileExistsError as error:
            raise CausalArtifactError(
                f"refusing to overwrite immutable artifact {relative_path!r}"
            ) from error
        except OSError as error:
            raise CausalArtifactError(
                f"could not publish artifact {relative_path!r}"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return ArtifactRef(relative_path, digest, len(data))

    def write_json(self, relative_path: str, payload: Any) -> ArtifactRef:
        """Canonicalize and atomically publish one JSON artifact."""
        return self.write_bytes(relative_path, canonical_json_bytes(payload))

    def ensure_bytes(self, relative_path: str, data: bytes) -> ArtifactRef:
        """Publish deterministic bytes once, or verify an identical prior writer.

        Shared inputs may be materialized concurrently by backend-pair workers.
        A losing writer is allowed to reuse the winner's file only
        when its expected size and digest match exactly.
        """
        expected = ArtifactRef(relative_path, sha256_bytes(data), len(data))
        try:
            return self.write_bytes(relative_path, data)
        except CausalArtifactError as error:
            path = self.root.joinpath(*PurePosixPath(expected.path).parts)
            if not path.exists():
                raise
            try:
                self.verify(expected)
            except CausalArtifactError:
                raise error
            return expected

    def ensure_json(self, relative_path: str, payload: Any) -> ArtifactRef:
        """Publish or verify one deterministic shared JSON artifact."""
        return self.ensure_bytes(relative_path, canonical_json_bytes(payload))

    def verify(self, reference: ArtifactRef) -> Path:
        """Verify path type, size, and digest before returning a file path."""
        path = self.resolve(reference.path)
        if not path.is_file() or path.is_symlink():
            raise CausalArtifactError(f"artifact {reference.path!r} is not a file")
        if path.stat().st_size != reference.size_bytes:
            raise CausalArtifactError(f"artifact {reference.path!r} has wrong size")
        if sha256_file(path) != reference.sha256:
            raise CausalArtifactError(f"artifact {reference.path!r} has wrong SHA-256")
        return path

    def read_bytes(self, reference: ArtifactRef) -> bytes:
        """Read an artifact only after verifying its exact bytes."""
        try:
            return self.verify(reference).read_bytes()
        except OSError as error:
            raise CausalArtifactError(
                f"cannot read artifact {reference.path!r}"
            ) from error

    def read_json(self, reference: ArtifactRef) -> Any:
        """Read and decode a hash-verified canonical JSON artifact."""
        return decode_canonical_json(self.read_bytes(reference))


def merge_run_directories(
    source_directory: Path, destination_root: Path
) -> dict[str, int]:
    """Merge downloaded per-pair run roots without overwriting evidence.

    Each immediate child of ``source_directory`` is one extracted pair artifact.
    Duplicate shared inputs are accepted only when their exact bytes agree.  The
    function rejects links and unexpected top-level paths before publication.
    """
    source = source_directory.resolve(strict=True)
    destination = destination_root.resolve(strict=False)
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise CausalArtifactError("merge source and destination must be disjoint")
    allowed_roots = {
        "contexts",
        "data",
        "starts",
        "chains",
        "diagnostics",
        "cells",
    }
    store = ArtifactStore(destination)
    published = 0
    identical = 0
    children = sorted(source.iterdir())
    if not children:
        raise CausalArtifactError("merge source directory is empty")
    for run_root in children:
        if run_root.is_symlink() or not run_root.is_dir():
            raise CausalArtifactError(
                "every merge source child must be a real directory"
            )
        for path in sorted(run_root.rglob("*")):
            if path.is_symlink():
                raise CausalArtifactError("merge sources may not contain symlinks")
            if path.is_dir():
                continue
            if not path.is_file():
                raise CausalArtifactError(
                    "merge sources may contain only regular files"
                )
            relative = path.relative_to(run_root).as_posix()
            validate_relative_path(relative)
            if PurePosixPath(relative).parts[0] not in allowed_roots:
                raise CausalArtifactError(
                    f"unexpected top-level merge artifact {relative!r}"
                )
            data = path.read_bytes()
            target = destination.joinpath(*PurePosixPath(relative).parts)
            existed = target.exists()
            store.ensure_bytes(relative, data)
            if existed:
                identical += 1
            else:
                published += 1
    return {"published": published, "identical": identical}
