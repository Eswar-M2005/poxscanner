"""Perceptual hash helpers for near-duplicate image detection.

Average hash (aHash) is implemented with Pillow only — no `imagehash` dependency.
Near-duplicates use Hamming distance ≤ 5 (see docs/contracts.md split policy).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from PIL import Image

HASH_SIZE = 8
NEAR_DUPLICATE_HAMMING = 5

try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    _RESAMPLE = Image.LANCZOS  # type: ignore[attr-defined]


def average_hash(image: Image.Image, hash_size: int = HASH_SIZE) -> int:
    """Return an aHash packed into an integer (`hash_size ** 2` bits).

    Pipeline: grayscale → `hash_size`×`hash_size` → bit is 1 iff pixel > mean.
    """
    if hash_size < 1:
        raise ValueError("hash_size must be >= 1")
    gray = image.convert("L")
    small = gray.resize((hash_size, hash_size), _RESAMPLE)
    pixels = list(small.tobytes())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value > mean:
            bits |= 1 << index
    return bits


def average_hash_hex(image: Image.Image, hash_size: int = HASH_SIZE) -> str:
    """Return aHash as a zero-padded lowercase hex string."""
    nbits = hash_size * hash_size
    width = max(1, (nbits + 3) // 4)
    return f"{average_hash(image, hash_size):0{width}x}"


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Number of differing bits between two integer hashes."""
    return (hash_a ^ hash_b).bit_count()


def are_near_duplicates(
    hash_a: int,
    hash_b: int,
    threshold: int = NEAR_DUPLICATE_HAMMING,
) -> bool:
    return hamming_distance(hash_a, hash_b) <= threshold


def cluster_by_hamming(
    id_to_hash: Mapping[str, int],
    threshold: int = NEAR_DUPLICATE_HAMMING,
) -> list[list[str]]:
    """Cluster ids whose hashes are linked by Hamming distance ≤ `threshold`.

    Union-find is transitive: A~B and B~C join one group even if A vs C is
    slightly over the threshold. Only clusters of size ≥ 2 are returned.
    Ids in each cluster are sorted for stable output.
    """
    ids = list(id_to_hash)
    if len(ids) < 2:
        return []

    parent = {key: key for key in ids}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    n = len(ids)
    for i in range(n):
        hash_i = id_to_hash[ids[i]]
        for j in range(i + 1, n):
            if hamming_distance(hash_i, id_to_hash[ids[j]]) <= threshold:
                union(ids[i], ids[j])

    groups: dict[str, list[str]] = defaultdict(list)
    for key in ids:
        groups[find(key)].append(key)

    clusters = [sorted(members) for members in groups.values() if len(members) > 1]
    clusters.sort(key=lambda members: members[0])
    return clusters
