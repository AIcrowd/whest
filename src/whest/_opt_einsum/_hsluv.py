"""Minimal HSLuv helpers for terminal label palettes.

Adapted from the MIT-licensed ``hsluv-python`` reference implementation:
https://github.com/hsluv/hsluv-python

The palette construction idea mirrors seaborn's ``husl_palette`` API, but
this module does not vendor seaborn code or add a seaborn dependency.
See NOTICE in this package for attribution details.
"""

from __future__ import annotations

import math
from functools import cache

_M = (
    (3.240969941904521, -1.537383177570093, -0.498610760293),
    (-0.96924363628087, 1.87596750150772, 0.041555057407175),
    (0.055630079696993, -0.20397695888897, 1.056971514242878),
)
_M_INV = (
    (0.41239079926595, 0.35758433938387, 0.18048078840183),
    (0.21263900587151, 0.71516867876775, 0.072192315360733),
    (0.019330818715591, 0.11919477979462, 0.95053215224966),
)
_REF_Y = 1.0
_REF_U = 0.19783000664283
_REF_V = 0.46831999493879
_KAPPA = 903.2962962
_EPSILON = 0.0088564516

# Frozen to preserve the current Rich label colors. Keep this in sync with
# qualitative_hsluv_palette(64, lightness=0.5, saturation=0.9).
PRECOMPUTED_QUALITATIVE_HSLUV_PALETTE_64 = (
    "#E3245A",
    "#2F70EF",
    "#228921",
    "#B92FED",
    "#987022",
    "#268389",
    "#7B5EF0",
    "#D529A6",
    "#D14923",
    "#5E8222",
    "#238755",
    "#2A7CBC",
    "#E7232C",
    "#9D4AF0",
    "#5569F0",
    "#DD2680",
    "#C82CCA",
    "#B75F22",
    "#7C7B22",
    "#408722",
    "#24856F",
    "#22893B",
    "#2C76D5",
    "#2880A3",
    "#E62443",
    "#DF3623",
    "#6864F0",
    "#D92893",
    "#8D55F0",
    "#AC3DF0",
    "#426DF0",
    "#E1256D",
    "#CF2AB8",
    "#C12EDC",
    "#A86822",
    "#C55522",
    "#6D7E22",
    "#4F8422",
    "#318822",
    "#8A7622",
    "#2E73E2",
    "#25847C",
    "#278196",
    "#228848",
    "#238662",
    "#22892E",
    "#2B79C8",
    "#297EAF",
    "#E42B23",
    "#D84023",
    "#E5244F",
    "#E72337",
    "#A544F0",
    "#845AF0",
    "#C52DD3",
    "#CC2BC1",
    "#DC278A",
    "#5F67F0",
    "#7161F0",
    "#D22AAF",
    "#9550F0",
    "#D7289D",
    "#DF2676",
    "#4B6BF0",
)


def _distance_line_from_origin(line: tuple[float, float]) -> float:
    slope, intercept = line
    return abs(intercept) / math.sqrt(slope**2 + 1)


def _length_of_ray_until_intersect(theta: float, line: tuple[float, float]) -> float:
    slope, intercept = line
    return intercept / (math.sin(theta) - slope * math.cos(theta))


def _get_bounds(lightness: float) -> list[tuple[float, float]]:
    result = []
    sub1 = ((lightness + 16) ** 3) / 1560896
    sub2 = sub1 if sub1 > _EPSILON else lightness / _KAPPA

    for channel in range(3):
        m1, m2, m3 = _M[channel]
        for t in range(2):
            top1 = (284517 * m1 - 94839 * m3) * sub2
            top2 = (
                838422 * m3 + 769860 * m2 + 731718 * m1
            ) * lightness * sub2 - 769860 * t * lightness
            bottom = (632260 * m3 - 126452 * m2) * sub2 + 126452 * t
            result.append((top1 / bottom, top2 / bottom))

    return result


def _max_chroma_for_lh(lightness: float, hue: float) -> float:
    hue_radians = math.radians(hue)
    lengths = [
        _length_of_ray_until_intersect(hue_radians, bound)
        for bound in _get_bounds(lightness)
    ]
    return min(length for length in lengths if length >= 0)


def _dot_product(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=False)
    )


def _from_linear(channel: float) -> float:
    if channel <= 0.0031308:
        return 12.92 * channel
    return 1.055 * math.pow(channel, 5 / 12) - 0.055


def _l_to_y(lightness: float) -> float:
    if lightness <= 8:
        return _REF_Y * lightness / _KAPPA
    return _REF_Y * (((lightness + 16) / 116) ** 3)


def _lch_to_luv(color: tuple[float, float, float]) -> tuple[float, float, float]:
    lightness, chroma, hue = color
    hue_radians = math.radians(hue)
    return (
        lightness,
        math.cos(hue_radians) * chroma,
        math.sin(hue_radians) * chroma,
    )


def _luv_to_xyz(color: tuple[float, float, float]) -> tuple[float, float, float]:
    lightness, u_value, v_value = color
    if lightness == 0:
        return (0.0, 0.0, 0.0)

    var_u = u_value / (13 * lightness) + _REF_U
    var_v = v_value / (13 * lightness) + _REF_V
    y_value = _l_to_y(lightness)
    x_value = y_value * 9 * var_u / (4 * var_v)
    z_value = y_value * (12 - 3 * var_u - 20 * var_v) / (4 * var_v)
    return (x_value, y_value, z_value)


def _xyz_to_rgb(color: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(_from_linear(_dot_product(row, color)) for row in _M)


def _hsluv_to_lch(color: tuple[float, float, float]) -> tuple[float, float, float]:
    hue, saturation, lightness = color
    if lightness > 100 - 1e-7:
        return (100.0, 0.0, hue)
    if lightness < 1e-8:
        return (0.0, 0.0, hue)
    max_chroma = _max_chroma_for_lh(lightness, hue)
    return (lightness, max_chroma * saturation / 100, hue)


def hsluv_to_rgb(color: tuple[float, float, float]) -> tuple[float, float, float]:
    return _xyz_to_rgb(_luv_to_xyz(_lch_to_luv(_hsluv_to_lch(color))))


def rgb_to_hex(color: tuple[float, float, float]) -> str:
    channels = []
    for channel in color:
        channel = min(max(channel, 0.0), 1.0)
        channels.append(int(math.floor(channel * 255 + 0.5)))
    return "#{:02X}{:02X}{:02X}".format(*channels)


def hsluv_to_hex(color: tuple[float, float, float]) -> str:
    return rgb_to_hex(hsluv_to_rgb(color))


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def rgb_distance_hex(left: str, right: str) -> float:
    left_rgb = _hex_to_rgb(left)
    right_rgb = _hex_to_rgb(right)
    return math.dist(left_rgb, right_rgb)


@cache
def rich_label_palette(slot_count: int) -> tuple[str, ...]:
    if slot_count <= 64:
        return PRECOMPUTED_QUALITATIVE_HSLUV_PALETTE_64
    return tuple(qualitative_hsluv_palette(slot_count, lightness=0.5, saturation=0.9))


def qualitative_hsluv_palette(
    n_colors: int,
    *,
    hue: float = 0.01,
    saturation: float = 0.9,
    lightness: float = 0.5,
    hue_samples: int | None = None,
) -> list[str]:
    """Build a deterministic, high-separation qualitative HSLuv palette.

    Parameters mirror seaborn's ``husl_palette`` convention, using normalized
    floats in ``[0, 1]`` for hue, saturation, and lightness.
    """
    if n_colors <= 0:
        return []

    if hue_samples is None:
        hue_samples = max(48, n_colors * 32)

    start_degrees = (hue % 1.0) * 360.0
    candidates = [
        hsluv_to_hex(
            (
                (start_degrees + 360.0 * index / hue_samples) % 360.0,
                saturation * 100.0,
                lightness * 100.0,
            )
        )
        for index in range(hue_samples)
    ]
    candidates = list(dict.fromkeys(candidates))

    selected = [candidates[0]]
    while len(selected) < min(n_colors, len(candidates)):
        remaining = [candidate for candidate in candidates if candidate not in selected]
        best = max(
            remaining,
            key=lambda candidate: min(
                rgb_distance_hex(candidate, chosen) for chosen in selected
            ),
        )
        selected.append(best)

    return selected[:n_colors]
