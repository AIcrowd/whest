"""Focused Rich-render regression tests for PathInfo."""

from __future__ import annotations

import io
import re

import numpy as np
import pytest
from rich.console import Console

import whest as we
from whest._perm_group import PermutationGroup

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def render_rich(info, *, columns: int = 132, no_color: bool = False) -> str:
    buf = io.StringIO()
    Console(
        file=buf,
        force_terminal=True,
        no_color=no_color,
        _environ={"COLUMNS": str(columns), "LINES": "40"},
    ).print(info)
    return ANSI_RE.sub("", buf.getvalue())


def render_verbose_rich(info, *, columns: int = 132, no_color: bool = False) -> str:
    buf = io.StringIO()
    Console(
        file=buf,
        force_terminal=True,
        no_color=no_color,
        _environ={"COLUMNS": str(columns), "LINES": "40"},
    ).print(info._rich_renderable(verbose=True))
    return ANSI_RE.sub("", buf.getvalue())


def _style_at(text, needle: str, start: int = 0):
    start = text.plain.index(needle, start)
    for span in text.spans:
        if span.start <= start < span.end:
            return span.style
    raise AssertionError(f"no span covers {needle!r} in {text.plain!r}")


def _symmetrized_tensor(shape, group: PermutationGroup):
    data = np.random.RandomState(0).randn(*shape)
    axes = group.axes if group.axes is not None else tuple(range(group.degree))
    total = np.zeros_like(data)
    for g in group.elements():
        af = g.array_form
        perm = list(range(len(shape)))
        for i in range(len(axes)):
            perm[axes[i]] = axes[af[i]]
        total = total + np.transpose(data, perm)
    return we.as_symmetric(total / group.order(), symmetry=group)


def _summary_pill_body(info, label: str):
    summary = info._rich_summary_strip()
    for pill in summary.renderables:
        body = pill.renderable
        if body.plain.startswith(f"{label}: "):
            return body
    raise AssertionError(f"no summary pill found for {label!r}")


def _rgb_from_style(style: str):
    match = re.search(r"#([0-9A-Fa-f]{6})", style)
    if match is None:
        raise AssertionError(f"style {style!r} does not contain a hex color")
    hex_value = match.group(1)
    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))


def test_rich_renderable_uses_a_single_composed_block():
    X = np.ones((5, 5))
    _, info = we.einsum_path("ij,jk,kl->il", X, X, X, optimize="greedy")

    renderable = info._rich_renderable()
    assert len(renderable.renderable.renderables) == 3


def test_label_styles_are_consistent_between_expression_and_step_text():
    X = np.ones((4, 4))
    _, info = we.einsum_path("ij,ik->jk", X, X)

    expr = info._rich_eq_text()
    subscript = info._rich_subscript_text(info.steps[0].subscript)

    assert expr.plain.endswith("ij,ik->jk")
    assert subscript.plain == info.steps[0].subscript
    assert _style_at(expr, "j") == _style_at(subscript, "j")
    assert _style_at(expr, "k") == _style_at(subscript, "k")


def test_active_labels_do_not_collide_within_one_expression(d4_case_info):
    from itertools import combinations
    from math import dist

    expr = d4_case_info._rich_subscript_text(d4_case_info.eq)
    labels = "abijkl"
    styles = {label: _style_at(expr, label) for label in labels}
    rgbs = {label: _rgb_from_style(style) for label, style in styles.items()}

    assert len(set(styles.values())) == len(labels)
    assert (
        min(dist(rgbs[left], rgbs[right]) for left, right in combinations(labels, 2))
        > 80
    )


def test_symmetry_class_styles_are_consistent_on_real_cases():
    x = np.ones((4, 4))
    a = np.ones((4, 4))

    _, s2_gram = we.einsum_path("ij,ik->jk", x, x)
    v = np.ones(4)
    _, s2_outer = we.einsum_path("i,j->ij", v, v)
    _, c3_trace = we.einsum_path("ij,jk,ki->", a, a, a)

    d4_group = PermutationGroup.dihedral(4, axes=(1, 2, 3, 4))
    t = _symmetrized_tensor((4, 4, 4, 4, 4), d4_group)
    w = np.ones((4, 4))
    _, d4_case = we.einsum_path("aijkl,ab->ijklb", t, w)

    gram_sym = s2_gram._rich_step_sym_text(s2_gram.steps[0])
    outer_sym = s2_outer._rich_step_sym_text(s2_outer.steps[0])
    trace_sym = c3_trace._rich_step_sym_text(c3_trace.steps[-1])
    d4_sym = d4_case._rich_step_sym_text(d4_case.steps[0])

    assert "S2" in gram_sym.plain
    assert "S2" in outer_sym.plain
    assert "W:" in trace_sym.plain
    assert "C3" in trace_sym.plain
    assert "D4" in d4_sym.plain
    assert _style_at(gram_sym, "S2") == _style_at(outer_sym, "S2")


def test_verbose_rich_print_uses_rich_layout_and_keeps_detail_rows(capsys):
    pytest.importorskip("rich")

    x = np.ones((4, 4))
    _, info = we.einsum_path("ai,bi,ci->abc", x, x, x)

    info.print(verbose=True)
    out = capsys.readouterr().out

    assert "subset=" in out
    assert "out_shape=" in out
    assert "cumulative=" in out
    assert "┏" in out


def test_summary_pills_are_single_line_and_keep_all_default_fields():
    X = np.ones((5, 5))
    _, info = we.einsum_path("ij,jk,kl->il", X, X, X, optimize="greedy")

    output = render_rich(info, no_color=True)
    for field in (
        "Complete contraction: ij,jk,kl->il",
        "Naive cost:",
        "Optimized cost:",
        "Speedup:",
        "Savings:",
        "Largest intermediate:",
        "Index sizes:",
        "Optimizer:",
    ):
        assert field in output


def test_speedup_pill_turns_green_when_speedup_is_above_one():
    X = np.ones((5, 5))
    _, info = we.einsum_path("ij,jk,kl->il", X, X, X, optimize="greedy")

    body = _summary_pill_body(info, "Speedup")
    value_start = body.plain.index(": ") + 2

    assert body.plain == "Speedup: 5.000x"
    assert _style_at(body, "5.000x", value_start) == "bold green"


def test_savings_pill_shows_total_dense_vs_optimized_savings():
    X = np.ones((5, 5))
    _, info = we.einsum_path("ij,jk,kl->il", X, X, X, optimize="greedy")

    body = _summary_pill_body(info, "Savings")

    assert body.plain == "Savings: 80.0%"


def test_index_sizes_pill_preserves_label_styles():
    X = np.ones((5, 5))
    _, info = we.einsum_path("ij,jk,kl->il", X, X, X, optimize="greedy")
    expr = info._rich_subscript_text(info.eq)

    pill = info._rich_metric_pill("Index sizes", info._rich_index_sizes_text())
    body = pill.renderable
    values_start = body.plain.index(": ") + 2

    assert body.plain == "Index sizes: i=j=k=l=5"
    assert _style_at(body, "i", values_start) == _style_at(expr, "i")
    assert _style_at(body, "l", values_start) == _style_at(expr, "l")


@pytest.fixture
def d4_case_info():
    d4_group = PermutationGroup.dihedral(4, axes=(1, 2, 3, 4))
    tensor = _symmetrized_tensor((4, 4, 4, 4, 4), d4_group)
    _, info = we.einsum_path("aijkl,ab->ijklb", tensor, np.ones((4, 4)))
    return info


def test_real_d4_case_keeps_critical_headers_and_values_unbroken(d4_case_info):
    output = render_rich(d4_case_info, no_color=True)

    assert "contract" in output
    assert "blas" in output
    assert "unique/total" in output
    assert "SYMM" in output
    assert "V:220/1,024" in output
    assert "- × D4{i,j,k,l} → D4{i,j,k,l}" in output


def test_default_rich_output_does_not_show_verbose_detail_rows():
    X = np.ones((4, 4))
    _, info = we.einsum_path("ai,bi,ci->abc", X, X, X)

    output = render_rich(info, no_color=True)

    assert "subset=" not in output
    assert "out_shape=" not in output
    assert "cumulative=" not in output


def test_verbose_rich_details_stay_with_their_step():
    X = np.ones((4, 4))
    _, info = we.einsum_path("ai,bi,ci->abc", X, X, X)

    output = render_verbose_rich(info, no_color=True)

    assert output.index("bi,ai->bia") < output.index("step 0:")
    assert output.index("step 0:") < output.index("bia,ci->abc")
    assert "out_shape=" in output
    assert "cumulative=" in output
