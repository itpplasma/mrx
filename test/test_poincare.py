"""The field-line tracer of :mod:`mrx.diagnostics.poincare`, on the equilibrium's own field.

``b0`` (:func:`mrx.relaxation.initial_conditions.potential_two_form`) is the curl of a vector potential whose
``theta`` and ``zeta`` components depend on ``r`` alone, so ``B^r = 0`` exactly: its field lines stay on their
logical surfaces and wind with the rotational transform of the file. The traced iota of every regular line
matches ``|iota(r)|`` of the file at the line's radius within a measured band, the discretisation error of the
field, and halving the step moves the lines by less than a measured drift.
"""
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np

from mrx.diagnostics.poincare import logical_field, poincare, trace

#: 1.25x ``max |iota_traced - |iota_file|| / max |iota_file|`` on li383 (16, 32, 12) p=2, 8 lines over 60 periods,
#: measured 1.06e-3 (2026-10-20). A tracer that missed the factor nfp or a wrong winding would be off by O(1).
IOTA_BAND = 1.33e-3
#: 1.25x the step-halving drift of the same trace in logical radius, measured 3.23e-4.
DRIFT_BAND = 4.1e-4


def test_traced_iota_is_the_file_iota(seq, b0):
    """The traced iota of the regular lines is the file's iota at their radius, and the step is resolved."""
    res = poincare(seq, b0, lines=8, periods=60, planes=1)
    shown = np.asarray(res["shown"])
    r = np.asarray(res["seed_r"])[shown]
    want = np.abs(np.asarray(seq.equilibrium["profiles"]["iota"](r)))
    err = float(np.abs(np.asarray(res["iota"])[shown] - want).max() / want.max())
    print(f"\n  {int(shown.sum())} of {shown.size} lines regular, iota {want.min():.4f}..{want.max():.4f}, "
          f"relative error {err:.2e}, drift {float(res['drift']):.2e}")
    assert shown.all()                  # the field is integrable: no line is chaotic or leaves the domain
    assert err < IOTA_BAND
    assert float(res["drift"]) < DRIFT_BAND


def _primitives(jaxpr, inside_scan=False, found=None):
    """The names of the primitives of ``jaxpr`` that run inside a ``scan``, the step loop of the tracer."""
    found = set() if found is None else found
    for eqn in jaxpr.eqns:
        if inside_scan:
            found.add(eqn.primitive.name)
        for sub in jax.extend.core.jaxprs_in_params(eqn.params):
            _primitives(sub, inside_scan or eqn.primitive.name == "scan", found)
    return found


def test_trace_applies_the_extraction_once(seq, b0):
    """The step loop of the tracer only evaluates the splines: the extraction ``E^T dof``, a scatter-add over
    all DoFs, runs once before it. The lines are those of the per-step extraction to round-off."""
    field, seeds = logical_field(seq), jnp.array([[0.3, 0.0], [0.6, 0.25], [0.9, 0.5]])
    loop = _primitives(jax.make_jaxpr(lambda d: trace(field, d, seeds, 2, 300))(b0).jaxpr)
    assert "scatter-add" not in loop, sorted(loop)
    ys = trace(field, b0, seeds, 2, 300)
    per_step = trace(lambda x, d: field(x, d), b0, seeds, 2, 300)
    assert float(jnp.max(jnp.abs(ys - per_step))) < 1e-12
