"""Hierarchie-Test fuer chamber_obb.errors (Story 6, AD-16).

Beweist, dass alle drei Subklassen auch ``ChamberOBBError`` sind, und dass
ein blankes ``except ChamberOBBError:`` jede von ihnen faengt -- das ist die
Garantie, auf der AD-16 beruht.
"""

import pytest

from chamber_obb.errors import (
    ChamberOBBError,
    InvalidImageError,
    InvalidParameterError,
    ModelLoadError,
)

SUBCLASSES = [InvalidImageError, ModelLoadError, InvalidParameterError]


@pytest.mark.parametrize("exc_type", SUBCLASSES)
def test_subclass_is_a_chamber_obb_error(exc_type):
    """Jede Subklasse ist eine Instanz von ChamberOBBError."""
    instance = exc_type("boom")
    assert isinstance(instance, ChamberOBBError)


@pytest.mark.parametrize("exc_type", SUBCLASSES)
def test_bare_except_chamber_obb_error_catches_every_subclass(exc_type):
    """``except ChamberOBBError:`` faengt jede Subklasse -- der AD-16-Vertrag."""
    caught = None
    try:
        raise exc_type("boom")
    except ChamberOBBError as exc:
        caught = exc

    assert isinstance(caught, exc_type)


def test_chamber_obb_error_does_not_inherit_from_value_error():
    """Bewusste Abkehr von bare ValueError (Boundaries): kein Dual-Inheritance."""
    assert not issubclass(ChamberOBBError, ValueError)
