"""
Unit tests of the module tunax.Grid.

"""

from tunax import Grid


def test_find_index():
    """
    Unit test of the method Grid.find_index.
    """
    grid = Grid.linear(50, 50)
    assert grid.find_index(25) == 25
