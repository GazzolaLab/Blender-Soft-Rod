import numpy as np
import pytest

from bsr.geometry.composite.stack import BaseStack


class MockObjectToStack:
    def __init__(self, value):
        self.value = value

    @property
    def object(self):
        return self.value

    @classmethod
    def create(cls, states):
        return cls(states["value"])

    def update_states(self, value):
        self.value = value


class MockStack(BaseStack[MockObjectToStack]):
    DefaultType = MockObjectToStack


class MockStackStack(BaseStack[MockStack]):
    DefaultType = MockStack


def test_stack():
    stack = MockStack.create({"value": np.array([1, 2, 3])})

    assert len(stack) == 3
    assert stack[0].object == 1
    assert stack[1].object == 2
    assert stack[2].object == 3


def test_recursive_stack():
    stack = MockStackStack.create(
        {
            "value": np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            )
        },
    )

    assert len(stack) == 2
    assert stack[0][0].object == 1
    assert stack[0][1].object == 2
    assert stack[0][2].object == 3
    assert stack[1][0].object == 4
    assert stack[1][1].object == 5
    assert stack[1][2].object == 6

    stack.update_states(np.ones((2, 3)))

    assert stack[0][0].object == 1
    assert stack[0][1].object == 1
    assert stack[0][2].object == 1
    assert stack[1][0].object == 1
    assert stack[1][1].object == 1
    assert stack[1][2].object == 1


@pytest.mark.parametrize(
    "update_data",
    [
        np.ones(2),
        np.ones((2, 3)),
        np.ones(4),
        np.ones((4, 3)),
    ],
)
def test_update_wrong_size(update_data):
    stack = MockStack.create(
        {
            "value": np.array(
                [1, 2, 3],
            )
        },
    )

    assert len(stack) == 3
    with pytest.raises(IndexError):
        stack.update_states(update_data)


class MockObjectToStack2:
    def __init__(self, value1, value2):
        self.value = (value1, value2)

    @property
    def object(self):
        return self.value

    @classmethod
    def create(cls, states):
        return cls(states["value1"], states["value2"])

    def update_states(self, value1, value2):
        self.value = (value1, value2)


class Mock2Stack(BaseStack[MockObjectToStack2]):
    DefaultType = MockObjectToStack2


def test_stack2():
    stack = Mock2Stack.create(
        {"value1": np.array([1, 2, 3]), "value2": np.array([4, 5, 6])}
    )

    assert len(stack) == 3
    assert stack[0].object == (1, 4)
    assert stack[1].object == (2, 5)
    assert stack[2].object == (3, 6)

    stack.update_states(np.ones(3), np.ones(3) * 2)

    assert stack[0].object == (1, 2)
    assert stack[1].object == (1, 2)
    assert stack[2].object == (1, 2)


def test_stack2_wrong_size():
    stack = Mock2Stack.create(
        {"value1": np.array([1, 2, 3]), "value2": np.array([4, 5, 6])}
    )

    assert len(stack) == 3
    with pytest.raises(IndexError):
        stack.update_states(np.ones(2), np.ones(2) * 2)
