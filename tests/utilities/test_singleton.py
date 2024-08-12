import pytest

from bsr.utilities import Singleton


class TestClass(SingletonMixin):
    def __init__(self, value):
        self.value = value


class TestSingletonMixin:
    def test_singleton_instance(self):
        obj1 = TestClass(10)
        obj2 = TestClass(20)  # 20 should be ignored

        # Both obj1 and obj2 should be the same instance
        assert obj1 is obj2

        # The value should remain as the first initialized value
        assert obj1.value == 10
        assert obj2.value == 10
