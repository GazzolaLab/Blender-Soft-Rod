import pytest

from bsr.utilities.singleton import Singleton


class TestClass(Singleton):
    def __init__(self):
        pass


class TestSingletonMixin:
    def test_singleton_instance(self):
        obj1 = TestClass()
        obj2 = TestClass()  # 20 should be ignored

        # Both obj1 and obj2 should be the same instance
        assert obj1 is obj2
