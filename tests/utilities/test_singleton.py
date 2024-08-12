import pytest

from bsr.utilities.singleton import Singleton


class TestClass(Singleton):
    def __init__(self):
        if not self.isInstantiated:
            self.value = None


class TestTheOtherClass(Singleton):
    def __init__(self):
        if not self.isInstantiated:
            self.value = None


class TestSingletonMixin:
    def test_singleton_instance(self):
        obj1 = TestClass()
        obj2 = TestClass()  # 20 should be ignored

        # Both obj1 and obj2 should be the same instance
        assert obj1 is obj2

    def test_singleton_maintains_state(self):
        obj1 = TestClass()
        obj1.value = 1
        obj2 = TestClass()

        # obj2 should have the same value as obj1
        assert obj2.value == 1

    def test_different_subclasses_are_different_singletons(self):

        obj1 = TestClass()
        obj2 = TestTheOtherClass()

        # obj1 and obj2 should be different instances
        assert obj1 is not obj2
