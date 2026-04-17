from bsr.utilities.singleton import SingletonMeta


class TempClass(metaclass=SingletonMeta):
    def __init__(self):
        self.value = None


class TempTheOtherClass(metaclass=SingletonMeta):
    def __init__(self):
        self.value = None


class TestSingletonMeta:
    def test_singleton_instance(self):
        obj1 = TempClass()
        obj2 = TempClass()

        # Both obj1 and obj2 should be the same instance
        assert obj1 is obj2

    def test_singleton_maintains_state(self):
        obj1 = TempClass()
        obj1.value = 1
        obj2 = TempClass()

        # obj2 should have the same value as obj1
        assert obj2.value == 1

    def test_different_subclasses_are_different_singletons(self):

        obj1 = TempClass()
        obj2 = TempTheOtherClass()

        # obj1 and obj2 should be different instances
        assert obj1 is not obj2
