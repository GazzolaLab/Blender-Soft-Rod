from typing_extensions import Self


class Singleton:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance  # type: ignore

    @property
    def isInstantiated(self):
        return self.__instance is not None
