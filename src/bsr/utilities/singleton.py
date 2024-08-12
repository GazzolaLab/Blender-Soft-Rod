from typing_extensions import Self


class Singleton:
    __instance = None

    def __new__(cls):  # type: ignore
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @property
    def isInstantiated(self) -> bool:
        return self.__instance is not None
