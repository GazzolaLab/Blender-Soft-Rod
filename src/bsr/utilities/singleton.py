class SingletonMeta(type):
    __instance: dict = {}

    def __call__(cls, *args, **kwargs):  # type: ignore
        if cls not in cls.__instance:
            cls.__instance[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls.__instance[cls]
