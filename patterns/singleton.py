"""
Metaclass for singleton
"""


class Singleton(type):
    """
    It will be used as a metaclass to create singleton class
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        method docstring
        """

        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
