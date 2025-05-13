from dotenv import load_dotenv
from pyaml_env import parse_config
from patterns import Singleton


load_dotenv()


class Config(metaclass=Singleton):
    def __init__(self) -> None:
        """
        Constructor of the class
        """
        try:
            self.conf = parse_config("config.yml", tag=None, default_value=None)
        except FileNotFoundError:
            print("Warning: Configuration file was not found, hence not parsed!")


    

