import numpy as np
import configparser
import tensorflow as tf
from os.path import exists

class Parameters:
    """    Class to handle the parameters for the training configuration.
    It reads from a configuration file and provides access to the parameters.
    """
    def __init__(self, config_name, training=False, verbose=False):
        self.config_name = config_name
        self.training = training
        self.verbose = verbose
        self.params = self.load_config()
        
        # create parser object and read config file
        fn = f'../config/{config_name}'
        if exists(fn):
            config = configparser.RawConfigParser()
            # automatically add fileformat if needed
            config_name.replace(".cfg","") + ".cfg"
            config.read(fn)
        else:
            raise FileNotFoundError("Unknown config file.")

        # and import all parameters as attributes
        self.config_str = ""
        for section in config.sections():
            s = f"\n---- {section} ----- "
            self.config_str += s + "<br />" # add linebreak for Tensorboard
            if verbose:
                print(s)
            for option in config.options(section):
                setattr(self, f"{option}", eval(config.get(section,option)))
                s = f"{option}: {eval(config.get(section,option))}"
                self.config_str += s + "<br />" # add linebreak for Tensorboard
                if verbose:
                    print(s)

        