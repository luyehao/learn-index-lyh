import logging
from dataFS import DataFS
import config


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='KVsystem.log', level=logging.DEBUG, format=LOG_FORMAT)

class model:
    def train(self,X,Y):
        pass

    def predict(self,k):
        pass

    def load_model(self,k):
        pass

class KVsystem:
    def __init__(self):
        self.dataFS=DataFS()

    def build(self):
        pass
