
import logging
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class Telemetry:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger = logging.getLogger("LeJEPA")
        self.logger.setLevel(logging.INFO)
        
        # Console Handler
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def log_metrics(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def info(self, msg):
        self.logger.info(msg)

    def close(self):
        self.writer.close()
