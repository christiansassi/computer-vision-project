import logging
import sys

class WrappedLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        msg = self.format(record)
        stream = self.stream
        stream.write(msg)
        self.flush()

def get_logger(level: int = logging.INFO) -> logging.Logger:

    logger = logging.getLogger()
    logger.setLevel(level=level)
    
    handler = WrappedLoggingHandler(sys.stdout)
    handler.setFormatter(fmt=logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    logger.addHandler(handler)

    return logger