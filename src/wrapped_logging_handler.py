import logging

class WrappedLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        msg = self.format(record)
        stream = self.stream
        stream.write(msg)
        self.flush()
