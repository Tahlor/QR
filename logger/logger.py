import json


class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self, logger=None):
        self.entries = {}
        self.logger = logger
    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry
        if self.logger:
            print(entry)
            #self.logger.info(entry)

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
