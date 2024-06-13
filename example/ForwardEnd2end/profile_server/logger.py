import json

class Logger(object):
    def __init__(self, log_path):
        self.event_id = 0
        self._event_start_log = dict()
        self._event_end_log = dict()
        self.event_log = dict()
        self.log_path = log_path

    def tick_start(self, name, timestamp) -> int: #id
        event_id = self.event_id
        self._event_start_log[event_id] = {"event_name": name, "start_timestamp" :timestamp}
        self.event_id += 1
        return event_id

    def tick_end(self, event_id, timestamp):
        self._event_end_log[event_id] = timestamp
        self.event_log[event_id] = self._event_start_log[event_id]
        self.event_log[event_id]["env_timestamp"] = timestamp
        self.event_log[event_id]["during"] = str(round(self._event_end_log[event_id]-
                                                    self._event_start_log[event_id]["start_timestamp"], 3)) + " s"

    def save(self):
        with open(self.log_path, "a") as f:
            f.write("\n")
            json.dump(self.event_log, f)

    def log_kv(self, key, value):
        if self.event_log.get(key) == None:
            self.event_log[key] = value
        else:
            if isinstance(self.event_log[key], list):
                self.event_log[key].append(value)
            else :
                self.event_log[key] = [self.event_log[key]]