from datetime import datetime

class TimeConversion:

    @staticmethod
    def UTC_to_POSIX(timestamp: str) -> int:
        return int(datetime.fromisoformat(timestamp).timestamp() * 1e6)