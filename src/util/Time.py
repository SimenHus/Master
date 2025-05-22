from datetime import datetime, timezone

from enum import Enum


class TimeFormat(Enum):
    UTC = 0
    POSIX = 1
    STX = 2

class TimeConversion:
    UTC_POSIX_FACTOR = 1e6

    @staticmethod
    def UTC_to_POSIX(timestamp: str) -> int:
        return int(datetime.fromisoformat(timestamp).timestamp() * TimeConversion.UTC_POSIX_FACTOR)
    
    @staticmethod
    def POSIX_to_UTC(timestamp: int) -> str:
        return str(datetime.fromtimestamp(timestamp / TimeConversion.UTC_POSIX_FACTOR))
    
    @staticmethod
    def POSIX_to_STX(timestamp: int) -> str:
        utc = datetime.fromisoformat(TimeConversion.POSIX_to_UTC(timestamp))
        return datetime.strftime(utc, '%Y-%m-%d-%H_%M_%S_%f')
    
    @staticmethod
    def STX_to_POSIX(timestamp: str) -> int:
        utc = TimeConversion.STX_to_UTC(timestamp)
        return TimeConversion.UTC_to_POSIX(utc)

    @staticmethod
    def STX_to_UTC(timestamp: str) -> str:
        return str(datetime.strptime(timestamp, '%Y-%m-%d-%H_%M_%S_%f'))
    
    @staticmethod
    def identify_format(timestamp: str) -> TimeFormat:
        known_formats = {
            TimeFormat.UTC: '%Y-%m-%d %H:%M:%S.%f',
            # TimeFormat.POSIX: 
            TimeFormat.STX: '%Y-%m-%d-%H_%M_%S_%f'
        }
        for key, format in known_formats.items():
            try:
                datetime.strptime(timestamp, format)
                return key
            except ValueError:
                continue
        return None

    @staticmethod
    def generic_to_POSIX(timestamp: str) -> int:
        if isinstance(timestamp, int): timestamp = str(timestamp)
        utc = TimeConversion.generic_to_UTC(timestamp)
        return TimeConversion.UTC_to_POSIX(utc)

    @staticmethod
    def generic_to_UTC(timestamp: str) -> str:
        format = TimeConversion.identify_format(timestamp)
        match format:
            case TimeFormat.POSIX: return TimeConversion.POSIX_to_UTC(timestamp)
            case TimeFormat.UTC: return timestamp
            case TimeFormat.STX: return TimeConversion.STX_to_UTC(timestamp)