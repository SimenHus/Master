from datetime import datetime, timezone

from enum import Enum


class TimeFormat(Enum):
    UTC = 0
    POSIX = 1
    STX = 2

class TimeConversion:
    'Posix resolution in ms'
    UTC_POSIX_FACTOR = 1e6

    @classmethod
    def dt_POSIX_to_SECONDS(clc, dt_posix: int) -> float:
        return dt_posix / clc.UTC_POSIX_FACTOR

    @classmethod
    def UTC_to_POSIX(clc, timestamp: str) -> int:
        return int(datetime.fromisoformat(timestamp).timestamp() * clc.UTC_POSIX_FACTOR)
    
    @classmethod
    def POSIX_to_UTC(clc, timestamp: int) -> str:
        return str(datetime.fromtimestamp(timestamp / clc.UTC_POSIX_FACTOR))
    
    @classmethod
    def POSIX_to_STX(clc, timestamp: int) -> str:
        utc = datetime.fromisoformat(clc.POSIX_to_UTC(timestamp))
        return datetime.strftime(utc, '%Y-%m-%d-%H_%M_%S_%f')
    
    @classmethod
    def STX_to_POSIX(clc, timestamp: str) -> int:
        utc = clc.STX_to_UTC(timestamp)
        return clc.UTC_to_POSIX(utc)

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

    @classmethod
    def generic_to_POSIX(clc, timestamp: str) -> int:
        if isinstance(timestamp, int): timestamp = str(timestamp)
        utc = clc.generic_to_UTC(timestamp)
        return clc.UTC_to_POSIX(utc)

    @classmethod
    def generic_to_UTC(clc, timestamp: str) -> str:
        format = clc.identify_format(timestamp)
        match format:
            case TimeFormat.POSIX: return clc.POSIX_to_UTC(timestamp)
            case TimeFormat.UTC: return timestamp
            case TimeFormat.STX: return clc.STX_to_UTC(timestamp)