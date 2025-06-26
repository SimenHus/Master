import sys
import time

class ProgressBar:
    def __init__(self, total, length=50, fill='█', empty='-', prefix='Progress:', suffix='Complete'):
        self.total = total
        self.length = length
        self.fill = fill
        self.empty = empty
        self.prefix = prefix
        self.suffix = suffix
        self.last_printed = -1  # To avoid duplicate prints

    def print(self, iteration):
        percent = 100 * (iteration / float(self.total))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + self.empty * (self.length - filled_length)
        if int(percent) != self.last_printed:
            sys.stdout.write(f'\r{self.prefix} |{bar}| {percent:.1f}% {self.suffix if iteration == self.total else ""}')
            sys.stdout.flush()
            self.last_printed = int(percent)
        if iteration == self.total:
            print()  # Newline on complete