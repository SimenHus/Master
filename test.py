from dataclasses import dataclass


@dataclass
class Factor:
    identifier: str
    dingdon: int
    index = -1

    def __eq__(self, other) -> bool:
        if not isinstance(other, Factor): return False
        return self.identifier == other.identifier

    def __hash__(self) -> int:
        return hash(self.identifier)
    

a = Factor('ligma', 2)
b = Factor('lonkma', 3)

myset = set()
myset.add(a)
myset.add(b)

print(Factor('ligma', 2) in myset)
print(b in myset)