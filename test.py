


class A:

    def __init__(self):
        self.a = 10

    def change(self):
        self.a = 4

myset = set()
myset.add(A())

for item in myset:
    print(item.a)
    item.change()

for item in myset: print(item.a)