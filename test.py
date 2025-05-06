


class Common:
    a: int
    b: int
    c = 5

class A(Common):

    def __init__(self):
        self.a = 6
        self.c = 3

class B(Common):

    def __init__(self):
        self.b = 7


first = A()
second = B()


# print(Common.a, Common.b, Common.c)
print(Common.c)
print(Common.a)