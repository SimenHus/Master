

class A:
    tings = 5

    def __init__(self):
        self.tings = 10

    @classmethod
    def myfunc(clc):
        print(clc.tings)

slong = A()

A.myfunc()
slong.myfunc()