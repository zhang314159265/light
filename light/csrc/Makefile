PYTHON_INC=/home/shunting/cpython/build/install/include/python3.9d
CFLAGS=-Wno-pointer-arith

pymodule:
	mkdir -p /tmp/myinstall/light
	g++ pymodule.cpp -shared -fPIC -I$(PYTHON_INC) $(CFLAGS) -o /tmp/myinstall/light/_C.so

all:
	g++ tests/TensorTest.cpp -I. -o /tmp/a.out -lgtest -lgtest_main $(CFLAGS)
	/tmp/a.out
