PYTHON_INC=/home/shunting/cpython/build/install/include/python3.9d
CFLAGS=-Wno-pointer-arith

pymodule:
	rm -rf build/install
	mkdir -p build/install
	cp -r light build/install
	g++ light/csrc/*.cpp -shared -fPIC -I$(PYTHON_INC) -I. $(CFLAGS) -o build/install/light/_C.so

test:
	PYTHONPATH=build/install pytest -vs tests/test.py -k test_nn_forward

cpptest:
	g++ light/csrc/tests/TensorTest.cpp -Ilight/csrc -o /tmp/a.out -lgtest -lgtest_main $(CFLAGS)
	/tmp/a.out
