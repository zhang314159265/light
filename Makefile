PYTHON_INC=/home/shunting/cpython/build/install/include/python3.9d
CFLAGS=-Wno-pointer-arith

pymodule:
	rm -rf build/install
	mkdir -p build/install
	cd build/install/ && ln -svf ../../light light
	g++ light/csrc/*.cpp -shared -fPIC -I$(PYTHON_INC) -I. $(CFLAGS) -o build/install/light/_C.so

test:
	# PYTHONPATH=build/install pytest -vs tests/test.py -k test_simple_backward
	# PYTHONPATH=build/install pytest -vs tests/test.py -k test_nn_backward
	# PYTHONPATH=build/install pytest -vs tests/test.py -k test_randint
	# PYTHONPATH=build/install pytest -vs tests/test.py -k test_classifier
	# PYTHONPATH=build/install pytest -vs tests/test.py -k test_linear
	PYTHONPATH=build/install pytest -vs tests/test.py -k test_max

# digit recognizer
dr:
	PYTHONPATH=build/install python3 model/digit_recognizer/mlp.py

cpptest:
	g++ light/csrc/tests/TensorTest.cpp -Ilight/csrc -o /tmp/a.out -lgtest -lgtest_main $(CFLAGS)
	/tmp/a.out
