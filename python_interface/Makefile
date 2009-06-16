HEADERS = $(shell find .. -name "*h")
CPPS = $(shell find .. -name "*cpp" | grep -v "main.cpp")

all:	horus.py horus_wrap.cxx $(CPPS)
	python setup.py install

horus_wrap.cxx horus.py:	$(HEADERS) horus.i
	swig -c++ -python horus.i

clean:
	rm horus.py || rm horus_wrap.cxx || python setup.py clean || rm -rf build/
