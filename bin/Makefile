LIBS = `pkg-config --libs opencv` -lhorus -lboost_system -lboost_filesystem

horus-cli:	cli.cpp
	$(CXX) -g -std=c++11 -o horus-cli cli.cpp $(LIBS)
