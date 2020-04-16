cc=c++

#Python2
#flags=-O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` -I/usr/include/python2.7 -Iinclude
#exec_py=.so

#Python 3
flags=-O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -Iinclude

exec_py=`python3-config --extension-suffix` 

code=src/bind.cc
bin=wnn

all:
	$(cc) $(flags) $(code) -o $(bin)$(exec_py)
