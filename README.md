# 🛠️ Setup

You need:
- Python, more specifically the conda environment specified in the `environment.yml` file
- C++
- pybind11

The computationally heavy stuff is written in C++ in the file `src/maximum_likelihood.cpp`.
Compiling this code creates a library which can be called from Python code.
Said library lives inside lives inside the file `maximum_likelihood.cpython-311-x86_64-linux-gnu.so`.
If if it does not run on your machine, then compile it yourself by running
```
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maximum_likelihood.cpp -o ../maximum_likelihood$(python3-config --extension-suffix) $(gsl-config --cflags) $(gsl-config --libs)
```

The actual inference work for the main model we study is done in the file 'joint_infernce.py'.

In order to replicate the results, one should run `joint_inference.py`, then `plotting.py` and lastly the notebook `fit_check.ipynb`.
