#include <Python.h>
//Check which of the two can be removeda (almost sure one can)
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <numeric>
#include <map>
#include <armadillo>

using namespace std;
using namespace arma;

#ifndef CHANNELATTRIBUTION
#define CHANNELATTRIBUTION

static PyObject* heuristic_models_cpp(PyObject* self, PyObject* args);
static PyObject* markov_model_cpp(PyObject* self, PyObject* args);


//Mapping table
static PyMethodDef channelattribution_methods[] = {
  { "heuristic_models", (PyCFunction) heuristic_models_cpp, METH_VARARGS, NULL },
  { "markov_model", (PyCFunction) markov_model_cpp, METH_VARARGS, NULL },
  { NULL, NULL, 0, NULL }
};

//Module specs (Python3)
static struct PyModuleDef channelattribution_module = {
    PyModuleDef_HEAD_INIT,
    "channelattribution._libc",   /* name of module */
    "Python wrapper of a Markov chain method for channel attribution", /* module documentation, may be NULL */
    -1,       /* -1 if the module keeps state in global variables. */
    channelattribution_methods
};

//Initializer (Python 3)
PyMODINIT_FUNC PyInit__libc(void){
    import_array();
    return PyModule_Create(&channelattribution_module);
};

//Initializer (Python 2)
// void initChannelAttribution(void) {
//    Py_InitModule3("channelattribution", ChannelAttribution_methods,
//                   "Channel Attribution based on Markov Chains");
// }

template<typename T> static string to_string(T pNumber){
 ostringstream oOStrStream;
 oOStrStream << pNumber;
 return oOStrStream.str();
};

//Class for Partition Function
class Fx
{
 SpMat<unsigned long int> S;
 SpMat<unsigned long int> S0;
 SpMat<unsigned long int> S1;
 vector<unsigned long int> lrS0;
 vector<unsigned long int> lrS;
 unsigned long int non_zeros,nrows,val0,lval0,i,j,k,s0,lrs0i;

 public:
  Fx(unsigned long int nrow0,unsigned long int ncol0): S(nrow0,ncol0), S0(nrow0,ncol0), S1(nrow0,ncol0), lrS0(nrow0,0), lrS(nrow0,0), non_zeros(0), nrows(nrow0) {}
  void add(unsigned long int, unsigned long int,unsigned long int);
  void cum();
  unsigned long int sim(unsigned long int, double);
  PyObject* tran_matx(vector<string>);
};


#endif
