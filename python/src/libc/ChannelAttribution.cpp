#include "ChannelAttribution.h"

using namespace std;
using namespace arma;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static PyObject* heuristic_models_cpp(PyObject* self, PyObject* args)
// RcppExport SEXP heuristic_models_cpp(SEXP Data_p,
//    SEXP var_path_p, SEXP var_conv_p, SEXP var_value_p)
{

 // BEGIN_RCPP

/*
Input parameter conversion
*/
 // char* var_path_c;
 // char* var_conv_c;
 // char* var_value_c;

 PyArrayObject* vy0, *vc0, *vv0;
 bool flg_var_value;

 // if (!PyArg_ParseTuple(args, "O!O!O!O!sss",
 //   &PyArray_Type, &vy0,
 //    &PyArray_Type, &vc0,
 //     &PyArray_Type,  &vv0,
 //      &var_path_c, &var_conv_c, &var_value_c)) {
 //      return NULL;
 //   }

 if (!PyArg_ParseTuple(args, "O!O!O!O!b",
   &PyArray_Type, &vy0,
    &PyArray_Type, &vc0,
     &PyArray_Type,  &vv0,
      &flg_var_value)) {
      return NULL;
   }

 // std::string var_value(var_value_c);
 // std::string var_path(var_path_c);
 // std::string var_conv(var_conv_c);

//  bool flg_var_value;
//  flg_var_value=0;
//  if(var_value.compare("0")!=0){
//   flg_var_value=1;
// };

npy_intp vy_n = vy0->dimensions[0];
npy_intp vv_n = vv0->dimensions[0];
npy_intp vc_n = vc0->dimensions[0];

string* vy1 = static_cast<string*>(PyArray_DATA(vy0));
double* vv1 = static_cast<double*>(PyArray_DATA(vv0));
long long int* vc1 =  static_cast<long long int*>(PyArray_DATA(vc0));

/*Make vectors*/
vector<double> vv;
if (flg_var_value){
  vv.assign(vv1, vv1+vv_n);
}
vector<long long int> vc(vc1, vc1+vc_n);
vector<string> vy(vy1, vy1+vy_n);

 int i,j,k;
 long long int lvy,ssize;
 long long int nchannels;
 string s,channel,channel_first,channel_last;

 lvy=(long long int) vy.size();
 nchannels=0;

 map<string,long long int> mp_channels;
 vector<string> vchannels;

 map<string,double> mp_first_conv;
 map<string,double> mp_first_val;
 map<string,double> mp_last_conv;
 map<string,double> mp_last_val;
 map<string,double> mp_linear_conv;
 map<string,double> mp_linear_val;
 map<string,double> mp0_linear_conv;
 map<string,double> mp0_linear_val;

 vector<string> vchannels_unique;
 double nchannels_unique;
 string kchannel;
 long long int n_path_length;

 for(i=0;i<lvy;i++){

  s=vy[i];

  s+=" >";
  ssize=(long long int) s.size();
  channel="";
  j=0;
  nchannels_unique=0;
  vchannels_unique.clear();

  n_path_length=0;
  mp0_linear_conv.clear();
  mp0_linear_val.clear();

  while(j<ssize){

    if((j>0) & (ssize>1)){

      if((s[j-1]=='>') & (s[j]==' ')){
	   j=j+1;
	  }
	  if((s[j]==' ') & (s[j+1]=='>')){
	   j=j+2;
	   break;
	  }

    }

    while((s[j]!=' ') & (s[j+1]!='>')){
	  channel+=s[j];
      ++j;
    }
    ++j;

    if(mp_channels.find(channel) == mp_channels.end()){
	 mp_channels[channel]=nchannels;
	 vchannels.push_back(channel);
	 ++nchannels;

   mp_first_conv[channel]=0;
	 mp_last_conv[channel]=0;
	 mp_linear_conv[channel]=0;
	 mp0_linear_conv[channel]=0;

	 if(flg_var_value==1){
	  mp_first_val[channel]=0;
	  mp_last_val[channel]=0;
	  mp_linear_val[channel]=0;
	  mp0_linear_val[channel]=0;
	 }

	}

    //lista canali unici
    if(nchannels_unique==0){
      vchannels_unique.push_back(channel);
	    ++nchannels_unique;
    }
    else if(find(vchannels_unique.begin(),vchannels_unique.end(),channel)==vchannels_unique.end()){
  	 vchannels_unique.push_back(channel);
  	 ++nchannels_unique;
    }

 	  mp0_linear_conv[channel]=mp0_linear_conv[channel]+vc[i];
    if(flg_var_value==1){
	     mp0_linear_val[channel]=mp0_linear_val[channel]+vv[i];
    }
	++n_path_length;

    channel_last=channel;

    channel="";
    ++j;

  }//end while j

  channel_first=vchannels_unique[0];
  mp_first_conv[channel_first]=mp_first_conv[channel_first]+vc[i];
  mp_last_conv[channel_last]=mp_last_conv[channel_last]+vc[i];

  //linear
  for(k=0;k<nchannels_unique;k++){
    kchannel=vchannels_unique[k];
    mp_linear_conv[kchannel]=mp_linear_conv[kchannel]+(mp0_linear_conv[kchannel]/n_path_length);
  }

  if(flg_var_value==1){
   mp_first_val[channel_first]=mp_first_val[channel_first]+vv[i];
   mp_last_val[channel_last]=mp_last_val[channel_last]+vv[i];
   for(k=0;k<nchannels_unique;k++){
    kchannel=vchannels_unique[k];
    mp_linear_val[kchannel]=mp_linear_val[kchannel]+(mp0_linear_val[kchannel]/n_path_length);
   }
  }


 }//end for i

//Depending of what has to be returned I might allocate certain arrays or not
double *vfirst_conv, *vlinear_conv, *vlast_conv,
 *vfirst_val, *vlast_val, *vlinear_val;

vfirst_conv = new double[nchannels];
vlinear_conv = new double[nchannels];
vlast_conv = new double[nchannels];

if (flg_var_value){
  vfirst_val = new double[nchannels];
  vlast_val = new double[nchannels];
  vlinear_val = new double[nchannels];
}

for(k=0;k<nchannels;k++){
  kchannel=vchannels[k];

  vfirst_conv[k]=mp_first_conv[kchannel];
  vlast_conv[k]=mp_last_conv[kchannel];
  vlinear_conv[k]=mp_linear_conv[kchannel];

  if(flg_var_value){
    vfirst_val[k]=mp_first_val[kchannel];
    vlast_val[k]=mp_last_val[kchannel];
    vlinear_val[k]=mp_linear_val[kchannel];
  };
}

 PyObject *vfirst_conv_np, *vlinear_conv_np, *vlast_conv_np,
  *vfirst_val_np, *vlinear_val_np, *vlast_val_np;

// Construct numpy arrays for ourtput
npy_intp* dims=new npy_intp[1];
dims[0]=nchannels;

vfirst_conv_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, vfirst_conv);
vlast_conv_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, vlast_conv);
vlinear_conv_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, vlinear_conv);

if (flg_var_value){
  vfirst_val_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, vfirst_val);
  vlast_val_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, vlast_val);
  vlinear_val_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, vlinear_val);
};

delete [] vfirst_conv;
delete [] vlinear_conv;
delete [] vlast_conv;
if (flg_var_value){
  delete [] vfirst_val;
  delete [] vlinear_val;
  delete [] vlast_val;
};

//Build a numpy array os strings out of vchannels_char (a bit of a boilerplate)
const char **vchannels_char = new const char*[vchannels.size()];
string* app = vchannels.data();
int app_n = vchannels.size();
for (i=0; i<app_n; i++){
  vchannels_char[i] = app[i].c_str();
}
dims[0] = vchannels.size();
PyObject* vchannels_np = PyArray_SimpleNewFromData(1, dims, NPY_STRING, vchannels_char);

delete [] dims;
delete [] vchannels_char;

if(flg_var_value==1)
  return Py_BuildValue("{s:O, s:O, s:O, s:O, s:O, s:O, s:O}",
  "channel_name", vchannels_np,
   "first_touch_conversions", vfirst_conv,
    "first_touch_value", vfirst_val_np,
     "last_touch_conversions", vlast_conv_np,
      "last_touch_value", vlast_val_np,
       "linear_touch_conversions", vlinear_conv_np,
        "linear_touch_value", vlinear_val_np
      );
 else
   return Py_BuildValue("{s:O, s:O, s:O, s:O, s:O}",
      "channel_from", vchannels_np,
       "first_touch", vfirst_conv_np,
        "last_touch", vlast_conv_np,
         "linear_touch", vlinear_conv_np
       );
}//end heuristic_models_cpp

void Fx::add(unsigned long int ichannel_old, unsigned long int ichannel, unsigned long int vxi)
{
  val0=S(ichannel_old,ichannel); //riempire f.p. transizione con vxi
  if(val0==0){
   lval0=lrS0[ichannel_old];
   S0(ichannel_old,lval0)=ichannel;
   lrS0[ichannel_old]=lval0+1;
   ++non_zeros;
  }
  S(ichannel_old,ichannel)=val0+vxi;
}

void Fx::cum()
{

 for(i=0;i<nrows;i++){
  lrs0i=lrS0[i];
  if(lrs0i>0){
   S1(i,0)=S(i,S0(i,0));
   for(j=1;j<lrs0i;j++){
    S1(i,j)=S1(i,j-1)+S(i,S0(i,j));
   }
   lrS[i]=S1(i,lrs0i-1);
  }
 }

}


unsigned long int Fx::sim(unsigned long int c, double uni)
{

 s0=floor(uni*lrS[c]+1);

 for(k=0; k<lrS0[c]; k++){
  if(S1(c,k)>=s0){return(S0(c,k));}
 }

 return 0;

}


PyObject* Fx::tran_matx(vector<string> vchannels)
{

 unsigned long int mij,sm3;
 // I use arrays instead of vectors as they'll be wrapped into PyArrayObject
 string* vM1 = new string[non_zeros];
 string* vM2 = new string[non_zeros];
 double* vM3 = new double[non_zeros];
 vector<double> vsm;
 vector<unsigned long int> vk;


 k=0;
 for(i=0;i<nrows;i++){
  sm3=0;
  for(j=0;j<lrS0[i];j++){
   mij=S(i,S0(i,j));
   if(mij>0){
      vM1[k]=vchannels[i];
	  vM2[k]=vchannels[S0(i,j)];
	  vM3[k]=mij;
      sm3=sm3+mij;
      ++k;
	}
  }

  vsm.push_back(sm3);
  vk.push_back(k);

 }//end for

 unsigned long int w=0;
 for(k=0;k<non_zeros;k++){
  if(k==vk[w]){++w;}
  vM3[k]=vM3[k]/vsm[w];
 }

npy_intp *dims = new npy_intp[1];
dims[0] = non_zeros;

const char **vM1_c, **vM2_c;
for (k=0; k<non_zeros; k++){
  vM1_c[k] = vM1[k].c_str();
  vM2_c[k] = vM2[k].c_str();
}

//Create a dictionary out of numpy objects
 PyObject* vM1_np = PyArray_SimpleNewFromData(1, dims, NPY_STRING, vM1_c);
 PyObject* vM2_np = PyArray_SimpleNewFromData(1, dims, NPY_STRING, vM2_c);
 PyObject* vM3_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, vM3);

return Py_BuildValue("{s:O, s:O, s:O}",
 "channel_from", vM1_np,
  "channel_to", vM2_np,
   "transition_probability", vM3_np
);
}

static PyObject* markov_model_cpp(PyObject* self, PyObject* args)

// RcppExport SEXP markov_model_cpp(SEXP Data_p,
//    SEXP var_path_p, SEXP var_conv_p, SEXP var_value_p, SEXP var_null_p,
 // SEXP order_p, SEXP nsim_p, SEXP max_step_p, SEXP out_more_p)
{

// char* var_path_c, *var_conv_c, *var_value_c, *var_null_c;
unsigned long long int order, nsim, max_step, out_more;
bool flg_var_value, flg_var_null;
PyArrayObject* vy0, *vc0, *vv0, *vn0;

// if (!PyArg_ParseTuple(args, "O!O!O!O!ssssKKKK",
if (!PyArg_ParseTuple(args, "O!O!O!O!KKKKbb",
  &PyArray_Type, &vy0,
   &PyArray_Type, &vc0,
    &PyArray_Type,  &vv0,
      &PyArray_Type,  &vn0,
       // &var_path_c, &var_conv_c, &var_value_c, &var_null_c,
        &order, &nsim, &max_step, &out_more,
         &flg_var_value, &flg_var_null)) {
     return NULL;
  }

// std::string var_value(var_value_c);
// std::string var_path(var_path_c);
// std::string var_conv(var_conv_c);
// std::string var_null(var_null_c);

//inp.b
// bool flg_var_value;
// flg_var_value=0;
// if(var_value.compare("0")!=0){
//  flg_var_value=1;
// }

// bool flg_var_null;
// flg_var_null=0;
// if(var_null.compare("0")!=0){
//  flg_var_null=1;
// }

npy_intp vy_n = vy0->dimensions[0];
npy_intp vv_n = vv0->dimensions[0];
npy_intp vc_n = vc0->dimensions[0];
npy_intp vn_n = vn0->dimensions[0];

string* vy1 = static_cast<string*>(PyArray_DATA(vy0));
double* vv1 = static_cast<double*>(PyArray_DATA(vv0));
unsigned long int* vc1 =  static_cast<unsigned long int*>(PyArray_DATA(vc0));
unsigned long int* vn1 = static_cast<unsigned long int*>(PyArray_DATA(vn0));

/*Make vectors*/
vector<double> vv;
if (flg_var_value){
  vv.assign(vv1, vv1+vv_n);
}
vector<unsigned long int> vn;
if (flg_var_null){
  vn.assign(vn1, vn1+vn_n);
}
vector<long long int> vc(vc1, vc1+vc_n);
vector<string> vy(vy1, vy1+vy_n);

// Ancillary variables.
unsigned long int i,j,k,lvy,ssize;
unsigned long int nchannels,nchannels_sim,npassi;
string s,channel,path;
map<string,unsigned long int> mp_channels,mp_channels_sim;
map<unsigned long int,unsigned long int> mp_npassi;
vector<unsigned long int> vnpassi;

lvy=(unsigned long int) vy.size();

 //////////////////////
 //CODIFICA DA ONE STEP
 //////////////////////

 //mappa dei conversion value
 unsigned long int l_vui=0;
 map<double,unsigned long int> mp_vui;
 vector<double> v_vui;
 vector<double> vu(lvy);
 double vui;

 vector<string> rchannels;
 unsigned long int lrchannels,j0,z;
 string channel_j;

 // vector<unsigned long int> vchannels_sim_id(order);
 // map<unsigned long int, vector<unsigned long int>> mp_channels_sim_id;
 vector<long int> vchannels_sim_id(order);
 map<unsigned long int, vector<long int>> mp_channels_sim_id;

 nchannels=0;
 nchannels_sim=0;

 vector<string> vy2(lvy);

 mp_channels["(start)"]=0;
 vector<string> vchannels;
 vchannels.push_back("(start)");
 ++nchannels;

 vector<string> vchannels_sim;
 for(z=0;z<order;z++){
  vchannels_sim_id[z]=-1;
 }
 if(order>1){
  mp_channels_sim["(start)"]=nchannels_sim;
  vchannels_sim.push_back("(start)");
  vchannels_sim_id[0]=nchannels_sim;
  mp_channels_sim_id[nchannels_sim]=vchannels_sim_id;
  ++nchannels_sim;
 }


 //definizione mappa conversion value
 if(flg_var_value==1){
  for(i=0;i<lvy;i++){
   vui=vv[i]/vc[i];
   vu[i]=vui;
   if(mp_vui.find(vui)==mp_vui.end()){
    mp_vui[vui]=l_vui;
    v_vui.push_back(vui);
    ++l_vui;
   }
  }
 }

 for(i=0;i<lvy;i++){

  s=vy[i];
  s+=" >";
  ssize=(unsigned long int) s.size();
  channel="";
  path="";
  j=0;
  npassi=0;
  rchannels.clear();

  //medium.touch

  while(j<ssize){

   if((j>0) & (ssize>1)){

     if((s[j-1]=='>') & (s[j]==' ')){
	  j=j+1;
	 }
	 if((s[j]==' ') & (s[j+1]=='>')){
	  j=j+2;
	  break;
	 }

   }

   while((s[j]!=' ') & (s[j+1]!='>')){
	 channel+=s[j];
     ++j;
   }
   ++j;

   if(mp_channels.find(channel) == mp_channels.end()){
    mp_channels[channel]=nchannels;
    vchannels.push_back(channel);
    ++nchannels;
  }

   if(order==1){

    if(npassi==0){
     path="0 ";
    }else{
     path+=" ";
    }

    path+=to_string(mp_channels[channel]);
    ++npassi;

   }else{

    rchannels.push_back(channel);

   }

   channel="";
   ++j;

  }//end while channel

  if(order>1){

	lrchannels=rchannels.size();
	for(z=0;z<order;z++){
	 vchannels_sim_id[z]=-1;
	}

    if(lrchannels>(order-1)){

     npassi=lrchannels-order+1;

     for(k=0;k<npassi;k++){

	  channel="";
	  channel_j="";

  	  z=0;
	  j0=k+order;
	  for(j=k;j<j0;j++){
	    channel_j=rchannels[j];
	    channel+=channel_j;
	    vchannels_sim_id[z]=mp_channels[channel_j];
	    ++z;
	    if(j<(j0-1)){
	     channel+=",";
	    }
	  }

	  if(mp_channels_sim.find(channel) == mp_channels_sim.end()){
	   mp_channels_sim[channel]=nchannels_sim;
       vchannels_sim.push_back(channel); //lo utilizzo per output more
	   mp_channels_sim_id[nchannels_sim]=vchannels_sim_id;
       ++nchannels_sim;
      }

	  path+=to_string(mp_channels_sim[channel]);
	  path+=" ";

	 }//end for k


	}else{

	  npassi=1;

	  channel="";
	  channel_j="";
	  for(j=0;j<lrchannels;j++){
	   channel_j=rchannels[j];
	   channel+=channel_j;
	   vchannels_sim_id[j]=mp_channels[channel_j];
	   if(j<(lrchannels-1)){
	     channel+=",";
	   }
	  }

	  if(mp_channels_sim.find(channel) == mp_channels_sim.end()){
	   mp_channels_sim[channel]=nchannels_sim;
       vchannels_sim.push_back(channel); //lo utilizzo per output more
	   mp_channels_sim_id[nchannels_sim]=vchannels_sim_id;
       ++nchannels_sim;
      }

      path+=to_string(mp_channels_sim[channel]);
	  path+=" ";

	}//end else

    path="0 "+path;

  }else{//end order > 1

	path+=" ";

  }

  vy2[i]=path+"e"; //aggiungo lo stato finale
  ++npassi;

 }//end for

 mp_channels["(conversion)"]=nchannels; //aggiungo canale conversion
 ++nchannels;
 vchannels.push_back("(conversion)");

 mp_channels["(null)"]=nchannels;
 ++nchannels;
 vchannels.push_back("(null)");

 if(order>1){
  mp_channels_sim["(conversion)"]=nchannels_sim;
  vchannels_sim.push_back("(conversion)");
  for(z=0;z<order;z++){
   vchannels_sim_id[0]=nchannels_sim;
  }
  mp_channels_sim_id[nchannels_sim]=vchannels_sim_id;
  ++nchannels_sim;

  mp_channels_sim["(null)"]=nchannels_sim;
  vchannels_sim.push_back("(null)");
  for(z=0;z<order;z++){
   vchannels_sim_id[0]=nchannels_sim;
  }
  mp_channels_sim_id[nchannels_sim]=vchannels_sim_id;
  ++nchannels_sim;

 }

 if(order==1){
  nchannels_sim=nchannels;
 }

 //cout << "Processed 2/4" << endl;

 /////////////////////////////////////////////////////
 //CREAZIONE DELLE MATRICI FUNZIONALI ALLE SIMULAZIONI
 ////////////////////////////////////////////////////

 unsigned long int ichannel,ichannel_old,vpi,vci,vni;
 string channel_old;

 npassi=0;

 Fx S(nchannels_sim,nchannels_sim);

 Fx fV(nchannels_sim,l_vui);

 for(i=0;i<lvy;i++){

  s=vy2[i];
  s+=" ";
  ssize= (unsigned long int) s.size();

  channel="";
  channel_old="";
  ichannel_old=0;
  ichannel=0;

  j=0;
  npassi=0;

  vci=vc[i];
  if(flg_var_null==1){
   vni=vn[i];
  }else{
   vni=0;
  }
  if(flg_var_value==1){
   vui=vu[i];
  }
  vpi=vci+vni;

  while(j<ssize){

   while(s[j]!=' '){

    if(j<ssize){
     channel+=s[j];
    }
    j=j+1;
   }

   if(channel.compare(channel_old)!=0){

    if(channel[0]!='0'){//se non è il channel start

     if(channel[0]=='e'){ //stato finale

	  ++npassi;

	  if(vci>0){ //se ci sono conversion
	   ichannel=nchannels_sim-2;
	   S.add(ichannel_old,ichannel,vci);
	   if(flg_var_value==1){
	    fV.add(ichannel_old,mp_vui[vui],vci);
	   }
	   if(vni>0){
		goto next_null;
	   }else{
		goto next_path;
	   }
	  }

	  if(vni>0){ //se non ci sono conversion
	   next_null:;
	   ichannel=nchannels_sim-1;
	   S.add(ichannel_old,ichannel,vni);
	   goto next_path;
      }

     }else{ //stato non finale

	  if(vpi>0){
       ichannel=atol(channel.c_str());
   	   S.add(ichannel_old,ichannel,vpi);
	  }

	 }

	 ++npassi;

    }else{ //stato iniziale

     ichannel=0;

    }

    channel_old=channel;
    ichannel_old=ichannel;

   }//end compare

   channel="";

   j=j+1;

  }//end while j<size

  next_path:;

 }//end for

 //out matrice di transizione

// TODO change the behavior here
 PyObject* res_mtx;
 if(out_more==1){
  if(order==1){
   res_mtx=S.tran_matx(vchannels);
  }else{
   res_mtx=S.tran_matx(vchannels_sim);
  }
 }


 //f.r. transizione
 S.cum();

 //return(0);

 //f.r. conversion value
 if(flg_var_value==1){
  fV.cum();
 }

 //distribuzione numeri uniformi
int iu, nuf=1e6;

//Instantiate and initialize the random number generator
uniform_real_distribution<double> unif(0.0, 1.0);
random_device rd;
default_random_engine re(rd());
//Generate an array of numbers
double* vunif = new double[nuf];
for (size_t i = 0; i < nuf; i++) {
  vunif[i] = unif(re);
}

 // NumericVector vunif=runif(nuf);

 //cout << "Processed 3/4" << endl;

 //SIMULAZIONI

 unsigned long int c,c_last,nconv,max_npassi;
 long int id0;
 double sval0,ssval;
 vector<bool> C(nchannels);
 vector<double> T(nchannels);
 vector<double> V(nchannels);

 nconv=0;
 sval0=0;
 ssval=0;
 c_last=0;
 iu=0;

 if(max_step==0){
  max_npassi=nchannels_sim*10;
 }else{
  max_npassi=1e6;
 }
 if(nsim==0){
  nsim=1e6;
 }


 for(i=0; i<nsim; i++){

  c=0;
  npassi=0;

  for(k=0; k<nchannels; k++){ //svuoto il vettore del flag canali visitati
   C[k]=0;
  }

  C[c]=1; //assegno 1 al channel start

  while(npassi<=max_npassi){ //interrompo quando raggiungo il massimo numero di passi

   if(iu>=nuf){
     for (size_t i = 0; i < nuf; i++) {
       vunif[i] = unif(re);
     }
     iu=0;
   }//genero il canale da visitare

   c=S.sim(c,vunif[iu]);
   ++iu;

   if(c==nchannels_sim-2){ //se ho raggiunto lo stato conversion interrompo
    goto go_to_conv;
   }else if(c==nchannels_sim-1){ //se ho raggiunto lo stato null interrompo
	goto go_to_null;
   }

   if(order==1){
	C[c]=1; //flaggo con 1 il canale visitato
   }else{
    for(k=0; k<order; k++){
	 id0=mp_channels_sim_id[c][k];
	 if(id0>=0){
      C[id0]=1;
     }else{
	  break;
	 }
	}
   }

   c_last=c; //salvo il canale visitato
   ++npassi;

  }//end while npassi

  go_to_conv:;

  if(c==nchannels_sim-2){ //solo se ho raggiunto la conversion assegno +1 ai canali interessati (se ho raggiunto il max numero di passi è come se fossi andato a null)

   ++nconv;//incremento le conversion

   //genero per il canale c_last un valore di conversion "sval0"
   if(flg_var_value==1){
    if(iu>=nuf){
      for (size_t i = 0; i < nuf; i++) {
        vunif[i] = unif(re);
      }
      iu=0;
    }
    sval0=v_vui[fV.sim(c_last,vunif[iu])];
    ++iu;
   }

   ssval=ssval+sval0;

   for (k=0; k<nchannels; k++){
    if(C[k]==1){
	 T[k]=T[k]+1;
	 if(flg_var_value==1){
	  V[k]=V[k]+sval0;
	 }
    }
   }

  }//end if conv

  go_to_null:;

 }//end for i


 T[0]=0; //pongo channel start = 0
 unsigned long int nch0;
 nch0=nchannels-3;
 T[nchannels-2]=0; //pongo channel conversion = 0
 T[nchannels-1]=0; //pongo channel null = 0

 double sn=0;
 for(i=0;i<lvy; i++){
  sn=sn+vc[i];
 }

 double sm=0;
 for(i=0;i<nchannels-1; i++){
  sm=sm+T[i];
 }

 vector<double> TV(nch0,0);
 vector<double> rTV(nch0,0);

 for (k=1; k<(nch0+1); k++){
  if(sm>0){
   TV[k-1]=(T[k]/sm)*sn;
   if(out_more==1){rTV[k-1]=T[k]/nconv;} //removal effects
  }
 }

 vector<double> VV(nch0,0);
 vector<double> rVV(nch0,0);

 if(flg_var_value==1){

  V[0]=0; //pongo channel start = 0
  V[nchannels-2]=0; //pongo channel conversion = 0
  V[nchannels-1]=0; //pongo channel null = 0

  sn=0;
  for(i=0;i<lvy; i++){
   sn=sn+vv[i];
  }

  sm=0;
  for(i=0;i<nchannels-1; i++){
   sm=sm+V[i];
  }

  for(k=1; k<(nch0+1); k++){
   if(sm>0){
    VV[k-1]=(V[k]/sm)*sn;
    if(out_more==1){rVV[k-1]=V[k]/ssval;} //removal effects
   }
  }
 }

//TODO: vchannels0,
PyObject *TV_np, *rTV_np, *VV_np, *rVV_np, *vchannels0_np;
npy_intp* dims = new npy_intp[1];
dims[0] = nch0;

TV_np =  PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, TV.data());
if (out_more)
  rTV_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, rTV.data());
if (flg_var_value){
  VV_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, VV.data());
    if (out_more)
      rVV_np = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, rVV.data());
}

 const char** vchannels0 = new const char*[nch0];
 for(k=1; k<(nch0+1); k++){
  vchannels0[k-1]=vchannels[k].c_str();
 }

vchannels0_np = PyArray_SimpleNewFromData(1, dims, NPY_STRING, vchannels0);

 if(flg_var_value==1){
  if(out_more==0){
    return Py_BuildValue("{s:O,s:O,s:O}",
     "channel_name", vchannels0_np,
     "total_conversion", TV_np,
     "total_conversion_value", VV_np);
  }
  else {
    PyObject *res1 = Py_BuildValue("{s:O,s:O,s:O}",
      "channel_name", vchannels0_np,
      "total_conversion", TV_np,
      "total_conversion_value", VV_np);

   PyObject *res3 = Py_BuildValue("{s:O,s:O,s:O}",
     "channel_name", vchannels0_np,
     "removal_effects_conversion", rTV_np,
     "removal_effects_conversion_value", rVV_np);

   return Py_BuildValue("{s:O,s:O,s:O}",
      "result", res1,
      "transition_matrix", res_mtx,
      "removal_effects", res3);
  }
 }
 else {
  if(out_more==0){
    return Py_BuildValue("{s:O,s:O}",
      "channel_name", vchannels0_np,
      "total_conversions", TV_np);
  }
  else {
    PyObject *res1 = Py_BuildValue("{s:O,s:O,s:O}",
     "channel_name", vchannels0_np,
     "total_conversion", TV_np);

   PyObject *res3 = Py_BuildValue("{s:O,s:O,s:O}",
     "channel_name", vchannels0_np,
     "removal_effects_conversion", rTV_np);

   return Py_BuildValue("{s:O,s:O,s:O}",
      "result", *res1,
      "transition_matrix", res_mtx,
      "removal_effects", res3);
  }
 }
}
