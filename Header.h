#pragma once
#include "sensor.h"


#define Omega 0.0
#define N 512 // 7167 //1792 //1792                 // Количество ячеек по x
#define M 512 // //1280 //1280                 // Количество ячеек по y
#define K (N*M)                // Количество ячеек в сетке
#define x_min -3.0 // -2760.0 // -2500.0 // -1300  //-2000                // -1500.0
#define x_max 1.5 //450.0
#define y_max 4.5 // 2250.0 // 1600.0 //1840.0
#define y_min (y_max/(2.0 * M)) 
#define dx ((x_max - x_min)/(N-1))     // Величина грани по dx (всей грани, а не половины)
#define dy ((y_max)/(M))     // Величина грани по dy
#define ggg (5.0/3.0)          // Показатель адиабаты
#define ga (5.0/3.0)          // Показатель адиабаты
#define M_inf  0.5
//#define phi_0 1.626      //1.626        //4.878      //1.626
#define g1 (ga - 1.0)
#define gg1 (ga - 1.0)
#define g2 (ga + 1.0)
#define gg2 (ga + 1.0)
#define gp ((g2/ga)/2.0)
#define gm ((g1/ga)/2.0)
#define gga ga
#define ER_S std::cout << "\n---------------------\nStandart error in file: Solvers.cpp\n" << endl
#define watch(x) cout << (#x) << " is " << (x) << endl
#define hy 0.1
#define hx -10.4
#define grad_p true
#define Nmin 1              // Каждую какую точку выводим?
#define THREADS_PER_BLOCK 256    // Количество нитей в одном потоке
// Необходимо, чтобы количество ячеек в сетке делилось на число нитей (лучше N делилось на число нитей)
#define kor 2000000.0


#define krit 0.3
#define ggg (5.0/3.0)          // Показатель адиабаты
#define ga (5.0/3.0)          // Показатель адиабаты
#define phi_0 1.626 // 1.626 //1.626        //4.878      //1.626
#define g1 (ga - 1.0)
#define gg1 (ga - 1.0)
#define g2 (ga + 1.0)
#define gg2 (ga + 1.0)
#define gp ((g2/ga)/2.0)
#define gm ((g1/ga)/2.0)
#define gga ga
#define kv(x) ( (x)*(x) )
#define kvv(x,y,z)  (kv(x) + kv(y) + kv(z))
#define modsphere(r, the, Vr, Vthe, Vphi)  (kv(Vr) + kv(r) * (kv(Vthe) + sin( (the) ) * kv(Vphi)))

#define eps 10e-10
#define eps8 10e-8
#define pi 3.14159265358979323846
#define sqrtpi 1.77245385
#define PI 3.14159265358979323846
#define cpi4 12.56637061435917295384
#define cpi8 25.13274122871834590768
#define spi4 ( 3.544907701811032 )
#define epsb 1e-6
#define eps_p 1e-6
#define eps_d 1e-3


#define Kn  0.419231
#define n_H  3.0
#define Velosity_inf -2.54186 //-2.54127
#define sqv_1 (2.54189 * pi * (kv(y_max) - kv(1.0)) )   //(2.5412983502 * pi * kv(y_max)) 
#define sqv_2 (y_max * sqrtpi * (x_max - x_min + dx))
#define sqv_3 (0.0000282543 * pi * kv(y_max))   //(2.5412983502 * pi * kv(y_max)) 
#define sqv_4 (2.54189 * pi * kv(1.0))   //(2.5412983502 * pi * kv(y_max)) 
#define sum_s (sqv_1 + sqv_2 + sqv_3 + sqv_4)
#define Number1 27000000   // Должно делится на 270
#define Number2 3240000 //324000
#define Number3 2700000 //27000
#define Number4 27000000 //27000
#define AllNumber (Number1 + Number2 + Number3 + Number4)
#define a_2 0.111452
#define sigma(x) (kv(1.0 - a_2 * log(x)))
#define geo 0.000009   // Геометрическая точность   0.1% от размера ячейки должно быть

#define ChEx true   // Нужно включить перезарядку?


extern __device__ int sign_(const double& x);
extern __device__ int sign(double& x);
extern __device__ double minmod_(double x, double y);
extern __device__ double linear_(double x1, double t1, double x2, double t2, double x3, double t3, double y);
extern __device__ void linear2_(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
    double& A, double& B);
extern __device__ void TVD(const double2& s_1, const double2& s_2, const double2& s_3, const double2& s_4, const double2& s_5,//
    const double2& s_6, const double2& s_7, const double2& s_8, const double2& s_9, double2& s12,//
    double2& s13, double2& s14, double2& s15, double2& s21, double2& s31, double2& s41, double2& s51, double ddx, double ddy, bool zero);
extern __device__ double HLLC_Korolkov_2D(const double2& Ls, const double2& Lu, const double2& Rs, const double2& Ru,//
    const double n1, const double n2, double2& Ps, double2& Pu, const double rad);
extern __device__ double HLLCQ_Korolkov_2D(const double2& Ls, const double2& Lu, const double2& Rs, const double2& Ru,//
    const double& LQ, const double& RQ, double n1, double n2, double2& Ps, double2& Pu, double& PQ, double rad);
extern __device__ double HLLCQ_Aleksashov(const double2& Ls, const double2& Lu, const double2& Rs, const double2& Ru,//
    const double& LQ, const double& RQ, double n1, double n2, double2& Ps, double2& Pu, double& PQ, double rad);
extern __device__ double HLLC_Aleksashov_2D(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad);

double polar_angle(double x, double y);
void spherical_skorost(double x, double y, double z, double Vx, double Vy, double Vz, double& Vr, double& Vphi, double& Vtheta);
void dekard_skorost(double x, double y, double z, double Vr, double Vphi, double Vtheta, double& Vx, double& Vy, double& Vz);
extern void M_K(vector<Sensor*> Sensors, const double2* s, const double2* u, double* nn1, double3* nn2, double* nn3);
void Velosity_initial(Sensor* s, double& Vx, double& Vy, double& Vz);
void Change_Velosity(Sensor* s, const double& Ur, const double& Uthe, const double& Uphi, //
    const double& Vr, const double& Vthe, const double& Vphi, double& X, double& Y, double& Z);
void Belong_point(const double& x, const double& y, int& n, int& m);
void peresich(const double& y,const double& z, const double& Vy, const double& Vz, const double& R, double& t1, double& t2);
double minplus(double x, double y);
bool Flying_exchange(double& KSI, double& Vx, double& Vy, double& Vz, double& X, double& Y,//
    double& Z, int& next, int head, int prev, const double& mu, double& I_do, const double& ro, //
    const double& vx, const double& vy, double* nn1, double3* nn2, double* nn3, bool& error, mutex& mut);
void Fly_exchenge(Sensor* sens, double x_0, double y_0, double z_0, double Vx, double Vy, double Vz, int ind, //
    const double2* s, const double2* u, double mu, double* nn1, double3* nn2, double* nn3, int num, mutex* mut, bool info = false);
bool Peresechenie(const double& x0, const double& y0, const double& x, const double& y, const double& z, const double& Vx,//
    const double& Vy, const double& Vz, int& mode, double& t);
void Velosity_initial2(Sensor* s, double& Vx, double& Vy, double& Vz);