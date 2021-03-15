#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <math.h>
#include <vector>
#include <string>
#include "Header.h"

__device__ void EnergyConservationLaws(const double2& PS, const double2& PU, const double2& s_1, const double2& u_1, //
    double2& s2, double2& u2, double T_do, double x, double y, double dV)
{
    s2.x = s_1.x - (T_do / dV) * PS.x - T_do * s_1.x * u_1.y / y;
    if (s2.x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf, T = %lf, ro = %lf \n", x, y, s2.x, T_do, s_1.x);
        s2.x = s_1.x;
    }
    u2.x = (s_1.x * u_1.x - (T_do / dV) * PU.x - T_do * s_1.x * u_1.y * u_1.x / y) / s2.x;
    u2.y = (s_1.x * u_1.y - (T_do / dV) * PU.y - T_do * s_1.x * u_1.y * u_1.y / y) / s2.x;
    s2.y = (((s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) - (T_do / dV) * PS.y - //
        T_do * u_1.y * (ggg * s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) / y) - //
        0.5 * s2.x * (u2.x * u2.x + u2.y * u2.y)) * (ggg - 1);

    if (s2.y <= 0)
    {
        s2.y = 0.000001;
    }
}


void InitialConditions(double2& s2, double2& u2, double x, double y)
{
    // u2 - скорость, s2 - плотность и давление
    if (x * x + y * y < 10000)
    {
        s2.x = 8.0;
        s2.y = 2.0;
        u2.x = -90.0;
        u2.y = 0.0; 
    }
    else
    {
        s2.x = 1.0;
        s2.y = 1.0;
        u2.x = -1.0;
        u2.y = 0.0;
    }
   
}


void Initialization(int& N_, int& M_, int& step_1_, int& step_2_, int& step_3_, int& step_4_, int& step_5_, int& step_6_,//
    bool& ots__, double& x_min_, double& x_max_, double& y_max_, double& U_ots_, double& krit_, int& Nmin_, double& dist_1_)
{
    // Параметры которые нужно менять:
    N_ = 256; // 7167 //1792 //1792                 // Количество ячеек по x
    M_ = 256; // //1280 //1280                 // Количество ячеек по y
    // Важно:  N*M должно делится на 256!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    x_min_ = -2760.0;
    x_max_ = 450.0;
    y_max_ = 2250.0;

    dist_1_ = 110.0;     // Расстояние внутренней сферы (с которой начинается счёт), внутри неё не считается
    Nmin_ = 5;              // Каждую какую точку выводим?

    ots__ = false;   // Особые условия отсоса жидкости через заднюю стенку
    U_ots_ = -5.0;  // Какой именно отсос включить

    step_1_ = 5000;   // Количество шагов по времени методом HLL
    step_2_ = 0;   // Количество шагов по времени методом HLLC
    step_3_ = 0;   // Количество шагов по времени методом GODUNOV
    step_4_ = 0;   // Количество шагов по времени методом HLL + TVD
    step_5_ = 0;   // Количество шагов по времени методом HLLC + TVD
    step_6_ = 0;   // Количество шагов по времени методом GODUNOV + TVD

    krit_ = 0.3;    // Критерий Куранта-Фридрихса-Леви  (<1)
}