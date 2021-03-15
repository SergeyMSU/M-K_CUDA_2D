#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <string>
#include "Header.h"
#include "sensor.h"

using namespace std;

//__device__ int sign(double& x);
__device__ double minmod(double x, double y);
__device__ double linear(double x1, double t1, double x2, double t2, double x3, double t3, double y);
__device__ void linear2(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
    double& A, double& B);
__global__ void add2(double2* s, double2* u, double2* s2, double2* u2, double* T, double* T_do, int method, int step);

__device__ double minmod(double x, double y)
{
    if (sign(x) + sign(y) == 0)
    {
        return 0.0;
    }
    else
    {
        return   ((sign(x) + sign(y)) / 2.0) * min(fabs(x), fabs(y));  ///minmod
        //return (2*x*y)/(x + y);   /// vanleer
    }
}

__device__ double linear(double x1, double t1, double x2, double t2, double x3, double t3, double y)
{
    double d = minmod((t1 - t2) / (x1 - x2), (t2 - t3) / (x2 - x3));
    return  (d * (y - x2) + t2);
}

__device__ void linear2(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
    double& A, double& B)
{
    // ГЛАВНОЕ ЗНАЧЕНИЕ - ЦЕНТРАЛЬНОЕ - НЕ ЗАБЫВАЙ ОБ ЭТОМ
    double d = minmod((t1 - t2) / (x1 - x2), (t2 - t3) / (x2 - x3));
    A = (d * (y1 - x2) + t2);
    B = (d * (y2 - x2) + t2);
    //printf("%lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf \n", x1, t1, x2, t2, x3, t3, y1, y2, A, B);
    return;
}


__device__ double  my_min(double a, double b)
{
    if (a <= b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

__device__ double  my_max(double a, double b)
{
    if (a >= b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

__device__ void lev(const double& enI, const double& pI, const double& rI, const double& enII,//
    const double& pII, const double& rII, double& uuu, double& fee);
__device__ void devtwo(const double& enI, const double& pI, const double& rI, const double& enII, const double& pII, const double& rII, //
    const double& w, double& p);
__device__ void newton(const double& enI, const double& pI, const double& rI, const double& enII, const double& pII, const double& rII, //
    const double& w, double& p);
__device__ void perpendicular(double a1, double a2, double a3, double& b1, double& b2, double& b3, //
    double& c1, double& c2, double& c3, bool t);
__device__ double Godunov_Solver_Alexashov(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad);
__host__ bool areaa(double x, double y, double ro, double p, double u, double v);

__host__ bool areaa(double  x, double y, double ro, double p, double u, double v)
{
    if (ro <= 0.0)
    {
        return true;
    }
    double Max = sqrt((u * u + v * v) / (ggg * p / ro));
    double T = p / ro;
    if ((x < 36.8) && (y < 336))
    {
        return true;
    }
    if (( fabs(ro - 1.0) < 0.000001) && (fabs(Max - 3.0) < 0.000001))
    {
        return false;
    }
    if ((x > 240.3)||(y > 616.4) )
    {
        return false;
    }
    if ((x < -368) && ( T > 0.12))
    {
        return true;
    }
    if (M > 3.3)
    {
        return true;
    }
    if ((x > 1.0)&&(ro < 1.7))
    {
        return true;
    }
    else
    {
        return false;
    }
    return false;
}

__device__ double Godunov_Solver_Alexashov(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double w = 0.0;
    double al = n1;
    double be = n2;
    double ge = 0.0;
    double time = 0.0;

    double al2 = -n2;
    double be2 = n1;
    double ge2 = 0.0;
    double al3 = 0.0;
    double be3 = 0.0;
    double ge3 = 1.0;

    double enI = al * Lu.x + be * Lu.y;
    double teI2 = al2 * Lu.x + be2 * Lu.y;
    double teI3 = al3 * Lu.x + be3 * Lu.y;
    double enII = al * Ru.x + be * Ru.y;
    double teII2 = al2 * Ru.x + be2 * Ru.y;
    double teII3 = al3 * Ru.x + be3 * Ru.y;

    double pI = Ls.y;
    double pII = Rs.y;
    double rI = Ls.x;
    double rII = Rs.x;

    int ipiz = 0;
    if (pI > pII)   // Смена местами величин
    {
        double eno2 = enII;;
        double teo22 = teII2;
        double teo23 = teII3;
        double p2 = pII;
        double r2 = rII;

        double eno1 = enI;
        double teo12 = teI2;
        double teo13 = teI3;
        double p1 = pI;
        double r1 = rI;

        enI = -eno2;
        teI2 = teo22;
        teI3 = teo23;
        pI = p2;
        rI = r2;

        enII = -eno1;
        teII2 = teo12;
        teII3 = teo13;
        pII = p1;
        rII = r1;
        w = -w;
        ipiz = 1;                                                                // ???? Он точно здесь должен быть?
    }

    double cI = 0.0;
    double cII = 0.0;
    if (rI != 0.0)
    {
        cI = __dsqrt_rn(ga * pI / rI);
    }
    if (rII != 0.0)
    {
        cII = __dsqrt_rn(ga * pII / rII);
    }

   /* printf("C2 !!!! = %lf =  kor  %lf \n", cII, ga * pII / rII);
    printf("%lf , %lf, %lf \n",ga,pII,rII);*/

    double a = __dsqrt_rn(rI * (g2 * pII + g1 * pI) / 2.0);
    double Uud = (pII - pI) / a;
    double Urz = -2.0 * cII / g1 * (1.0 - pow((pI / pII), gm));
    double Uvk = -2.0 * (cII + cI) / g1;
    double Udf = enI - enII;

    int il, ip;
    double p, r, te2, te3, en;

    if (Udf < Uvk)
    {
        il = -1;
        ip = -1;
    }
    else if ((Udf >= Uvk) && (Udf <= Urz))
    {
        p = pI * pow(((Udf - Uvk) / (Urz - Uvk)), (1.0 / gm));
        il = 0;
        ip = 0;
    }
    else if ((Udf > Urz) && (Udf <= Uud))
    {
        devtwo(enI, pI, rI, enII, pII, rII, w, p);
        il = 1;
        ip = 0;
    }
    else if (Udf > Uud)
    {
        newton(enI, pI, rI, enII, pII, rII, w, p);
        il = 1;
        ip = 1;
    }

    //*********TWO SHOCKS**********************************************
    if ((il == 1) && (ip == 1))
    {
       /* printf("TWO SHOCKS\n");*/
        double aI = __dsqrt_rn(rI * (g2 / 2.0 * p + g1 / 2.0 * pI));
        double aII = __dsqrt_rn(rII * (g2 / 2.0 * p + g1 / 2.0 * pII));

        double u = (aI * enI + aII * enII + pI - pII) / (aI + aII);
        double dI = enI - aI / rI;
        double dII = enII + aII / rII;


        double UU = max(fabs(dI), fabs(dII));
        if (UU > eps8)
        {
            time = krit * rad / UU;
        }
        else
        {
            time = krit * rad / eps8;
        }


        if (w <= dI)
        {
            en = enI;
            p = pI;
            r = rI;
            te2 = teI2;
            te3 = teI3;
        }
        else if ((w > dI) && (w <= u))
        {
            en = u;
            p = p;
            r = rI * aI / (aI - rI * (enI - u));
            te2 = teI2;
            te3 = teI3;
        }
        else if ((w > u) && (w < dII))
        {
            en = u;
            p = p;
            r = rII * aII / (aII + rII * (enII - u));
            te2 = teII2;
            te3 = teII3;
        }
        else if (w >= dII)
        {
            en = enII;
            p = pII;
            r = rII;
            te2 = teII2;
            te3 = teII3;
        }
    }


    //*********LEFT - SHOCK, RIGHT - EXPANSION FAN*******************
    if ((il == 1) && (ip == 0))
    {
        //printf("LEFT - SHOCK, RIGHT - EXPANSION FAN\n");
        double aI = __dsqrt_rn(rI * (g2 / 2.0 * p + g1 / 2.0 * pI));
        double aII;
        if (fabs(p - pII) < eps)
        {
            aII = rII * cII;
        }
        else
        {
            aII = gm * rII * cII * (1.0 - p / pII) / (1.0 - pow((p / pII), gm));
        }

        double u = (aI * enI + aII * enII + pI - pII) / (aI + aII);
        double dI = enI - aI / rI;
        double dII = enII + cII;
        double ddII = u + cII - g1 * (enII - u) / 2.0;

        double UU = max(fabs(dI), fabs(dII));
        UU = max(UU, fabs(ddII));
        if (UU > eps8)
        {
            time = krit * rad / UU;
        }
        else
        {
            time = krit * rad / eps8;
        }

        if (w <= dI)
        {
            en = enI;
            p = pI;
            r = rI;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w > dI) && (w <= u))
        {
            en = u;
            p = p;
            r = rI * aI / (aI - rI * (enI - u));
            te2 = teI2;
            te3 = teI3;
        }
        if ((w > u) && (w <= ddII))
        {
            double ce = cII - g1 / 2.0 * (enII - u);
            en = u;
            p = p;
            r = ga * p / ce / ce;
            te2 = teII2;
            te3 = teII3;
        }
        if ((w > ddII) && (w < dII))
        {
            double ce = -g1 / g2 * (enII - w) + 2.0 / g2 * cII;
            en = w - ce;
            p = pII * pow((ce / cII), (1.0 / gm));
            r = ga * p / ce / ce;
            te2 = teII2;
            te3 = teII3;
        }
        if (w >= dII)
        {
            en = enII;
            p = pII;
            r = rII;
            te2 = teII2;
            te3 = teII3;
        }
    }
    //*********TWO EXPANSION FANS**************************************
    if ((il == 0) && (ip == 0))
    {
        //printf("TWO EXPANSION FANS\n");
        double aI;
        //printf("p = %lf\n", p);
        if (fabs(p - pI) < eps)
        {
            aI = rI * cI;
        }
        else
        {
            aI = gm * rI * cI * (1.0 - p / pI) / (1.0 - pow((p / pI), gm));
        }
        //printf("aI = %lf\n", aI);

        double aII;
        if (fabs(p - pII) < eps)
        {
            aII = rII * cII;
        }
        else
        {
            aII = gm * rII * cII * (1.0 - p / pII) / (1.0 - pow((p / pII), gm));
        }

        //printf("aII = %lf\n", aI);

        double u = (aI * enI + aII * enII + pI - pII) / (aI + aII);
        double dI = enI - cI;
        double ddI = u - cI - g1 * (enI - u) / 2.0;
        double dII = enII + cII;
        double ddII = u + cII - g1 * (enII - u) / 2.0;
        /*printf("enII = %lf\n", enII);
        printf("cII = %lf\n", cII);
        printf("u = %lf\n", u);
        printf("dI = %lf\n", dI);
        printf("dII = %lf\n", dII);
        printf("ddI = %lf\n", ddI);
        printf("ddII = %lf\n", ddII);*/

        double UU = max(fabs(dI), fabs(dII));
        UU = max(UU, fabs(ddII));
        UU = max(UU, fabs(ddI));
        if (UU > eps8)
        {
            time = krit * rad / UU;
        }
        else
        {
            time = krit * rad / eps8;
        }


        if (w <= dI)
        {
            //printf("1\n");
            en = enI;
            p = pI;
            r = rI;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w > dI) && (w < ddI))
        {
            //printf("2\n");
            double ce = g1 / g2 * (enI - w) + 2.0 / g2 * cI;
            en = w + ce;
            p = pI * pow((ce / cI), (1.0 / gm));
            r = ga * p / ce / ce;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w >= ddI) && (w <= u))
        {
            //printf("3\n");
            double ce = cI + g1 / 2.0 * (enI - u);
            en = u;
            p = p;
            r = ga * p / ce / ce;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w > u) && (w <= ddII))
        {
            //printf("4\n");
            double ce = cII - g1 / 2.0 * (enII - u);
            en = u;
            p = p;
            r = ga * p / ce / ce;
            te2 = teII2;
            te3 = teII3;
        }
        if ((w > ddII) && (w < dII))
        {
            //printf("5\n");
            double ce = -g1 / g2 * (enII - w) + 2.0 / g2 * cII;
            en = w - ce;
            p = pII * pow((ce / cII), (1.0 / gm));
            r = ga * p / ce / ce;
            te2 = teII2;
            te3 = teII3;
        }
        if (w >= dII)
        {
            //printf("6\n");
            en = enII;
            p = pII;
            r = rII;
            te2 = teII2;
            te3 = teII3;
        }
    }

    //*********VAKUUM ************************************************
    if ((il == -1) && (ip == -1))
    {
        //printf("VAKUUM\n");
        double dI = enI - cI;
        double ddI = enI + 2.0 / gg1 * cI;
        double dII = enII + cII;
        double ddII = enII - 2.0 / gg1 * cII;


        double UU = max(fabs(dI), fabs(dII));
        UU = max(UU, fabs(ddII));
        UU = max(UU, fabs(ddI));
        if (UU > eps8)
        {
            time = krit * rad / UU;
        }
        else
        {
            time = krit * rad / eps8;
        }


        if (w <= dI)
        {
            en = enI;
            p = pI;
            r = rI;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w > dI) && (w < ddI))
        {
            double ce = gg1 / gg2 * (enI - w) + 2.0 / gg2 * cI;
            en = w + ce;
            p = pI * pow((ce / cI), (1.0 / gm));
            r = gga * p / ce / ce;
            te2 = teI2;
            te3 = teI3;
        }
        if ((w >= ddI) && (w <= ddII))
        {
            en = w;
            p = 0.0;
            r = 0.0;
            te2 = 0.0;
            te3 = 0.0;
        }
        if ((w > ddII) && (w < dII))
        {
            double ce = -gg1 / gg2 * (enII - w) + 2.0 / gg2 * cII;
            en = w - ce;
            p = pII * pow((ce / cII), (1.0 / gm));
            r = gga * p / ce / ce;
            te2 = teII2;
            te3 = teII3;
        }
        if (w >= dII)
        {
            en = enII;
            p = pII;
            r = rII;
            te2 = teII2;
            te3 = teII3;
        }
    }


    if (ipiz == 1)
    {
        en = -en;
        w = -w;
    }

    double uo = al * en + al2 * te2 + al3 * te3;
    double vo = be * en + be2 * te2 + be3 * te3;
    double wo = ge * en + ge2 * te2 + ge3 * te3;


    double eo = p / g1 + 0.5 * r * (uo * uo + vo * vo + wo * wo);
    en = al * uo + be * vo + ge * wo;

    Ps.x = (r * (en - w));
    Pu.x = (r * (en - w) * uo + al * p);
    Pu.y = (r * (en - w) * vo + be * p);
    //qqq[3] = (r * (en - w) * wo + ge * p);
    Ps.y = ((en - w) * eo + en * p);


    return time;

}

__device__ void perpendicular(double a1, double a2, double a3, double& b1, double& b2, double& b3, //
    double& c1, double& c2, double& c3, bool t)
{
    if (t == false)
    {
        double A = a1 * a1 + a2 * a2;
        if (A > 0.01 * (A + a3 * a3))
        {
            double B = sqrt(A);
            b1 = -a2 / B;
            b2 = a1 / B;
            b3 = 0.0;
            double C = sqrt(A * (A + a3 * a3));
            c1 = -a1 * a3 / C;
            c2 = -a2 * a3 / C;
            c3 = A / C;
            return;
        }
        A = a1 * a1 + a3 * a3;
        if (A > 0.01 * (A + a2 * a2))
        {
            double B = sqrt(A);
            b1 = -a3 / B;
            b2 = 0.0;
            b3 = a1 / B;
            double C = sqrt(A * (A + a2 * a2));
            c1 = a1 * a2 / C;
            c2 = -A / C;
            c3 = a2 * a3 / C;
            return;
        }
    }
    else
    {
        double A = a1 * a1 + a2 * a2;
        if (A > 0.01)
        {
            double B = sqrt(A);
            b1 = -a2 / B;
            b2 = a1 / B;
            b3 = 0.0;;
            c1 = -a1 * a3 / B;
            c2 = -a2 * a3 / B;
            c3 = A / B;
            return;
        }
        A = a1 * a1 + a3 * a3;
        if (A > 0.01)
        {
            double B = sqrt(A);
            b1 = -a3 / B;
            b2 = 0.0;
            b3 = a1 / B;

            c1 = a1 * a2 / B;
            c2 = -A / B;
            c3 = a2 * a3 / B;
            return;
        }
    }

}

__device__ void newton(const double& enI, const double& pI, const double& rI, const double& enII, const double& pII, const double& rII, //
    const double& w, double& p)
{
    double fI, fIs, fII, fIIs;
    double cI = __dsqrt_rn(ga * pI / rI);
    double cII = __dsqrt_rn(ga * pII / rII);
    double pn = pI * rII * cII + pII * rI * cI + (enI - enII) * rI * cI * rII * cII;
    pn = pn / (rI * cI + rII * cII);

    double pee = pn;

    int kiter = 0;
a1:
    p = pn;
    if (p <= 0.0)
    {
        printf("84645361\n");
    }

    kiter = kiter + 1;

    fI = (p - pI) / (rI * cI * __dsqrt_rn(gp * p / pI + gm));
    fIs = (ga + 1.0) * p / pI + (3.0 * ga - 1.0);
    fIs = fIs / (4.0 * ga * rI * cI * pow((gp * p / pI + gm), (3.0 / 2.0)));

    fII = (p - pII) / (rII * cII * __dsqrt_rn(gp * p / pII + gm));
    fIIs = (ga + 1.0) * p / pII + (3.0 * ga - 1.0);
    fIIs = fIIs / (4.0 * ga * rII * cII * pow((gp * p / pII + gm), (3.0 / 2.0)));


    if (kiter == 1100)
    {
        printf("0137592\n");
    }

    pn = p - (fI + fII - (enI - enII)) / (fIs + fIIs);

    if (fabs(pn / pee - p / pee) >= eps)
    {
        goto a1;
    }

    p = pn;

    return;
}

__device__ void devtwo(const double& enI, const double& pI, const double& rI, const double& enII, const double& pII, const double& rII, //
    const double& w, double& p)
{
    const double epsil = 10e-10;
    double kl, kp, kc, ksi, ksir, um, ksit;
    int kpizd;

    kl = pI;
    kp = pII;


    lev(enI, pI, rI, enII, pII, rII, kl, ksi);
    lev(enI, pI, rI, enII, pII, rII, kp, ksir);

    if (fabs(ksi) <= epsil)
    {
        um = kl;
        goto a1;
    }

    if (fabs(ksir) <= epsil)
    {
        um = kp;
        goto a1;
    }

    kpizd = 0;

a2:
    kpizd = kpizd + 1;

    if (kpizd == 1100)
    {
        printf("121421414\n");
        printf("%lf, %lf,%lf,%lf,%lf,%lf,\n", enI, pI, rI, enII, pII, rII);
    }


    kc = (kl + kp) / 2.0;

    lev(enI, pI, rI, enII, pII, rII, kc, ksit);

    if (fabs(ksit) <= epsil)
    {
        goto a3;
    }

    if ((ksi * ksit) <= 0.0)
    {
        kp = kc;
        ksir = ksit;
    }
    else
    {
        kl = kc;
        ksi = ksit;
    }

    goto a2;

a3:
    um = kc;
a1:

    p = um;

    return;
}

__device__ void lev(const double& enI, const double& pI, const double& rI, const double& enII,//
    const double& pII, const double& rII, double& uuu, double& fee)
{
    double cI = __dsqrt_rn(ga * pI / rI);
    double cII = __dsqrt_rn(ga * pII / rII);

    double fI = (uuu - pI) / (rI * cI * __dsqrt_rn(gp * uuu / pI + gm));

    double fII = 2.0 / g1 * cII * (pow((uuu / pII), gm) - 1.0);

    double f1 = fI + fII;
    double f2 = enI - enII;
    fee = f1 - f2;
    return;
}

__device__ double HLLC_Aleksashov(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double n[3];
    n[0] = n1;
    n[1] = n2;
    n[2] = 0.0;
    //int id_bn = 1;
    //int n_state = 1;
    double FR[8], FL[8];
    double UL[8], UZ[8], UR[8];
    double UZL[8], UZR[8];

    double vL[3], vR[3], bL[3], bR[3];
    double vzL[3], vzR[3], bzL[3], bzR[3];
    double qv[3];
    double aco[3][3];

    double wv = 0.0;
    double r1 = Ls.x;
    double u1 = Lu.x;
    double v1 = Lu.y;
    double w1 = 0.0;
    double p1 = Ls.y;
    double bx1 = 0.0;
    double by1 = 0.0;
    double bz1 = 0.0;


    double r2 = Rs.x;
    double u2 = Ru.x;
    double v2 = Ru.y;
    double w2 = 0.0;
    double p2 = Rs.y;
    double bx2 = 0.0;
    double by2 = 0.0;
    double bz2 = 0.0;

    double ro = (r2 + r1) / 2.0;
    double ap = (p2 + p1) / 2.0;
    double abx = (bx2 + bx1) / 2.0;
    double aby = (by2 + by1) / 2.0;
    double abz = (bz2 + bz1) / 2.0;


    double bk = abx * n[0] + aby * n[1] + abz * n[2];
    double b2 = kv(abx) + kv(aby) + kv(abz);

    double d = b2 - kv(bk);
    aco[0][0] = n[0];
    aco[1][0] = n[1];
    aco[2][0] = n[2];
    if (d > eps)
    {
        d = __dsqrt_rn(d);
        aco[0][1] = (abx - bk * n[0]) / d;
        aco[1][1] = (aby - bk * n[1]) / d;
        aco[2][1] = (abz - bk * n[2]) / d;
        aco[0][2] = (aby * n[2] - abz * n[1]) / d;
        aco[1][2] = (abz * n[0] - abx * n[2]) / d;
        aco[2][2] = (abx * n[1] - aby * n[0]) / d;
    }
    else
    {
        double aix, aiy, aiz;
        if ((fabs(n[0]) < fabs(n[1])) && (fabs(n[0]) < fabs(n[2])))
        {
            aix = 1.0;
            aiy = 0.0;
            aiz = 0.0;
        }
        else if (fabs(n[1]) < fabs(n[2]))
        {
            aix = 0.0;
            aiy = 1.0;
            aiz = 0.0;
        }
        else
        {
            aix = 0.0;
            aiy = 0.0;
            aiz = 1.0;
        }

        double aik = aix * n[0] + aiy * n[1] + aiz * n[2];
        d = __dsqrt_rn(1.0 - kv(aik));
        aco[0][1] = (aix - aik * n[0]) / d;
        aco[1][1] = (aiy - aik * n[1]) / d;
        aco[2][1] = (aiz - aik * n[2]) / d;
        aco[0][2] = (aiy * n[2] - aiz * n[1]) / d;
        aco[1][2] = (aiz * n[0] - aix * n[2]) / d;
        aco[2][2] = (aix * n[1] - aiy * n[0]) / d;
    }

    for (int i = 0; i < 3; i++)
    {
        vL[i] = aco[0][i] * u1 + aco[1][i] * v1 + aco[2][i] * w1;
        vR[i] = aco[0][i] * u2 + aco[1][i] * v2 + aco[2][i] * w2;
        bL[i] = aco[0][i] * bx1 + aco[1][i] * by1 + aco[2][i] * bz1;
        bR[i] = aco[0][i] * bx2 + aco[1][i] * by2 + aco[2][i] * bz2;
    }

    double aaL = bL[0] / __dsqrt_rn(r1);
    double b2L = kv(bL[0]) + kv(bL[1]) + kv(bL[2]);
    double b21 = b2L / r1;
    double cL = __dsqrt_rn(ga * p1 / r1);
    double qp = __dsqrt_rn(b21 + cL * (cL + 2.0 * aaL));
    double qm = __dsqrt_rn(b21 + cL * (cL - 2.0 * aaL));
    double cfL = (qp + qm) / 2.0;
    double ptL = p1 + b2L / 2.0;

    double aaR = bR[0] / __dsqrt_rn(r2);
    double b2R = kv(bR[0]) + kv(bR[1]) + kv(bR[2]);
    double b22 = b2R / r2;
    double cR = __dsqrt_rn(ga * p2 / r2);
    qp = __dsqrt_rn(b22 + cR * (cR + 2.0 * aaR));
    qm = __dsqrt_rn(b22 + cR * (cR - 2.0 * aaR));
    double cfR = (qp + qm) / 2.0;
    double ptR = p2 + b2R / 2.0;

    double aC = (aaL + aaR) / 2.0;
    double b2o = (b22 + b21) / 2.0;
    double cC = __dsqrt_rn(ga * ap / ro);
    qp = __dsqrt_rn(b2o + cC * (cC + 2.0 * aC));
    qm = __dsqrt_rn(b2o + cC * (cC - 2.0 * aC));
    double cfC = (qp + qm) / 2.0;
    double vC1 = (vL[0] + vR[0]) / 2.0;

    double SL = min((vL[0] - cfL), (vR[0] - cfR));
    double SR = max((vL[0] + cfL), (vR[0] + cfR));

    double suR = SR - vR[0];
    double suL = SL - vL[0];
    double SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);

    if (SR <= SL)
    {
        printf("231\n");
    }

    double SM00 = SM;
    double SR00 = SR;
    double SL00 = SL;
    double SM01, SR01, SL01;
    if ((SM00 >= SR00) || (SM00 <= SL00))
    {
        SL = min((vL[0] - cfL), (vR[0] - cfR));
        SR = max((vL[0] + cfL), (vR[0] + cfR));
        suR = SR - vR[0];
        suL = SL - vL[0];
        SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);
        SM01 = SM;
        SR01 = SR;
        SL01 = SL;
        if ((SM01 >= SR01) || (SM01 <= SL01))
        {
            printf("251\n");
        }
    }


    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double upt1 = (kv(u1) + kv(v1) + kv(w1)) / 2.0;
    double sbv1 = u1 * bx1 + v1 * by1 + w1 * bz1;

    double upt2 = (kv(u2) + kv(v2) + kv(w2)) / 2.0;
    double sbv2 = u2 * bx2 + v2 * by2 + w2 * bz2;

    double e1 = p1 / g1 + r1 * upt1 + b2L / 2.0;
    double e2 = p2 / g1 + r2 * upt2 + b2R / 2.0;

    FL[0] = r1 * vL[0];
    FL[1] = r1 * vL[0] * vL[0] + ptL - kv(bL[0]);
    FL[2] = r1 * vL[0] * vL[1] - bL[0] * bL[1];
    FL[3] = r1 * vL[0] * vL[2] - bL[0] * bL[2];
    FL[4] = (e1 + ptL) * vL[0] - bL[0] * sbv1;
    FL[5] = 0.0;
    FL[6] = vL[0] * bL[1] - vL[1] * bL[0];
    FL[7] = vL[0] * bL[2] - vL[2] * bL[0];

    FR[0] = r2 * vR[0];
    FR[1] = r2 * vR[0] * vR[0] + ptR - kv(bR[0]);
    FR[2] = r2 * vR[0] * vR[1] - bR[0] * bR[1];
    FR[3] = r2 * vR[0] * vR[2] - bR[0] * bR[2];
    FR[4] = (e2 + ptR) * vR[0] - bR[0] * sbv2;
    FR[5] = 0.0;
    FR[6] = vR[0] * bR[1] - vR[1] * bR[0];
    FR[7] = vR[0] * bR[2] - vR[2] * bR[0];

    UL[0] = r1;
    UL[4] = e1;
    UR[0] = r2;
    UR[4] = e2;


    for (int ik = 0; ik < 3; ik++)
    {
        UL[ik + 1] = r1 * vL[ik];
        UL[ik + 5] = bL[ik];
        UR[ik + 1] = r2 * vR[ik];
        UR[ik + 5] = bR[ik];
    }

    for (int ik = 0; ik < 8; ik++)
    {
        UZ[ik] = (SR * UR[ik] - SL * UL[ik] + FL[ik] - FR[ik]) / (SR - SL);
    }

    double suRm = suR / (SR - SM);
    double suLm = suL / (SL - SM);
    double rzR = r2 * suRm;
    double rzL = r1 * suLm;
    vzR[0] = SM;
    vzL[0] = SM;
    double ptzR = ptR + r2 * suR * (SM - vR[0]);
    double ptzL = ptL + r1 * suL * (SM - vL[0]);
    double ptz = (ptzR + ptzL) / 2.0;
    bzR[0] = UZ[5];
    bzL[0] = UZ[5];

    vzR[1] = UZ[2] / UZ[0];
    vzR[2] = UZ[3] / UZ[0];
    vzL[1] = vzR[1];
    vzL[2] = vzR[2];

    vzR[1] = vR[1] + UZ[5] * (bR[1] - UZ[6]) / suR / r2;
    vzR[2] = vR[2] + UZ[5] * (bR[2] - UZ[7]) / suR / r2;
    vzL[1] = vL[1] + UZ[5] * (bL[1] - UZ[6]) / suL / r1;
    vzL[2] = vL[2] + UZ[5] * (bL[2] - UZ[7]) / suL / r1;

    bzR[1] = UZ[6];
    bzR[2] = UZ[7];
    bzL[1] = bzR[1];
    bzL[2] = bzR[2];

    double sbvz = (UZ[5] * UZ[1] + UZ[6] * UZ[2] + UZ[7] * UZ[3]) / UZ[0];

    double ezR = e2 * suRm + (ptz * SM - ptR * vR[0] + UZ[5] * (sbv2 - sbvz)) / (SR - SM);
    double ezL = e1 * suLm + (ptz * SM - ptL * vL[0] + UZ[5] * (sbv1 - sbvz)) / (SL - SM);

    if (fabs(UZ[5]) < eps)
    {
        vzR[1] = vR[1];
        vzR[2] = vR[2];
        vzL[1] = vL[1];
        vzL[2] = vL[2];
        bzR[1] = bR[1] * suRm;
        bzR[2] = bR[2] * suRm;
        bzL[1] = bL[1] * suLm;
        bzL[2] = bL[2] * suLm;
    }
    UZL[0] = rzL;
    UZL[4] = ezL;
    UZR[0] = rzR;
    UZR[4] = ezR;

    for (int ik = 0; ik < 3; ik++)
    {
        UZL[ik + 1] = vzL[ik] * rzL;
        UZL[ik + 5] = bzL[ik];
        UZR[ik + 1] = vzR[ik] * rzR;
        UZR[ik + 5] = bzR[ik];
    }

    if (SL > wv)
    {
        Ps.x = FL[0] - wv * UL[0];
        Ps.y = FL[4] - wv * UL[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FL[ik] - wv * UL[ik];
        }
    }
    else if ( (SL <= wv) && (SM >= wv) )
    {
        Ps.x = FL[0] + SL * (rzL - r1) - wv * UZL[0];
        Ps.y = FL[4] + SL * (ezL - e1) - wv * UZL[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
        }
    }
    else if ((SM <= wv)&&(SR >= wv))
    {
        Ps.x = FR[0] + SR * (rzR - r2) - wv * UZR[0];
        Ps.y = FR[4] + SR * (ezR - e2) - wv * UZR[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
        }
    }
    else if (SR < wv)
    {
        Ps.x = FR[0] - wv * UR[0];
        Ps.y = FR[4] - wv * UR[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FR[ik] + - wv * UR[ik];
        }
    }
    else
    {
        printf("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD\n");
    }


    Pu.x = aco[0][0] * qv[0] + aco[0][1] * qv[1] + aco[0][2] * qv[2];
    Pu.y = aco[1][0] * qv[0] + aco[1][1] * qv[1] + aco[1][2] * qv[2];

    return time;
}

__device__ double HLLC_Aleksashov2(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double r1 = Ls.x;
    double p1 = Ls.y;
    double u1 = Lu.x;
    double v1 = Lu.y;

    double r2 = Rs.x;
    double p2 = Rs.y;
    double u2 = Ru.x;
    double v2 = Ru.y;



    // c------ - n_state = 2 - two - state(3 speed) HLLC(Contact Discontinuity)


    double ro = (r2 + r1) / 2.0;
    double ap = (p2 + p1) / 2.0;

    double aco[2][2];
    aco[0][0] = n1;
    aco[1][0] = n2;
    aco[0][1] = -n2;
    aco[1][1] = n1;

    //aco(1, 1) = al
    //aco(2, 1) = be
    //aco(3, 1) = ge

    double vL[2];
    double vR[2];

    vL[0] = aco[0][0] * u1 + aco[1][0] * v1;
    vL[1] = aco[0][1] * u1 + aco[1][1] * v1;
    vR[0] = aco[0][0] * u2 + aco[1][0] * v2;
    vR[1] = aco[0][1] * u2 + aco[1][1] * v2;

    /*if ((r1 <= eps) || (r2 <= eps) || (p1 <= 0) || (p2 <= 0) )
    {
        printf("EREREREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n");
    }*/

    double cL = __dsqrt_rn(ga * p1 / r1);
    double cR = __dsqrt_rn(ga * p2 / r2);
    double cC = __dsqrt_rn(ga * ap / ro);

    double SL, SR;

    //SL = min((vL[0] - cL), (vC1 - cC));
    //SR = max((vR[0] + cR), (vC1 + cC));

    SL = min((vL[0] - cL), (vR[0] - cR));
    SR = max((vL[0] + cL), (vR[0] + cR));

    double t = 10000000;
    t = min(t, krit * rad / max(fabs(SL), fabs(SR)));

    double suR = SR - vR[0];
    double suL = SL - vL[0];
    double SM = 0.0;
    if (fabs(suR * r2 - suL * r1) > 0)
    {
        SM = (suR * r2 * vR[0] - p2 + p1 - suL * r1 * vL[1]) / (suR * r2 - suL * r1);
    }

    if (SR < SL)
    {
        printf("12102022020,    ERROR in HLCC_Alexashov  \n");
    }

    double upt1 = (u1 * u1 + v1 * v1) / 2.0;
    double upt2 = (u2 * u2 + v2 * v2) / 2.0;
    double e1 = p1 / g1 + r1 * upt1;
    double e2 = p2 / g1 + r2 * upt2;
    double FL[4];
    double FR[4];
    double UL[4];
    double UR[4];

    FL[0] = r1 * vL[0];
    FL[1] = r1 * vL[0] * vL[0] + p1;
    FL[2] = r1 * vL[0] * vL[1];
    FL[3] = (e1 + p1) * vL[0];

    FR[0] = r2 * vR[0];
    FR[1] = r2 * vR[0] * vR[0] + p2;
    FR[2] = r2 * vR[0] * vR[1];
    FR[3] = (e2 + p2) * vR[0];

    UL[0] = r1;
    UL[3] = e1;
    UR[0] = r2;
    UR[3] = e2;

    UL[1] = r1 * vL[0];
    UL[2] = r1 * vL[1];
    UR[1] = r2 * vR[0];
    UR[2] = r2 * vR[1];


    double suRm = suR / (SR - SM);
    double suLm = suL / (SL - SM);
    double rzR = r2 * suRm;
    double rzL = r1 * suLm;

    double ptzR = p2 + r2 * suR * (SM - vR[0]);
    double ptzL = p1 + r1 * suL * (SM - vL[0]);
    double ptz = (ptzR + ptzL) / 2.0;
    double vzR[2];
    double vzL[2];

    vzR[0] = SM;
    vzL[0] = SM;
    vzR[1] = vR[1];
    vzL[1] = vL[1];

    double ezR = e2 * suRm + (ptz * SM - p2 * vR[0]) / (SR - SM);
    double ezL = e1 * suLm + (ptz * SM - p1 * vL[0]) / (SL - SM);

    double UZL[4];
    double UZR[4];

    UZL[0] = rzL;
    UZL[3] = ezL;
    UZR[0] = rzR;
    UZR[3] = ezR;

    for (int i = 1; i < 3; i++)
    {
        UZL[i] = vzL[i - 1] * rzL;
        UZR[i] = vzR[i - 1] * rzR;
    }

    double qv[2];

    if (SL > 0.0)
    {
        Ps.x = FL[0];
        Ps.y = FL[3];
        qv[0] = FL[1];
        qv[1] = FL[2];
    }
    else if ((SL <= 0.0) && (SM >= 0.0))
    {
        Ps.x = FL[0] + SL * (rzL - r1);
        Ps.y = FL[3] + SL * (ezL - e1);
        qv[0] = FL[1] + SL * (UZL[1] - UL[1]);
        qv[1] = FL[2] + SL * (UZL[2] - UL[2]);
    }
    else if ((SM <= 0.0) && (SR >= 0.0))
    {
        Ps.x = FR[0] + SR * (rzR - r2);
        Ps.y = FR[3] + SR * (ezR - e2);
        qv[0] = FR[1] + SR * (UZR[1] - UR[1]);
        qv[1] = FR[2] + SR * (UZR[2] - UR[2]);
    }
    else if (SR < 0.0)
    {
        Ps.x = FR[0];
        Ps.y = FR[3];
        qv[0] = FR[1];
        qv[1] = FR[2];
    }
    else
    {
        printf("21702022020,    ERROR in HLCC_Alexashov  \n");
        printf(" SL = %lf, SM = %lf, SR = %lf\n", SL, SM, SR);
        printf(" r1 = %lf, p1 = %lf, u1 = %lf, v1 = %lf\n", r1, p1, u1, v1);
        printf(" r2 = %lf, p2 = %lf, u2 = %lf, v2 = %lf\n", r2, p2, u2, v2);
        printf(" vl[0] = %lf, cL = %lf, vR[0] = %lf, cR = %lf\n", vL[0], cL, vR[0], cR);
        /*SL = min((vL[0] - cL), (vR[0] - cR));
        SR = max((vL[0] + cL), (vR[0] + cR));*/
    }

    Pu.x = aco[0][0] * qv[0] + aco[0][1] * qv[1];
    Pu.y = aco[1][0] * qv[0] + aco[1][1] * qv[1];

    return t;
}

__device__ double HLLC_Korolkov(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double ro_L = Ls.x;
    double p_L = Ls.y;
    double v1_L = Lu.x;
    double v2_L = Lu.y;

    double ro_R = Rs.x;
    double p_R = Rs.y;
    double v1_R = Ru.x;
    double v2_R = Ru.y;

    double e_L, e_R;
    double Vkv_L, Vkv_R;
    double c_L, c_R;

    Vkv_L = v1_L * v1_L + v2_L * v2_L;
    Vkv_R = v1_R * v1_R + v2_R * v2_R;

    c_L = __dsqrt_rn(ggg * p_L / ro_L);
    c_R = __dsqrt_rn(ggg * p_R / ro_R);
    e_L = p_L / (ggg - 1.0) + ro_L * Vkv_L / 2.0;  /// Полная энергия слева
    e_R = p_R / (ggg - 1.0) + ro_R * Vkv_R / 2.0;  /// Полная энергия справа

    double Vn_L = v1_L * n1 + v2_L * n2;
    double Vn_R = v1_R * n1 + v2_R * n2;

    double D_L = min(Vn_L, Vn_R) - max(c_L, c_R);
    double D_R = max(Vn_L, Vn_R) + max(c_L, c_R);
    /*double D_L = min(Vn_L - c_L, Vn_R - c_R);
    double D_R = max(Vn_L + c_L, Vn_R + c_R);*/
    double t = 10000000;
    t = min(t, krit * rad / max(fabs(D_L), fabs(D_R)));

    double fx1 = ro_L * v1_L;
    double fx2 = ro_L * v1_L * v1_L + p_L;
    double fx3 = ro_L * v1_L * v2_L;
    double fx5 = (e_L + p_L) * v1_L;

    double fy1 = ro_L * v2_L;
    double fy2 = ro_L * v1_L * v2_L;
    double fy3 = ro_L * v2_L * v2_L + p_L;
    double fy5 = (e_L + p_L) * v2_L;

    double fl_1 = fx1 * n1 + fy1 * n2;
    double fl_2 = fx2 * n1 + fy2 * n2;
    double fl_3 = fx3 * n1 + fy3 * n2;
    double fl_5 = fx5 * n1 + fy5 * n2;

    if (D_L > Omega)
    {
        Ps.x = fl_1; /// Нужно будет домножить на площадь грани и шаг по времени
        Pu.x = fl_2;
        Pu.y = fl_3;
        Ps.y = fl_5;
        return t;
    }

    double hx1 = ro_R * v1_R;
    double hx2 = ro_R * v1_R * v1_R + p_R;
    double hx3 = ro_R * v1_R * v2_R;
    double hx5 = (e_R + p_R) * v1_R;

    double hy1 = ro_R * v2_R;
    double hy2 = ro_R * v1_R * v2_R;
    double hy3 = ro_R * v2_R * v2_R + p_R;
    double hy5 = (e_R + p_R) * v2_R;

    double fr_1 = hx1 * n1 + hy1 * n2;
    double fr_2 = hx2 * n1 + hy2 * n2;
    double fr_3 = hx3 * n1 + hy3 * n2;
    double fr_5 = hx5 * n1 + hy5 * n2;

    if (D_R < Omega)
    {
        Ps.x = fr_1; /// Нужно будет домножить на площадь грани и шаг по времени
        Pu.x = fr_2;
        Pu.y = fr_3;
        Ps.y = fr_5;
        return t;
    }

    double u_L = Vn_L;
    double u_R = Vn_R;

    double D_C = ((D_R - u_R) * ro_R * u_R - (D_L - u_L) * ro_L * u_L - p_R + p_L) / ((D_R - u_R) * ro_R - (D_L - u_L) * ro_L);

    double roro_L = ro_L * ((D_L - u_L) / (D_L - D_C));
    double roro_R = ro_R * ((D_R - u_R) / (D_R - D_C));

    /// Находим давление в центральной области (оно одинаковое слева и справа)
    double P_T = (p_L * ro_R * (u_R - D_R) - p_R * ro_L * (u_L - D_L) - ro_L * ro_R * (u_L - D_L) * (u_R - D_R) * (u_R - u_L)) / (ro_R * (u_R - D_R) - ro_L * (u_L - D_L));

    if (D_L <= Omega && D_C >= Omega)  /// Попали во вторую область (слева)
    {
        double Vx = v1_L + (D_C - Vn_L) * n1;
        double Vy = v2_L + (D_C - Vn_L) * n2;

        double ee_L = P_T / (ggg - 1.0) + roro_L * (Vx * Vx + Vy * Vy) / 2.0;
        //double ee_L = e_L - ((P_T - p_L)/2.0)*(1/roro_L - 1/ro_L);
        /*double ee_L = ((D_L - u_L) * e_L - p_L * u_L + P_T * D_C) / (D_L - D_C);*/

        double dq1 = roro_L - ro_L;
        double dq2 = roro_L * Vx - ro_L * v1_L;
        double dq3 = roro_L * Vy - ro_L * v2_L;
        double dq5 = ee_L - e_L;

        Ps.x = D_L * dq1 + fl_1; /// Нужно будет домножить на площадь грани и шаг по времени
        Pu.x = D_L * dq2 + fl_2;
        Pu.y = D_L * dq3 + fl_3;
        Ps.y = D_L * dq5 + fl_5;
        return t;
    }
    else if (D_R >= Omega && D_C <= Omega)  /// Попали во вторую область (справа)
    {
        double Vx = v1_R + (D_C - Vn_R) * n1;
        double Vy = v2_R + (D_C - Vn_R) * n2;

        double ee_R = P_T / (ggg - 1.0) + roro_R * (Vx * Vx + Vy * Vy) / 2.0;
        /*double ee_R = ((D_R - u_R) * e_R - p_R * u_R + P_T * D_C) / (D_R - D_C);*/

        double dq1 = roro_R - ro_R;
        double dq2 = roro_R * Vx - ro_R * v1_R;
        double dq3 = roro_R * Vy - ro_R * v2_R;
        double dq5 = ee_R - e_R;

        Ps.x = D_R * dq1 + fr_1; /// Нужно будет домножить на площадь грани и шаг по времени
        Pu.x = D_R * dq2 + fr_2;
        Pu.y = D_R * dq3 + fr_3;
        Ps.y = D_R * dq5 + fr_5;
        return t;
    }
    return t;
}

__device__ double HLL(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad)
{
    double ro_L = Ls.x;
    double p_L = Ls.y;
    double v1_L = Lu.x;
    double v2_L = Lu.y;

    double ro_R = Rs.x;
    double p_R = Rs.y;
    double v1_R = Ru.x;
    double v2_R = Ru.y;

    double e_L, e_R;
    double Vkv_L, Vkv_R;
    double c_L, c_R;

    Vkv_L = v1_L * v1_L + v2_L * v2_L;
    Vkv_R = v1_R * v1_R + v2_R * v2_R;
    if (ro_L <= 0)
    {
        c_L = 0.0;
    }
    else
    {
        c_L = sqrt(ggg * p_L / ro_L);
    }

    if (ro_R <= 0)
    {
        c_R = 0.0;
    }
    else
    {
        c_R = sqrt(ggg * p_R / ro_R);
    }
    e_L = p_L / (ggg - 1.0) + ro_L * Vkv_L / 2.0;  /// Полная энергия слева
    e_R = p_R / (ggg - 1.0) + ro_R * Vkv_R / 2.0;  /// Полная энергия справа

    double Vn_L = v1_L * n1 + v2_L * n2;
    double Vn_R = v1_R * n1 + v2_R * n2;
    double D_L = my_min(Vn_L, Vn_R) - my_max(c_L, c_R);
    double D_R = my_max(Vn_L, Vn_R) + my_max(c_L, c_R);
    double t = 10000000;
    t = my_min(t, krit * rad / my_max(fabs(D_L), fabs(D_R)));

    double fx1 = ro_L * v1_L;
    double fx2 = ro_L * v1_L * v1_L + p_L;
    double fx3 = ro_L * v1_L * v2_L;
    double fx5 = (e_L + p_L) * v1_L;

    double fy1 = ro_L * v2_L;
    double fy2 = ro_L * v1_L * v2_L;
    double fy3 = ro_L * v2_L * v2_L + p_L;
    double fy5 = (e_L + p_L) * v2_L;

    double fl_1 = fx1 * n1 + fy1 * n2;
    double fl_2 = fx2 * n1 + fy2 * n2;
    double fl_3 = fx3 * n1 + fy3 * n2;
    double fl_5 = fx5 * n1 + fy5 * n2;

    /*double U_L1 = ro_L;
    double U_L2 = ro_L * v1_L;
    double U_L3 = ro_L * v2_L;
    double U_L5 = e_L;*/

    if (D_L > Omega)
    {
        Ps.x = fl_1; /// Нужно будет домножить на площадь грани и шаг по времени
        Pu.x = fl_2;
        Pu.y = fl_3;
        Ps.y = fl_5;
        return t;
    }
    else
    {
        double hx1 = ro_R * v1_R;
        double hx2 = ro_R * v1_R * v1_R + p_R;
        double hx3 = ro_R * v1_R * v2_R;
        double hx5 = (e_R + p_R) * v1_R;

        double hy1 = ro_R * v2_R;
        double hy2 = ro_R * v1_R * v2_R;
        double hy3 = ro_R * v2_R * v2_R + p_R;
        double hy5 = (e_R + p_R) * v2_R;

        double fr_1 = hx1 * n1 + hy1 * n2;
        double fr_2 = hx2 * n1 + hy2 * n2;
        double fr_3 = hx3 * n1 + hy3 * n2;
        double fr_5 = hx5 * n1 + hy5 * n2;

        /*double U_R1 = ro_R;
        double U_R2 = ro_R * v1_R;
        double U_R3 = ro_R * v2_R;
        double U_R5 = e_R;*/

        if (D_R < Omega)
        {
            Ps.x = fr_1; /// Нужно будет домножить на площадь грани и шаг по времени
            Pu.x = fr_2;
            Pu.y = fr_3;
            Ps.y = fr_5;
            return t;
        }
        else
        {
            double dq1 = ro_R - ro_L;
            double dq2 = ro_R * v1_R - ro_L * v1_L;
            double dq3 = ro_R * v2_R - ro_L * v2_L;
            double dq5 = e_R - e_L;

            //double U1 = (D_R * U_R1 - D_L * U_L1 - fr_1 + fl_1) / (D_R - D_L);
            //double U2 = (D_R * U_R2 - D_L * U_L2 - fr_2 + fl_2) / (D_R - D_L);
            //double U3 = (D_R * U_R3 - D_L * U_L3 - fr_3 + fl_3) / (D_R - D_L);
            //double U5 = (D_R * U_R5 - D_L * U_L5 - fr_5 + fl_5) / (D_R - D_L);


            Ps.x = (D_R * fl_1 - D_L * fr_1 + D_L * D_R * dq1) / (D_R - D_L); /// Нужно будет домножить на площадь грани и шаг по времени
            Pu.x = (D_R * fl_2 - D_L * fr_2 + D_L * D_R * dq2) / (D_R - D_L);
            Pu.y = (D_R * fl_3 - D_L * fr_3 + D_L * D_R * dq3) / (D_R - D_L);
            Ps.y = (D_R * fl_5 - D_L * fr_5 + D_L * D_R * dq5) / (D_R - D_L);
            return t;
        }
    }
}

__global__ void funk_time(double* T, double* T_do, double* TT, int* i)
{
    *T_do = *T;
    *TT = *TT + *T_do;
    *T = 10000000;
    *i = *i + 1;
    if (*i % 5000 == 0)
    {
        printf("i = %d,  TT = %lf \n", *i, *TT);
    }
    return;
}

//__global__ void add(double2* s, double2* u, double2* s2, double2* u2, double* T, double* T_do, int method)
//{
//    int index = blockIdx.x * blockDim.x + threadIdx.x;   // Глобальный индекс текущей ячейки (текущего потока)
//    int n = index % N;                                   // номер ячейки по x (от 0)
//    int m = (index - n) / N;                             // номер ячейки по y (от 0)
//    double y = y_min + m * (y_max) / (M);
//    double x = x_min + n * (x_max - x_min) / (N - 1);
//    double dist = __dsqrt_rn(x * x + y * y);
//
//    // Нужно оптимизировать работу с памятью, для этого создавать shared массив и вытаскивать данные из глобальной памяти в него
//    __shared__ double2 a[THREADS_PER_BLOCK + 2][3];
//    __shared__ double2 b[THREADS_PER_BLOCK + 2][3];
//
//    if (m == 0)
//    {
//        a[threadIdx.x + 1][1] = s[index];
//        a[threadIdx.x + 1][2] = s[(m + 1) * N + n];
//        if ((threadIdx.x == 0) && (n != 0))
//        {
//            a[0][1] = s[index - 1];
//        }
//        if ((threadIdx.x == THREADS_PER_BLOCK - 1) && (n != N - 1))
//        {
//            a[THREADS_PER_BLOCK + 1][1] = s[index + 1];
//        }
//
//        b[threadIdx.x + 1][1] = u[index];
//        b[threadIdx.x + 1][2] = u[(m + 1) * N + n];
//        if ((threadIdx.x == 0) && (n != 0))
//        {
//            b[0][1] = u[index - 1];
//        }
//        if ((threadIdx.x == THREADS_PER_BLOCK - 1) && (n != N - 1))
//        {
//            b[THREADS_PER_BLOCK + 1][1] = u[index + 1];
//        }
//    }
//    else if (m == M - 1)
//    {
//        a[threadIdx.x + 1][1] = s[index];
//        b[threadIdx.x + 1][1] = u[index];
//    }
//    else
//    {
//        a[threadIdx.x + 1][1] = s[index];
//        a[threadIdx.x + 1][0] = s[(m - 1) * N + n];
//        a[threadIdx.x + 1][2] = s[(m + 1) * N + n];
//        if ((threadIdx.x == 0) && (n != 0))
//        {
//            a[0][1] = s[index - 1];
//        }
//        if ((threadIdx.x == THREADS_PER_BLOCK - 1) && (n != N - 1))
//        {
//            a[THREADS_PER_BLOCK + 1][1] = s[index + 1];
//        }
//
//        b[threadIdx.x + 1][1] = u[index];
//        b[threadIdx.x + 1][0] = u[(m - 1) * N + n];
//        b[threadIdx.x + 1][2] = u[(m + 1) * N + n];
//        if ((threadIdx.x == 0) && (n != 0))
//        {
//            b[0][1] = u[index - 1];
//        }
//        if ((threadIdx.x == THREADS_PER_BLOCK - 1) && (n != N - 1))
//        {
//            b[THREADS_PER_BLOCK + 1][1] = u[index + 1];
//        }
//    }
//    // Скачали данные.  Синхронизируемся
//    __syncthreads();
//
//
//
//    //Обработка особых ячеек
//    if ((n == N - 1) || (m == M - 1) || (dist < 110)) // Жёсткие граничные условия
//    {
//        // В этих ячейках значения параметров зафиксированы и не меняются с течением времени)
//        s2[index].x = a[threadIdx.x + 1][1].x;
//        s2[index].y = a[threadIdx.x + 1][1].y;
//        u2[index].x = b[threadIdx.x + 1][1].x;
//        u2[index].y = b[threadIdx.x + 1][1].y;
//        return;
//    }
//
//    double2 s_1, s_2, s_3, s_4, s_5, u_1, u_2, u_3, u_4, u_5;      // Переменные всех соседей и самой ячейки
//    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
//        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // Вектора потоков
//    double tmin = 1000;
//
//    s_1 = a[threadIdx.x + 1][1];
//    u_1 = b[threadIdx.x + 1][1];
//    s_2 = a[threadIdx.x + 2][1];
//    u_2 = b[threadIdx.x + 2][1];
//    s_3 = a[threadIdx.x + 1][0];
//    u_3 = b[threadIdx.x + 1][0];
//    s_4 = a[threadIdx.x][1];
//    u_4 = b[threadIdx.x][1];
//    s_5 = a[threadIdx.x + 1][2];
//    u_5 = b[threadIdx.x + 1][2];
//
//    if ((n == 0) && (m == 0))
//    {
//        s_3 = s_1;
//        u_3.x = u_1.x;
//        u_3.y = -u_1.y;
//        s_4 = s_1;
//        u_4 = u_1;
//    }
//    else if (n == 0)
//    {
//        s_4 = s_1;
//        u_4 = u_1;
//    }
//    else if (m == 0)
//    {
//        s_3 = s_1;
//        u_3.x = u_1.x;
//        u_3.y = -u_1.y;
//    }
//
//
//    if (method == 0)
//    {
//        tmin = min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dy));
//        tmin = min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dx));
//        tmin = min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dy));
//        tmin = min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dx));
//    }
//    else if (method == 1)
//    {
//        tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dy));
//        tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dx));
//        tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dy));
//        tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dx));
//    }
// 
// 
//
//    if (*T > tmin)
//    {
//        __threadfence();
//        *T = tmin;
//    }
//
//    double2 PS = { 0,0 };
//    double2 PU = { 0,0 };
//
//    /*if ((x > 400) &&( x < 405) && (y < 5))
//    {
//        printf("%lf, %lf, %lf, %lf, %lf\n", Ps12.x, Ps13.x, Ps14.x, Ps15.x, Ps12.x* dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx);
//    }*/
//
//    PS.x = Ps12.x * dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx;
//    PS.y = Ps12.y * dy + Ps13.y * dx + Ps14.y * dy + Ps15.y * dx;
//    PU.x = Pu12.x * dy + Pu13.x * dx + Pu14.x * dy + Pu15.x * dx;
//    PU.y = Pu12.y * dy + Pu13.y * dx + Pu14.y * dy + Pu15.y * dx;
//
//    double dV = dx * dy;
//
//    s2[index].x = s[index].x - (*T_do / dV) * PS.x - *T_do * s[index].x * u[index].y / y;
//    if (s2[index].x <= 0)
//    {
//        printf("Problems! x = %lf, y = %lf, ro = %lf\n", x, y, s2[index].x);
//        s2[index].x = 0.000001;
//    }
//    u2[index].x = (s[index].x * u[index].x - (*T_do / dV) * PU.x - *T_do * s[index].x * u[index].y * u[index].x / y) / s2[index].x;
//    u2[index].y = (s[index].x * u[index].y - (*T_do / dV) * PU.y - *T_do * s[index].x * u[index].y * u[index].y / y) / s2[index].x;
//    s2[index].y = (((s[index].y / (ggg - 1) + s[index].x * (u[index].x * u[index].x + u[index].y * u[index].y) * 0.5) - (*T_do / dV) * PS.y - //
//        *T_do * u[index].y * (ggg * s[index].y / (ggg - 1) + s[index].x * (u[index].x * u[index].x + u[index].y * u[index].y) * 0.5) / y) - //
//        0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1);
//    if (s2[index].y <= 0)
//    {
//        s2[index].y = 0.000001;
//    }
//}
//

__global__ void Ker_Dekard(double2* s, double2* u, double2* s2, double2* u2, double* T, double* T_do, int method)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;   // Глобальный индекс текущей ячейки (текущего потока)
    int n = index % N;                                   // номер ячейки по x (от 0)
    int m = (index - n) / N;                             // номер ячейки по y (от 0)
    double y = y_min + m * (y_max) / (M);
    double x = x_min + n * (x_max - x_min) / (N - 1);
    //double dist = __dsqrt_rn(x * x + y * y);

    double2 s_1, s_2, s_3, s_4, s_5, u_1, u_2, u_3, u_4, u_5;      // Переменные всех соседей и самой ячейки
    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // Вектора потоков
    double tmin = 1000;

    s_1 = s[index];
    u_1 = u[index];
    if ((n == N - 1)||(m == M-1)) // Жёсткие граничные условия
    {
        // В этих ячейках значения параметров зафиксированы и не меняются с течением времени)
        //s2[index] = s_1;
        //u2[index] = u_1;
        return;
    }
    s_2 = s[(m)*N + n + 1];
    u_2 = u[(m)*N + n + 1];

    if ((n == 0))
    {
        s_4 = s_1;
        u_4 = u_1;
    }
    else
    {
        s_4 = s[(m)*N + n - 1];
        u_4 = u[(m)*N + n - 1];
    }

    if ((m == 0))
    {
        s_3 = s_1;
        u_3 = u_1;
    }
    else
    {
        s_3 = s[(m - 1) * N + (n)];
        u_3 = u[(m - 1) * N + (n)];
    }

    if ((m == M - 1))
    {
        s_5 = s_1;
        u_5 = u_1;
    }
    else
    {
        s_5 = s[(m + 1) * N + (n)];
        u_5 = u[(m + 1) * N + (n)];
    }


    if (method == 0)
    {
        tmin = min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dy));
        tmin = min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dx));
        tmin = min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dy));
        tmin = min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dx));
    }
    else if (method == 1)
    {
        if (x + dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dy));
        }
        if (y - dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dx));
        }
        if (x - dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dy));
        }
        if (y + dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dx));
        }
    }


    if (*T > tmin)
    {
        //__threadfence();
        *T = tmin;
    }

    double2 PS = { 0,0 };
    double2 PU = { 0,0 };

    /*if ((x > 400) &&( x < 405) && (y < 5))
    {
        printf("%lf, %lf, %lf, %lf, %lf\n", Ps12.x, Ps13.x, Ps14.x, Ps15.x, Ps12.x* dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx);
    }*/

    PS.x = Ps12.x * dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx;
    PS.y = Ps12.y * dy + Ps13.y * dx + Ps14.y * dy + Ps15.y * dx;
    PU.x = Pu12.x * dy + Pu13.x * dx + Pu14.x * dy + Pu15.x * dx;
    PU.y = Pu12.y * dy + Pu13.y * dx + Pu14.y * dy + Pu15.y * dx;

    double dV = dx * dy;

    s2[index].x = s[index].x - (*T_do / dV) * PS.x;
    if (s2[index].x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf\n", x, y, s2[index].x);
        s2[index].x = 0.0001;
    }
    u2[index].x = (s[index].x * u[index].x - (*T_do / dV) * PU.x) / s2[index].x;
    u2[index].y = (s[index].x * u[index].y - (*T_do / dV) * PU.y) / s2[index].x;
    s2[index].y = (((s[index].y / (ggg - 1) + s[index].x * (u[index].x * u[index].x + u[index].y * u[index].y) * 0.5) - (*T_do / dV) * PS.y) - //
        0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1);
    if (s2[index].y <= 0)
    {
        s2[index].y = 0.000001;
    }
}

__global__ void add2(double2* s, double2* u, double2* s2, double2* u2, double* T, double* T_do, int method, int step)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;   // Глобальный индекс текущей ячейки (текущего потока)
    int n = index % N;                                   // номер ячейки по x (от 0)
    int m = (index - n) / N;                             // номер ячейки по y (от 0)
    double y = y_min + m * dy;
    double x = x_min + n * dx;
    double dist = sqrt(x * x + y * y);

    double2 s_1, s_2, s_3, s_4, s_5, u_1, u_2, u_3, u_4, u_5;      // Переменные всех соседей и самой ячейки
    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // Вектора потоков
    double tmin = 1000;

    if (index < 0 || index > N * M - 1)
    {
        printf("Error index = %d \n", index);
    }

    double n1, n2, nn;

    s_1 = s[index];
    u_1 = u[index];
    double dist2 = kv(x + 0.35) / kv(0.65) + kv(y) / kv(0.55);
    if ( (dist2 < 1.0)  ) // Жёсткие граничные условия
    {
        // В этих ячейках значения параметров зафиксированы и не меняются с течением времени)
        s2[index] = s_1;
        u2[index] = u_1;
        return;
    }



    if (n == N - 1)
    {
        s_2 = { 1.0, 1.0};
        u_2 = { Velosity_inf, 0.0 };
    }
    else
    {
        s_2 = s[(m)*N + n + 1];
        u_2 = u[(m)*N + n + 1];
    }


    if ((n == 0))
    {
        s_4.x = s_1.x;
        s_4.y = s_1.y;
        u_4 = u_1;
        //u_4.x = -2.2;
        if (  (u_4.x > 0.5 * Velosity_inf)  )
        {
            u_4.x = 0.5 * Velosity_inf;              // Условие отсоса жидкости
        }
    }
    else
    {
        s_4 = s[(m)*N + n - 1];
        u_4 = u[(m)*N + n - 1];
    }

    if ((m == M - 1))
    {
        s_5 = s_1;
        u_5 = u_1;
    }
    else
    {
        s_5 = s[(m + 1) * N + (n)];
        u_5 = u[(m + 1) * N + (n)];
    }
    
    if ((m == 0))
    {
        s_3 = s_1;
        u_3.x = u_1.x;
        u_3.y = -u_1.y;
    }
    else
    {
        s_3 = s[(m - 1) * N + (n)];
        u_3 = u[(m - 1) * N + (n)];
    }


    if (method == 0)
    {
        tmin = my_min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dx));
        tmin = my_min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dy));
        tmin = my_min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dx));
        tmin = my_min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dy));
    }
    else if (method == 1)
    {
        if (x + dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dx));
        }
        if (y - dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dy));
        }
        if (x - dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dx));
        }
        if (y + dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dy));
        }

    }
    else if (method == 2)
    {
        if (x + dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dx));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dx));
        }
        if (y - dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dy));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dy));
        }
        if (x - dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dx));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dx));
        }
        if (y + dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dy));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dy));
        }

    }


    if (*T > tmin)
    {
        //atomicExch(T, tmin);
        *T = tmin;
    }

    double2 PS = { 0.0, 0.0 };
    double2 PU = { 0.0, 0.0 };

    /*if ((x > 400) &&( x < 405) && (y < 5))
    {
        printf("%lf, %lf, %lf, %lf, %lf\n", Ps12.x, Ps13.x, Ps14.x, Ps15.x, Ps12.x* dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx);
    }*/

    PS.x = (Ps12.x + Ps14.x) * dy + (Ps13.x + Ps15.x) * dx;
    PS.y = (Ps12.y + Ps14.y) * dy + (Ps13.y + Ps15.y) * dx;
    PU.x = (Pu12.x + Pu14.x) * dy + (Pu13.x + Pu15.x) * dx;
    PU.y = (Pu12.y + Pu14.y) * dy + (Pu13.y + Pu15.y) * dx;

    double dV = dx * dy;

    s2[index].x = s_1.x - (*T_do / dV) * PS.x - (*T_do/y) * s_1.x * u_1.y;

    //s2[index].x = s_1.x - (*T_do / dV) * PS.x;
    if (s2[index].x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf, T = %lf, ro = %lf \n", x, y, s2[index].x, *T_do, s_1.x);
        s2[index].x = s_1.x;
    }
    u2[index].x = (s_1.x * u_1.x - (*T_do / dV) * PU.x - (*T_do/y) * s_1.x * u_1.y * u_1.x) / s2[index].x;
    u2[index].y = (s_1.x * u_1.y - (*T_do / dV) * PU.y - (*T_do/y) * s_1.x * u_1.y * u_1.y) / s2[index].x;
    s2[index].y = (((s_1.y / (ggg - 1.0) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) - (*T_do / dV) * PS.y - //
        (*T_do/y) * u_1.y * (ggg * s_1.y / (ggg - 1.0) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) ) - //
        0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1.0);
    //u2[index].x = (s_1.x * u_1.x - (*T_do / dV) * PU.x ) / s2[index].x;
    //u2[index].y = (s_1.x * u_1.y - (*T_do / dV) * PU.y) / s2[index].x;
    

    //s2[index].y = ( ( (s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) - (*T_do / dV) * PS.y ) - //
    //    0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1);
    if (s2[index].y <= 0)
    {
        s2[index].y = 0.000001;
    }

}

__global__ void add_MK(double2* s, double2* u, double2* s2, double2* u2, double* nn1, double3* nn2, double* nn3, //
                            double* T, double* T_do, int method, int step)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;   // Глобальный индекс текущей ячейки (текущего потока)
    int n = index % N;                                   // номер ячейки по x (от 0)
    int m = (index - n) / N;                             // номер ячейки по y (от 0)
    double y = y_min + m * dy;
    double x = x_min + n * dx;
    double dist = sqrt(x * x + y * y);

    double2 s_1, s_2, s_3, s_4, s_5, u_1, u_2, u_3, u_4, u_5;      // Переменные всех соседей и самой ячейки
    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // Вектора потоков
    double tmin = 1000;

    if (index < 0 || index > N * M - 1)
    {
        printf("Error index = %d \n", index);
    }

    double n1, n2, nn;

    s_1 = s[index];
    u_1 = u[index];
    double dist2 = kv(x + 0.35) / kv(0.65) + kv(y) / kv(0.55);
    if ((dist2 < 1.0)) // Жёсткие граничные условия
    {
        // В этих ячейках значения параметров зафиксированы и не меняются с течением времени)
        s2[index] = s_1;
        u2[index] = u_1;
        return;
    }



    if (n == N - 1)
    {
        s_2 = { 1.0, 1.0 };
        u_2 = { Velosity_inf, 0.0 };
    }
    else
    {
        s_2 = s[(m)*N + n + 1];
        u_2 = u[(m)*N + n + 1];
    }


    if ((n == 0))
    {
        s_4.x = s_1.x;
        s_4.y = s_1.y;
        u_4 = u_1;
        //u_4.x = -2.2;
        if ((u_4.x > 0.5 * Velosity_inf))
        {
            u_4.x = 0.5 * Velosity_inf;              // Условие отсоса жидкости
        }
    }
    else
    {
        s_4 = s[(m)*N + n - 1];
        u_4 = u[(m)*N + n - 1];
    }

    if ((m == M - 1))
    {
        s_5 = s_1;
        u_5 = u_1;
    }
    else
    {
        s_5 = s[(m + 1) * N + (n)];
        u_5 = u[(m + 1) * N + (n)];
    }

    if ((m == 0))
    {
        s_3 = s_1;
        u_3.x = u_1.x;
        u_3.y = -u_1.y;
    }
    else
    {
        s_3 = s[(m - 1) * N + (n)];
        u_3 = u[(m - 1) * N + (n)];
    }


    if (method == 0)
    {
        tmin = my_min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dx));
        tmin = my_min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dy));
        tmin = my_min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dx));
        tmin = my_min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dy));
    }
    else if (method == 1)
    {
        if (x + dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dx));
        }
        if (y - dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dy));
        }
        if (x - dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dx));
        }
        if (y + dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dy));
        }

    }
    else if (method == 2)
    {
        if (x + dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dx));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s_1, u_1, s_2, u_2, 1, 0, Ps12, Pu12, dx));
        }
        if (y - dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dy));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s_1, u_1, s_3, u_3, 0, -1, Ps13, Pu13, dy));
        }
        if (x - dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dx));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s_1, u_1, s_4, u_4, -1, 0, Ps14, Pu14, dx));
        }
        if (y + dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dy));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s_1, u_1, s_5, u_5, 0, 1, Ps15, Pu15, dy));
        }

    }


    if (*T > tmin)
    {
        //atomicExch(T, tmin);
        *T = tmin;
    }

    double2 PS = { 0.0, 0.0 };
    double2 PU = { 0.0, 0.0 };

    /*if ((x > 400) &&( x < 405) && (y < 5))
    {
        printf("%lf, %lf, %lf, %lf, %lf\n", Ps12.x, Ps13.x, Ps14.x, Ps15.x, Ps12.x* dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx);
    }*/

    PS.x = (Ps12.x + Ps14.x) * dy + (Ps13.x + Ps15.x) * dx;
    PS.y = (Ps12.y + Ps14.y) * dy + (Ps13.y + Ps15.y) * dx;
    PU.x = (Pu12.x + Pu14.x) * dy + (Pu13.x + Pu15.x) * dx;
    PU.y = (Pu12.y + Pu14.y) * dy + (Pu13.y + Pu15.y) * dx;

    double dV = dx * dy;

    s2[index].x = s_1.x - (*T_do / dV) * PS.x - (*T_do / y) * s_1.x * u_1.y;

    //s2[index].x = s_1.x - (*T_do / dV) * PS.x;
    if (s2[index].x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf, T = %lf, ro = %lf \n", x, y, s2[index].x, *T_do, s_1.x);
        s2[index].x = s_1.x;
    }
    u2[index].x = (s_1.x * u_1.x - (*T_do / dV) * PU.x - (*T_do / y) * s_1.x * u_1.y * u_1.x + *T_do * (n_H/Kn) * nn2[index].x) / s2[index].x;
    u2[index].y = (s_1.x * u_1.y - (*T_do / dV) * PU.y - (*T_do / y) * s_1.x * u_1.y * u_1.y + *T_do * (n_H / Kn) * nn2[index].y) / s2[index].x;
    s2[index].y = (((s_1.y / (ggg - 1.0) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) - (*T_do / dV) * PS.y - //
        (*T_do / y) * u_1.y * (ggg * s_1.y / (ggg - 1.0) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) + *T_do * (n_H / Kn) * nn3[index]) - //
        0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1.0);
    //u2[index].x = (s_1.x * u_1.x - (*T_do / dV) * PU.x ) / s2[index].x;
    //u2[index].y = (s_1.x * u_1.y - (*T_do / dV) * PU.y) / s2[index].x;


    //s2[index].y = ( ( (s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) - (*T_do / dV) * PS.y ) - //
    //    0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1);
    if (s2[index].y <= 0)
    {
        s2[index].y = 0.000001;
    }

}

__global__ void Kernel_TVD(double2* s, double2* u, double2* s2, double2* u2, double* T, double* T_do, int method)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;   // Глобальный индекс текущей ячейки (текущего потока)
    int n = index % N;                                   // номер ячейки по x (от 0)
    int m = (index - n) / N;                             // номер ячейки по y (от 0)
    double y = y_min + m * (y_max) / (M);
    double x = x_min + n * (x_max - x_min) / (N - 1);
    double dist = __dsqrt_rn(x * x + y * y);

    double2 s_1, s_2, s_3, s_4, s_5, u_1, u_2, u_3, u_4, u_5;      // Переменные всех соседей и самой ячейки
    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // Вектора потоков
    double tmin = 1000;

    s_1 = s[index];
    u_1 = u[index];
    //if ((n == N - 1) || (m == M - 1) || (dist < 110)) // Жёсткие граничные условия
   if ((n == N - 1) || (dist < 110)) // Жёсткие граничные условия
    {
        // В этих ячейках значения параметров зафиксированы и не меняются с течением времени)
        s2[index] = s_1;
        u2[index] = u_1;
        return;
    }
    s_2 = s[(m)*N + n + 1];
    u_2 = u[(m)*N + n + 1];

    if (m == M - 1)
    {
        s_5 = s_1;
        u_5 = u_1;
    }
    else
    {
        s_5 = s[(m + 1) * N + n];
        u_5 = u[(m + 1) * N + n];
    }
    if ((n == 0))
    {
        s_4.x = s_1.x;
        s_4.y = 1.0 / (ggg * M_inf * M_inf);      // Неотражающее давление
        u_4 = u_1;
        //if (u_1.x > 0.0)
        //{
        //    u_4.x = -0.3;              // Условие отсоса жидкости
        //}
        double Max = sqrt((u_1.x * u_1.x + u_1.y * u_1.y) / (ggg * s_1.y / s_1.x));
        if ( (u_1.x > -5.0)&&(Max < 1) )
        {
            u_4.x = -5.0;              // Условие сверх- отсоса жидкости
        }
    }
    else
    {
        s_4 = s[(m)*N + n - 1];
        u_4 = u[(m)*N + n - 1];
    }

    if ((m == 0))
    {
        s_3 = s_1;
        u_3.x = u_1.x;
        u_3.y = -u_1.y;
    }
    else
    {
        s_3 = s[(m - 1) * N + (n)];
        u_3 = u[(m - 1) * N + (n)];
    }

    double2 s12 = { 0.0 ,0.0 };
    double2 s13 = { 0.0 ,0.0 };
    double2 s14 = { 0.0 ,0.0 };
    double2 s15 = { 0.0 ,0.0 };
    double2 u12 = { 0.0 ,0.0 };
    double2 u13 = { 0.0 ,0.0 };
    double2 u14 = { 0.0 ,0.0 };
    double2 u15 = { 0.0 ,0.0 };
    double2 s21 = { 0.0 ,0.0 };
    double2 s31 = { 0.0 ,0.0 };
    double2 s41 = { 0.0 ,0.0 };
    double2 s51 = { 0.0 ,0.0 };
    double2 u21 = { 0.0 ,0.0 };
    double2 u31 = { 0.0 ,0.0 };
    double2 u41 = { 0.0 ,0.0 };
    double2 u51 = { 0.0 ,0.0 };
    double A = 0, B = 0;
    // Заполняем значениями соседей-соседей
    if (n > N - 3)
    {
        s21 = s_2;
        u21 = u_2;
    }
    else
    {
        s21 = s[(m) * N + (n + 2)];
        u21 = u[(m)*N + (n + 2)];
    }
    if (n == 0)
    {
        s41 = s_4;
        u41 = u_4;
    }
    else if (n == 1)
    {
        s41 = s_4;
        u41 = u_4;
        //if (u41.x > 0.0)
        //{
        //    u41.x = -0.3;             // Условие отсоса жидкости
        //}
        double Max = sqrt((u_4.x * u_4.x + u_4.y * u_4.y) / (ggg * s_4.y / s_4.x));
        if ((u41.x > -5.0) && (Max < 1))
        {
            u41.x = -5.0;              // Условие отсоса жидкости
        }
    }
    else
    {
        s41 = s[(m)*N + (n - 2)];
        u41 = u[(m)*N + (n - 2)];
    }
    if (m > M - 3)
    {
        s51 = s_5;
        u51 = u_5;
    }
    else
    {
        s51 = s[(m + 2)*N + (n)];
        u51 = u[(m + 2)*N + (n)];
    }
    if (m == 1)
    {
        s31 = s_3;
        u31.x = u_3.x;
        u31.y = -u_3.y;
    }
    else if (m == 0) 
    {
        s31 = s_5;
        u31.x = u_5.x;
        u31.y = -u_5.y;
    }
    else
    {
        s31 = s[(m - 2)*N + (n)];
        u31 = u[(m - 2)*N + (n)];
    }

    linear2(x - dx, s_4.x,      x, s_1.x,   x + dx, s_2.x,  x - dx/2.0, x + dx/2.0,     A, B);
    if (B <= 0)
    {
        s12.x = s_1.x;
    }
    else
    {
        s12.x = B;
    }
    if (A <= 0)
    {
        s14.x = s_1.x;
    }
    else
    {
        s14.x = A;
    }
    linear2(x - dx, s_4.y,       x, s_1.y,    x + dx, s_2.y,  x - dx / 2.0, x + dx / 2.0,    A, B);
    if ((B <= 0) || (grad_p == false) )
    {
        s12.y = s_1.y;
    }
    else
    {
        s12.y = B;
    }
    if ( (A <= 0) || (grad_p == false) )
    {
        s14.y = s_1.y;
    }
    else
    {
        s14.y = A;
    }
    linear2(x - dx, u_4.x,      x, u_1.x,   x + dx, u_2.x,       x - dx / 2.0, x + dx / 2.0,     A, B);
    u12.x = B;
    u14.x = A;
    linear2(x - dx, u_4.y,       x, u_1.y,   x + dx, u_2.y,     x - dx / 2.0, x + dx / 2.0,      A, B);
    u12.y = B;
    u14.y = A;

    linear2(y - dy, s_3.x,      y, s_1.x,       y + dy, s_5.x,      y - dy / 2.0, y + dy / 2.0,     A, B);
    if (B <= 0)
    {
        s15.x = s_1.x;
    }
    else
    {
        s15.x = B;
    }
    if (A <= 0)
    {
        s13.x = s_1.x;
    }
    else
    {
        s13.x = A;
    }
    linear2(y - dy, s_3.y,      y, s_1.y,       y + dy, s_5.y,      y - dy / 2.0, y + dy / 2.0,         A, B);
    if ((B <= 0) || (grad_p == false) )
    {
        s15.y = s_1.y;
    }
    else
    {
        s15.y = B;
    }
    if ( (A <= 0) || (grad_p == false) )
    {
        s13.y = s_1.y;
    }
    else
    {
        s13.y = A;
    }
    linear2(y - dy, u_3.x,      y, u_1.x,       y + dy, u_5.x,       y - dy / 2.0, y + dy / 2.0,        A, B);
    u15.x = B;
    u13.x = A;
    linear2(y - dy, u_3.y,       y, u_1.y,      y + dy, u_5.y,      y - dy / 2.0, y + dy / 2.0,         A, B);
    u15.y = B;
    u13.y = A;

    s21.x = linear(x, s_1.x,     x + dx, s_2.x,      x + 2.0 * dx, s21.x,       x + dx / 2.0);
    if (s21.x <= 0) s21.x = s_2.x;
    s21.y = linear(x, s_1.y,      x + dx, s_2.y,    x + 2.0 * dx, s21.y,    x + dx / 2.0);
    if ( (s21.y <= 0) || (grad_p == false) ) s21.y = s_2.y;
    u21.x = linear(x, u_1.x,    x + dx, u_2.x,      x + 2.0 * dx, u21.x,    x + dx / 2.0);
    u21.y = linear(x, u_1.y,    x + dx, u_2.y,      x + 2.0 * dx, u21.y,    x + dx / 2.0);

    s41.x = linear(x, s_1.x,    x - dx, s_4.x,      x - 2.0 * dx, s41.x,        x - dx / 2.0);
    if (s41.x <= 0) s41.x = s_4.x;
    s41.y = linear(x, s_1.y,    x - dx, s_4.y,      x - 2.0 * dx, s41.y,         x - dx / 2.0);
    if ((s41.y <= 0) || (grad_p == false) ) s41.y = s_4.y;
    u41.x = linear(x, u_1.x,    x - dx, u_4.x,      x - 2.0 * dx, u41.x,        x - dx / 2.0);
    u41.y = linear(x, u_1.y,    x - dx, u_4.y,      x - 2.0 * dx, u41.y,         x - dx / 2.0);

    s31.x = linear(y, s_1.x,        y - dy, s_3.x,      y - 2.0 * dy, s31.x,        y - dy / 2.0);
    if (s31.x <= 0) s31.x = s_3.x;
    s31.y = linear(y, s_1.y,        y - dy, s_3.y,      y - 2.0 * dy, s31.y,        y - dy / 2.0);
    if ( (s31.y <= 0) || (grad_p == false) ) s31.y = s_3.y;
    u31.x = linear(y, u_1.x,        y - dy, u_3.x,      y - 2.0 * dy, u31.x,        y - dy / 2.0);
    u31.y = linear(y, u_1.y,        y - dy, u_3.y,      y - 2.0 * dy, u31.y,        y - dy / 2.0);

    s51.x = linear(y, s_1.x,        y + dy, s_5.x,      y + 2.0 * dy, s51.x,        y + dy / 2.0);
    if (s51.x <= 0) s51.x = s_5.x;
    s51.y = linear(y, s_1.y,        y + dy, s_5.y,      y + 2.0 * dy, s51.y,        y + dy / 2.0);
    if ( (s51.y <= 0)||(grad_p == false) ) s51.y = s_5.y;
    u51.x = linear(y, u_1.x,        y + dy, u_5.x,      y + 2.0 * dy, u51.x,        y + dy / 2.0);
    u51.y = linear(y, u_1.y,        y + dy, u_5.y,      y + 2.0 * dy, u51.y,        y + dy / 2.0);


    if (method == 0)
    {
        tmin = min(tmin, HLL(s12, u12, s21, u21, 1, 0, Ps12, Pu12, dy));
        tmin = min(tmin, HLL(s13, u13, s31, u31, 0, -1, Ps13, Pu13, dx));
        tmin = min(tmin, HLL(s14, u14, s41, u41, -1, 0, Ps14, Pu14, dy));
        tmin = min(tmin, HLL(s15, u15, s51, u51, 0, 1, Ps15, Pu15, dx));
    }
    else if (method == 1)
    {
        if (x + dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s12, u12, s21, u21, 1, 0, Ps12, Pu12, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s12, u12, s21, u21, 1, 0, Ps12, Pu12, dy));
        }
        if (y - dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s13, u13, s31, u31, 0, -1, Ps13, Pu13, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s13, u13, s31, u31, 0, -1, Ps13, Pu13, dx));
        }
        if (x - dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s14, u14, s41, u41, -1, 0, Ps14, Pu14, dy));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s14, u14, s41, u41, -1, 0, Ps14, Pu14, dy));
        }
        if (y + dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s15, u15, s51, u51, 0, 1, Ps15, Pu15, dx));
        }
        else
        {
            tmin = min(tmin, HLLC_Aleksashov(s15, u15, s51, u51, 0, 1, Ps15, Pu15, dx));
        }
    }
    else if (method == 2)
    {
        if (x + dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s12, u12, s21, u21, 1, 0, Ps12, Pu12, dy));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s12, u12, s21, u21, 1, 0, Ps12, Pu12, dy));
        }
        if (y - dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s13, u13, s31, u31, 0, -1, Ps13, Pu13, dx));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s13, u13, s31, u31, 0, -1, Ps13, Pu13, dx));
        }
        if (x - dx / 2.0 < hx)
        {
            tmin = min(tmin, HLL(s14, u14, s41, u41, -1, 0, Ps14, Pu14, dy));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s14, u14, s41, u41, -1, 0, Ps14, Pu14, dy));
        }
        if (y + dy / 2.0 < hy)
        {
            tmin = min(tmin, HLL(s15, u15, s51, u51, 0, 1, Ps15, Pu15, dx));
        }
        else
        {
            tmin = min(tmin, Godunov_Solver_Alexashov(s15, u15, s51, u51, 0, 1, Ps15, Pu15, dx));
        }
    }
    else
    {
        printf("Error in method 2375\n");
    }


    if (*T > tmin)
    {
       // __threadfence();
        *T = tmin;
    }

    double2 PS = { 0,0 };
    double2 PU = { 0,0 };

    /*if ((x > 400) &&( x < 405) && (y < 5))
    {
        printf("%lf, %lf, %lf, %lf, %lf\n", Ps12.x, Ps13.x, Ps14.x, Ps15.x, Ps12.x* dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx);
    }*/

    PS.x = Ps12.x * dy + Ps13.x * dx + Ps14.x * dy + Ps15.x * dx;
    PS.y = Ps12.y * dy + Ps13.y * dx + Ps14.y * dy + Ps15.y * dx;
    PU.x = Pu12.x * dy + Pu13.x * dx + Pu14.x * dy + Pu15.x * dx;
    PU.y = Pu12.y * dy + Pu13.y * dx + Pu14.y * dy + Pu15.y * dx;

    double dV = dx * dy;

    s2[index].x = s[index].x - (*T_do / dV) * PS.x - *T_do * s[index].x * u[index].y / y;
    if (s2[index].x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf\n", x, y, s2[index].x);
        s2[index].x = 0.0001;
    }
    u2[index].x = (s[index].x * u[index].x - (*T_do / dV) * PU.x - *T_do * s[index].x * u[index].y * u[index].x / y) / s2[index].x;
    u2[index].y = (s[index].x * u[index].y - (*T_do / dV) * PU.y - *T_do * s[index].x * u[index].y * u[index].y / y) / s2[index].x;
    s2[index].y = (((s[index].y / (ggg - 1) + s[index].x * (u[index].x * u[index].x + u[index].y * u[index].y) * 0.5) - (*T_do / dV) * PS.y - //
        *T_do * u[index].y * (ggg * s[index].y / (ggg - 1) + s[index].x * (u[index].x * u[index].x + u[index].y * u[index].y) * 0.5) / y) - //
        0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y)) * (ggg - 1);
    if (s2[index].y <= 0)
    {
        s2[index].y = 0.000001;
    }
}

__global__ void test(void)
{
    double2 s_1 = { 1, 0.0666666 };
    double2 u_1 = { -1, 0 };
    double2 s_2 = { 1, 0.0666666 };
    double2 u_2 = { -1, 0 };
    double2 P1, P2;
    Godunov_Solver_Alexashov(s_1, u_1, s_2, u_2, 1, 0, P1, P2, dy);
    printf("%lf\n", P1.x);
    Godunov_Solver_Alexashov(s_1, u_1, s_2, u_2, -1, 0, P1, P2, dy);
    printf("%lf\n", P1.x);
    
}

void print_file_mini(double2* host_s_p, double2* host_u_p, double* nn1, double3* nn2, double* nn3, string name)
{
    ofstream fout;
    fout.open(name);
    int nn = (int)((N + Nmin - 1) / Nmin);
    int mm = (int)((M + Nmin - 1) / Nmin);
    fout << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\", \"Ro\", \"P\", \"Vx\", \"Vy\", \"Max\", \"T\",\"Ro_H\",\"Vx_H\",\"Vr_H\",\"Vphi_H\",\"T_H\", ZONE T = \"HP\", N = " << nn * mm //
        << " , E = " << (nn - 1) * (mm - 1) << ", F = FEPOINT, ET = quadrilateral" << endl;
    //double ss = (sqv_1 * pi * kv(y_max) + sqv_2 * 2.0 * pi * y_max *  (x_max - x_min + dx));
    for (int k = 0; k < K; k++)
    {
        int n = k % N;                                   // номер ячейки по x (от 0)
        int m = (k - n) / N;                             // номер ячейки по y (от 0)
        if ((n % Nmin != 0) || (m % Nmin != 0))
        {
            continue;
        }

        double y = y_min + m * (y_max) / (M);
        double x = x_min + n * (x_max - x_min) / (N - 1);
        double no = (1.0 * AllNumber * (pi * kv(y + dy/2.0) * dx - pi * kv(y - dy / 2.0) * dx));
        double Max = 0.0, Temp = 0.0;
        double nn = nn1[k];// sum_s* nn1[k] / no;
        //double nn = sqv_1 * nn1[k] / no;
        double  n3 = 0.0;
        double v1 = 0.0, v2 = 0.0, v3 = 0.0;
        if (nn1[k] > 0.000001)
        {
            v1 = nn2[k].x / nn1[k];
            v2 = nn2[k].y / nn1[k];
            v3 = nn2[k].z / nn1[k];
            n3 = (2.0 / 3.0) * (nn3[k] / nn1[k] - kvv(v1, v2, v3));
        }

        
        if (host_s_p[k].x > 0.0)
        {
            Max = sqrt((host_u_p[k].x * host_u_p[k].x + host_u_p[k].y * host_u_p[k].y) / (ggg * host_s_p[k].y / host_s_p[k].x));
            Temp = host_s_p[k].y / host_s_p[k].x;
        }
        fout << x << " " << y << " " << host_s_p[k].x << " " << host_s_p[k].y <<//
            " " << host_u_p[k].x << " " << host_u_p[k].y << " " << //
            Max << " " << Temp << " " << nn << " " <<  v1 << " " << v2 << " " << v3 << " " <<  //
            n3 << endl;
    }

    for (int k = 0; k < nn * mm; k = k + 1)
    {
        int n = k % nn;                                   // номер ячейки по x (от 0)
        int m = (k - n) / nn;
        if ((m < mm - 1) && (n < nn - 1))
        {
            fout << m * nn + n + 1 << " " << m * nn + n + 2 << " " << (m + 1) * nn + n + 2 << " " << (m + 1) * nn + n + 1 << endl;
        }
    }
    fout.close();
}

void Save_file(double2* host_s_p, double2* host_u_p, double* nn1, double3* nn2, double* nn3, string name)
{
    ofstream fout;
    fout.open(name);

    for (int k = 0; k < K; k++)
    {
        int n = k % N;                                   // номер ячейки по x (от 0)
        int m = (k - n) / N;                             // номер ячейки по y (от 0)
        double y = y_min + m * (y_max) / (M);
        double x = x_min + n * (x_max - x_min) / (N - 1);
        fout << x << " " << y << " " << host_s_p[k].x << " " << host_s_p[k].y <<//
            " " << host_u_p[k].x << " " << host_u_p[k].y << " " << nn1[k] << " " << //
            nn2[k].x << " " << nn2[k].y << " " << nn2[k].z << " " << nn3[k] << endl;
    }

    fout.close();
}

int main(void)
{
    double2* host_s, * host_u;
    double2* s, * u;
    double2* host_s2, * host_u2;
    int* host_i;
    double2* s2, * u2;
    int* dev_i;
    double* host_T, * host_T_do, * host_TT;
    double* T, * T_do, * TT;
    int size = K * sizeof(double2);
    double* nn1, * nn3;
    double3* nn2;
    double* dev_nn1, * dev_nn3;
    double3* dev_nn2;

    cudaEvent_t start, stop;
    cudaError_t cudaStatus;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //выделяем память для device копий для host_s, host_u
    cudaMalloc((void**)&s, size);
    cudaMalloc((void**)&u, size);
    cudaMalloc((void**)&s2, size);
    cudaMalloc((void**)&u2, size);
    cudaMalloc((void**)&dev_nn1, K * sizeof(double));
    cudaMalloc((void**)&dev_nn2, K * sizeof(double3));
    cudaMalloc((void**)&dev_nn3, K * sizeof(double));
    cudaMalloc((void**)&T, sizeof(double));
    cudaMalloc((void**)&T_do, sizeof(double));
    cudaMalloc((void**)&TT, sizeof(double));
    cudaMalloc((void**)&dev_i, sizeof(int));

    ifstream fin2;
    fin2.open("rnd_Dima.dat");
    vector<Sensor*> Sensors;
    double d, a, b, c;
    for (int i = 0; i < 270; i++)
    {
        fin2 >> d >> a >> b >> c;
        auto s = new Sensor(a, b, c);
        Sensors.push_back(s);
    }

    host_s = (double2*)malloc(size);
    host_u = (double2*)malloc(size);
    host_s2 = (double2*)malloc(size);
    host_u2 = (double2*)malloc(size);
    nn1 = (double*)malloc(K * sizeof(double));
    nn2 = (double3*)malloc(K * sizeof(double3));
    nn3 = (double*)malloc(K * sizeof(double));
    host_T = (double*)malloc(sizeof(double));
    host_T_do = (double*)malloc(sizeof(double));
    host_TT = (double*)malloc(sizeof(double));
    host_i = (int*)malloc(sizeof(int));

    *host_T = 10000000;
    *host_T_do = 0.000000001;
    *host_TT = 0.0;
    *host_i = 0;
    //cout << "dy" << dy << endl;
    //for (int k = 0; k < M; k++)  // Заполняем начальные условия
    //{
    //    double y = y_min + k * (y_max) / (M);
    //    cout << y << endl;
    //}

    double k_ = 0.1;
    double l_ = 1.0;
    double chi = 36.1059; // 36.1059
    std::cout << dx << " " << dy << endl;
    for (int k = 0; k < K; k++)  // Заполняем начальные условия
    {
        nn1[k] = 0.0;
        nn2[k] = { 0.0, 0.0, 0.0 };
        nn3[k] = 0.0;
        int n = k % N;                                   // номер ячейки по x (от 0)
        int m = (k - n) / N;                             // номер ячейки по y (от 0)
        double y = y_min + m * dy;
        double x = x_min + n * dx;
        double dist = sqrt(x * x + y * y);
        double r_0 = 0.00256418;
        double ro = (1.0) / (chi * chi * r_0 * r_0);
        double P_E = ro * chi  * chi / (ggg * 0.2 * 0.2);
        double dist2 = kv(x + 0.35) / kv(0.65) + kv(y) / kv(0.55);
        if (dist2 <= 1.0)
        {
            host_s[k] = { ro * r_0 * r_0/ (dist * dist) , P_E * pow(r_0 / dist, 2.0 * ggg) };
            host_u[k] = { chi * x/dist , chi * y / dist };
            host_s2[k] = { ro * r_0 * r_0 / (dist * dist) , P_E * pow(r_0 / dist, 2.0 * ggg) };
            host_u2[k] = { chi * x / dist , chi * y / dist };
        }
        else 
        {
            host_s[k] = { 1.0, 1.0 };
            host_u[k] = { Velosity_inf, 0.0 };
            host_s2[k] = { 1.0, 1.0 };
            host_u2[k] = { Velosity_inf, 0.0 };
        }
    }


    //
    double c1, c2, a1, a2, a3, a4, a5, a6, a7, a8, a9;
    ifstream fin;
    fin.open("chi_36_start_all.txt"); 

    for (int k = 0; k < K; k++)
    {
        fin >> c1 >> c2 >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7 >> a8 >> a9;
        host_s[k].x = a1;
        host_s[k].y = a2;
        host_u[k].x = a3;
        host_u[k].y = a4;
        host_s2[k].x = a1;
        host_s2[k].y = a2;
        host_u2[k].x = a3;
        host_u2[k].y = a4;
        nn1[k] = a5;
        nn2[k].x = a6;
        nn2[k].y = a7;
        nn2[k].z = a8;
        nn3[k] = a9;
    }
    fin.close();

    
    for (int k = 0; k < K; k++)  // Заполняем начальные условия
    {
        /*nn1[k] = 0.0;
        nn2[k] = { 0.0, 0.0, 0.0 };
        nn3[k] = 0.0;*/
        int n = k % N;                                   // номер ячейки по x (от 0)
        int m = (k - n) / N;                             // номер ячейки по y (от 0)
        double y = y_min + m * dy;
        double x = x_min + n * dx;
        double dist = sqrt(x * x + y * y);
        double r_0 = 0.00256418;
        double ro = (1.0) / (chi * chi * r_0 * r_0);
        double P_E = ro * chi * chi / (ggg * 0.2 * 0.2);
        double dist2 = kv(x + 0.35) / kv(0.65) + kv(y) / kv(0.55);
        if (dist2 <= 1.0)
        {
            host_s[k] = { ro * r_0 * r_0 / (dist * dist) , P_E * pow(r_0 / dist, 2.0 * ggg) };
            host_u[k] = { chi * x / dist , chi * y / dist };
            host_s2[k] = { ro * r_0 * r_0 / (dist * dist) , P_E * pow(r_0 / dist, 2.0 * ggg) };
            host_u2[k] = { chi * x / dist , chi * y / dist };
        }
    }
    
   
    bool device = true;
    //копируем ввод на device
    if (device)
    {
        cudaMemcpy(s, host_s, size, cudaMemcpyHostToDevice);
        cudaMemcpy(u, host_u, size, cudaMemcpyHostToDevice);
        cudaMemcpy(s2, host_s2, size, cudaMemcpyHostToDevice);
        cudaMemcpy(u2, host_u2, size, cudaMemcpyHostToDevice);
        cudaMemcpy(T, host_T, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(T_do, host_T_do, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(TT, host_TT, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_i, host_i, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_nn1, nn1, K * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_nn2, nn2, K * sizeof(double3), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_nn2, nn2, K * sizeof(double), cudaMemcpyHostToDevice);
    }

    ofstream set;
    set.open("Setka.txt");
    set << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\"  ZONE T= \"HP\", N="<<  2 * (N + M) << " , E= "<< (N + M) <<", F=FEPOINT, ET=LINESEG" << endl;
    for (int i = 0; i < N; i++)
    {
        double x0 = x_min + i * dx;
        set << x0 - dx/2.0 << " " << 0.0 << endl;
        set << x0 - dx/2.0 << " " << y_max << endl;
    }
    for (int i = 0; i < M; i++)
    {
        double y0 = y_min + i * dy;
        set << x_min - dx/2.0 << " " << y0 - dy/2.0 << endl;
        set << x_max + dx / 2.0 << " " << y0 - dy / 2.0 << endl;
    }
    for (int i = 0; i < N + M; i++)
    {
        set << 2 * i + 1 << " " << 2 * i + 2 << endl;
    }

    std::cout << "START" << endl;


    for (int i = 0; i < 0; i = i + 2)  // Сколько шагов по времени делаем?
    {
        if (i == 0)
        {
            cout << "HLL" << endl;
        }
        add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, s2, u2, T, T_do, 0, i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, s, u, T, T_do, 0, i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));   exit(-1); }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); exit(-1); }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        if (i % 50000 == 0 && i > 1)
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("8000 step - Time:  %.2f sec\n", elapsedTime / 1000.0);
            cudaEventRecord(start, 0);
            cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
            string name = "HLL" + to_string(i) + ".txt";
            print_file_mini(host_s, host_u, nn1, nn2, nn3, name);
        }
    }
    for (int i = 0; i < 0; i = i + 2)  // Сколько шагов по времени делаем?
    {
        if (i == 0)
        {
            cout << "HLLC" << endl;
        }
        add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, s2, u2, T, T_do, 1, i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, s, u, T, T_do, 1, i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));   exit(-1); }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); exit(-1); }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        if (i % 50000 == 0 || i == 15000 || i == 20000)
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("8000 step - Time:  %.2f sec\n", elapsedTime / 1000.0);
            cudaEventRecord(start, 0);
            cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
            string name = "16_09_" + to_string(i) + ".txt";
            print_file_mini(host_s, host_u, nn1, nn2, nn3, name);
        }
    }
    for (int i = 0; i < 0; i = i + 2)  // Сколько шагов по времени делаем?
    {
        if (i == 0)
        {
            cout << "Godunov" << endl;
        }
        add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, s2, u2, T, T_do, 2, i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        add2 << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, s, u, T, T_do, 2, i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));   exit(-1); }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); exit(-1); }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        if (i % 5000 == 0 && i > 1)
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("8000 step - Time:  %.2f sec\n", elapsedTime / 1000.0);
            cudaEventRecord(start, 0);
            cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
            string name = "Godunov_" + to_string(i) + ".txt";
            print_file_mini(host_s, host_u, nn1, nn2, nn3, name);
        }
    }


    for (auto& i: Sensors)
    {
        i->Restart();
    }

    /*M_K(Sensors, host_s, host_u, nn1, nn2, nn3);
    string name = "chi_36_start_all.txt";
    Save_file(host_s, host_u, nn1, nn2, nn3, name);*/

    cudaMemcpy(dev_nn1, nn1, K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_nn2, nn2, K * sizeof(double3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_nn2, nn2, K * sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < 400000; i = i + 2)  // Сколько шагов по времени делаем?
    {
        if (i == 0)
        {
            cout << "HLLC_MK" << endl;
        }
        add_MK << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, s2, u2, dev_nn1, dev_nn2, dev_nn3, T, T_do, 1, i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        add_MK << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, s, u, dev_nn1, dev_nn2, dev_nn3, T, T_do, 1, i);;
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));   exit(-1); }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); exit(-1); }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        if (i % 80000 == 0 || i == 15000 || i == 20000)
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("step - Time:  %.2f sec\n", elapsedTime / 1000.0);
            cudaEventRecord(start, 0);
            cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
            string name = "13_03_" + to_string(i) + ".txt";
            print_file_mini(host_s, host_u, nn1, nn2, nn3, name);
        }
    }

    cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
    print_file_mini(host_s, host_u, nn1, nn2, nn3, "End_1.txt");
    Save_file(host_s, host_u, nn1, nn2, nn3, "1_plasma_iterate.txt");

 

    if (false)
    {
            // Старт одномерной программы
        /*Sensor* sens = Sensors[0];
        double Vx1 = 0.0, Vx2 = 0.0;
        double mu1, mu2;
        double ksi3, ksi4, ksi5, ksi6, w;
        double z = 0;
        double p1 = fabs(Velosity_inf) * sqrtpi / (1.0 + fabs(Velosity_inf) * sqrtpi);

        double n1 = 0.0, n2 = 0.0, VV1 = 0.0, T1 = 0.0, n3 = 0.0, n4, VV2, VV3, VV4, T2, T3, T4;*/

        /*ofstream xout;
        xout.open("Vx_do_2.txt");

        ofstream xout2;
        xout2.open("Vx_do_1.txt");

        ofstream xout3;
        xout3.open("Vx_1.txt");

        ofstream xout4;
        xout4.open("Vx_2.txt");*/

        //int kk1 = 800000;
        //int kk2 = 500000;
        //double sq1 = 0.599821;
        //double sq2 = 0.0998206;
        //mu1 = sq1/(sq1 + sq2) * (kk1 + kk2)/kk1;
        //mu2 = sq2 / (sq1 + sq2) * (kk1 + kk2) / kk2;
        //for (int ii = 3; ii < 4; ii++)
        //{
        //    sens = Sensors[ii];
        //    n1 = 0.0;
        //    n2 = 0.0;
        //    n3 = 0.0;
        //    n4 = 0.0;
        //    VV1 = 0.0;
        //    T1 = 0.0;
        //    VV2 = 0.0;
        //    T2 = 0.0;
        //    VV3 = 0.0;
        //    T3 = 0.0;
        //    VV4 = 0.0;
        //    T4 = 0.0;
        //    for (int i = 1; i <= kk1; i++) // Запуск с правой границы
        //    {
        //        z = 0;
        //        Vx1 = 0.0;
        //        Vx2 = 0.0;
        //        p1 = fabs(Velosity_inf) * sqrtpi / (1.0 + fabs(Velosity_inf) * sqrtpi);
        //        do
        //        {
        //            ksi3 = sens->MakeRandom();
        //            ksi4 = sens->MakeRandom();
        //            ksi5 = sens->MakeRandom();
        //            ksi6 = sens->MakeRandom();

        //            if (p1 > ksi3)
        //            {
        //                z = cos(pi * ksi5) * sqrt(-log(ksi4));
        //            }
        //            else
        //            {
        //                if (ksi4 <= 0.5)
        //                {
        //                    z = -sqrt(-log(2.0 * ksi4));
        //                }
        //                else
        //                {
        //                    z = sqrt(-log(2.0 * (1.0 - ksi4)));
        //                }
        //            }
        //        } while (fabs(z + Velosity_inf) / (fabs(Velosity_inf) + fabs(z)) <= ksi6 || z > -Velosity_inf);

        //        Vx1 = z + Velosity_inf;
        //        n1 += mu1 / fabs(Vx1);
        //        VV1 += Vx1 * mu1 / fabs(Vx1);
        //        T1 +=  kv(Vx1) * mu1 / fabs(Vx1);
        //        xout2 << Vx1 << endl;



        //        w = Velosity_inf - Vx1;
        //        p1 = fabs(w) * sqrtpi / (1.0 + fabs(w) * sqrtpi);
        //        do
        //        {
        //            ksi3 = sens->MakeRandom();
        //            ksi4 = sens->MakeRandom();
        //            ksi5 = sens->MakeRandom();
        //            ksi6 = sens->MakeRandom();

        //            if (p1 > ksi3)
        //            {
        //                z = cos(pi * ksi5) * sqrt(-log(ksi4));
        //            }
        //            else
        //            {
        //                if (ksi4 <= 0.5)
        //                {
        //                    z = -sqrt(-log(2.0 * ksi4));
        //                }
        //                else
        //                {
        //                    z = sqrt(-log(2.0 * (1.0 - ksi4)));
        //                }
        //            }
        //        } while (fabs(z + w) / (fabs(w) + fabs(z)) <= ksi6);
        //        Vx2 = z + Velosity_inf;

        //        if (Vx2 > 0)
        //        {
        //            n2 += mu1 / fabs(Vx2);
        //            VV2 += Vx2 * mu1 / fabs(Vx2);
        //            T2 += kv(Vx2) * mu1 / fabs(Vx2);
        //            xout3 << Vx2 << endl;
        //        }
        //        else
        //        {
        //            n4 += mu1 / fabs(Vx2);
        //            VV4 += Vx2 * mu1 / fabs(Vx2);
        //            T4 += kv(Vx2) * mu1 / fabs(Vx2);
        //            xout4 << Vx2 << endl;
        //        }

        //    }
        //    for (int i = 1; i <= kk2; i++)  // Для левой границы
        //    {
        //        z = 0;
        //        Vx1 = 0.0;
        //        Vx2 = 0.0;
        //        p1 = fabs(Velosity_inf) * sqrtpi / (1.0 + fabs(Velosity_inf) * sqrtpi);
        //        do
        //        {
        //            ksi3 = sens->MakeRandom();
        //            ksi4 = sens->MakeRandom();
        //            ksi5 = sens->MakeRandom();
        //            ksi6 = sens->MakeRandom();

        //            if (p1 > ksi3)
        //            {
        //                z = cos(pi * ksi5) * sqrt(-log(ksi4));
        //            }
        //            else
        //            {
        //                z = sqrt(-log(1.0 - ksi4));
        //            }
        //        } while (fabs(z + Velosity_inf) / (fabs(Velosity_inf) + fabs(z)) <= ksi6 || z < -Velosity_inf);
        //        Vx1 = z + Velosity_inf;
        //        n3 += mu2 / fabs(Vx1);
        //        VV3 += Vx1 * mu2 / fabs(Vx1);
        //        T3 += mu2 *  kv(Vx1) / fabs(Vx1);
        //        xout << Vx1 << endl;

        //        w = Velosity_inf - Vx1;
        //        p1 = fabs(w) * sqrtpi / (1.0 + fabs(w) * sqrtpi);
        //        do
        //        {
        //            ksi3 = sens->MakeRandom();
        //            ksi4 = sens->MakeRandom();
        //            ksi5 = sens->MakeRandom();
        //            ksi6 = sens->MakeRandom();

        //            if (p1 > ksi3)
        //            {
        //                z = cos(pi * ksi5) * sqrt(-log(ksi4));
        //            }
        //            else
        //            {
        //                if (ksi4 <= 0.5)
        //                {
        //                    z = -sqrt(-log(2.0 * ksi4));
        //                }
        //                else
        //                {
        //                    z = sqrt(-log(2.0 * (1.0 - ksi4)));
        //                }
        //            }
        //        } while (fabs(z + w) / (fabs(w) + fabs(z)) <= ksi6);
        //        Vx2 = z + Velosity_inf;

        //        if (Vx2 > 0)
        //        {
        //            n2 += mu2 / fabs(Vx2);
        //            VV2 += Vx2 * mu2 / fabs(Vx2);
        //            T2 += kv(Vx2) * mu2 / fabs(Vx2);
        //            xout3 << Vx2 << endl;
        //        }
        //        else
        //        {
        //            n4 += mu2 / fabs(Vx2);
        //            VV4 += Vx2 * mu2 / fabs(Vx2);
        //            T4 += kv(Vx2) * mu2 / fabs(Vx2);
        //            xout4 << Vx2 << endl;
        //        }
        //    }

        //    VV1 = VV1 / n1;
        //    T1 = (2.0) * (T1 / n1 - kv(VV1));
        //    n1 = (sq1 + sq2) *  n1 / (kk1 + kk2);

        //    VV2 = VV2 / n2;
        //    T2 = (2.0) * (T2 / n2 - kv(VV2));
        //    n2 = (sq1 + sq2) * n2 / (kk1 + kk2);

        //    VV3 = VV3 / n3;
        //    T3 = (2.0) * (T3 / n3 - kv(VV3));
        //    n3 = (sq1 + sq2) * n3 / (kk1 + kk2);

        //    VV4 = VV4 / n4;
        //    T4 = (2.0) * (T4 / n4 - kv(VV4));
        //    n4 = (sq1 + sq2) * n4 / (kk1 + kk2);

        //    watch(n1);
        //    watch(VV1);
        //    watch(T1);

        //    watch(n2);
        //    watch(VV2);
        //    watch(T2);

        //    watch(n3);
        //    watch(VV3);
        //    watch(T3);

        //    watch(n4);
        //    watch(VV4);
        //    watch(T4);
        //}
    }

    /*M_K(Sensors, host_s, host_u, nn1, nn2, nn3);
    string name = "Godunov_.txt";
    print_file_mini(host_s, host_u, nn1, nn2, nn3, name);*/


    //for (int i = 0; i < 30000; i = i + 2)  // Сколько шагов по времени делаем?
    //{
    //    Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, s2, u2, T, T_do, 1);
    //    cudaStatus = cudaGetLastError();
    //    if (cudaStatus != cudaSuccess) {
    //        fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //        exit(-1);
    //    }
    //    cudaStatus = cudaDeviceSynchronize();
    //    if (cudaStatus != cudaSuccess) {
    //        fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    //        exit(-1);
    //    }

    //    funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
    //    cudaStatus = cudaGetLastError();
    //    if (cudaStatus != cudaSuccess) {
    //        fprintf(stderr, "2  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //        exit(-1);
    //    }
    //    cudaStatus = cudaDeviceSynchronize();
    //    if (cudaStatus != cudaSuccess) {
    //        fprintf(stderr, "2  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    //        exit(-1);
    //    }

    //    Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, s, u, T, T_do, 1);
    //    cudaStatus = cudaGetLastError();
    //    if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));   exit(-1); }
    //    cudaStatus = cudaDeviceSynchronize();
    //    if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); exit(-1); }

    //    funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
    //    cudaStatus = cudaGetLastError();
    //    if (cudaStatus != cudaSuccess) {
    //        fprintf(stderr, "4  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //        exit(-1);
    //    }
    //    cudaStatus = cudaDeviceSynchronize();
    //    if (cudaStatus != cudaSuccess) {
    //        fprintf(stderr, "4  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    //        exit(-1);
    //    }

    //}

    for (int i = 0; i < 0; i = i + 2)  // Сколько шагов по времени делаем?
    {
        if (i == 0)
        {
            cout << "HLL + TVD" << endl;
        }
        Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, s2, u2, T, T_do, 0);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, s, u, T, T_do, 0);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));   exit(-1); }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); exit(-1); }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        if (i % 300 == 0 && i > 1)
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("300 step - Time:  %.2f sec\n", elapsedTime / 1000.0);
            cudaEventRecord(start, 0);
            /*cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
            string name = "14_06_" + to_string(i+1) + ".txt";
            print_file_mini(host_s, host_u, nn1, nn2, nn3, name);*/
        }
    }
    for (int i = 0; i < 0; i = i + 2)  // Сколько шагов по времени делаем?
    {
        if (i == 0)
        {
            cout << "HLLC + TVD" << endl;
        }
        Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, s2, u2, T, T_do, 1);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, s, u, T, T_do, 1);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));   exit(-1); }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); exit(-1); }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        if (i % 300 == 0 && i > 1)
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("300 step - Time:  %.2f sec\n", elapsedTime / 1000.0);
            cudaEventRecord(start, 0);
            /*cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
            string name = "14_06_" + to_string(i+1) + ".txt";
            print_file_mini(host_s, host_u, nn1, nn2, nn3, name);*/
        }
    }
    for (int i = 0; i < 0; i = i + 2)  // Сколько шагов по времени делаем?
    {
        if (i == 0)
        {
            cout << "Godunov + tvd" << endl;
        }
        Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, s2, u2, T, T_do, 2);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "2  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }
        
        Kernel_TVD << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, s, u, T, T_do, 2);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "3  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));   exit(-1); }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {fprintf(stderr, "3  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); exit(-1);}
        
        funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "4  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        if (i % 300 == 0 && i>1)
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("300 step - Time:  %.2f sec\n", elapsedTime/1000.0);
            cudaEventRecord(start, 0);
            /*cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
            string name = "14_06_" + to_string(i+1) + ".txt";
            print_file_mini(host_s, host_u, nn1, nn2, nn3, name);*/
        }
    }
    //for (int i = 0; i < 20000; i = i + 2)  // Сколько шагов по времени делаем?
    //{
    //    // запускаем add() kernel на GPU, передавая параметры
    //    Ker_Dekard << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s, u, s2, u2, T, T_do, 1);
    //    funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
    //    Ker_Dekard << < K / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (s2, u2, s, u, T, T_do, 1);
    //    funk_time << <1, 1 >> > (T, T_do, TT, dev_i);
    //}


    // copy device result back to host copy of c
    if (device)
    {
        cudaMemcpy(host_s, s, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_u, u, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_s2, s2, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_u2, u2, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_T, T, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_TT, TT, sizeof(double), cudaMemcpyDeviceToHost);


        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
    }




    printf("Time:  %.2f millisec\n", elapsedTime);

    if (device)
    {
        cudaFree(s);
        cudaFree(u);
        cudaFree(s2);
        cudaFree(u2);
        cudaFree(T);
        cudaFree(T_do);
        cudaFree(TT);
        cudaFree(dev_i);
        cudaFree(dev_nn1);
        cudaFree(dev_nn2);
        cudaFree(dev_nn3);
    }
    
    ofstream fout;
    fout.open("all_paramets.txt");

    ofstream fout2;
    fout2.open("param_for_texplot.txt");

    ofstream fout5;
    fout5.open("param_for_texplot_mini.txt");

    ofstream fout3;
    fout3.open("param_y=0.txt");

    ofstream fout4;
    fout4.open("inform.txt");

    fout2 << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\", \"Ro\", \"P\", \"Vx\", \"Vy\", \"Max\", \"T\", ZONE T = \"HP\", N = " << K //
        << " , E = " << (N - 1) * (M - 1) << ", F = FEPOINT, ET = quadrilateral" << endl;
    int nn = (int)((N + Nmin - 1) / Nmin);
    int mm = (int)((M + Nmin - 1) / Nmin);
    fout5 << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\", \"Ro\", \"P\", \"Vx\", \"Vy\", \"Max\", \"T\", \"Zav\", ZONE T = \"HP\", N = " << nn * mm //
        << " , E = " << (nn - 1)*(mm - 1) << ", F = FEPOINT, ET = quadrilateral" << endl;

    for (int k = 0; k < K; k++)
    {
        int n = k % N;                                   // номер ячейки по x (от 0)
        int m = (k - n) / N;                             // номер ячейки по y (от 0)
        double y = y_min + m * (y_max) / (M);
        double x = x_min + n * (x_max - x_min) / (N - 1);
        fout << x << " " << y << " " << host_s[k].x << " " << host_s[k].y <<//
            " " << host_u[k].x << " " << host_u[k].y << endl;
        //double Max = 0.0, Temp = 0.0;
        //if (host_s[k].x > 0)
        //{
        //    Max = sqrt((host_u[k].x * host_u[k].x + host_u[k].y * host_u[k].y) / (ggg * host_s[k].y / host_s[k].x));
        //    Temp = host_s[k].y / host_s[k].x;
        //}
        //fout2 << x / 184.0 << " " << y / 184.0 << " " << host_s[k].x << " " << host_s[k].y <<//
        //    " " << host_u[k].x << " " << host_u[k].y << " " << //
        //    Max << " " << Temp << endl;
    }

    //for (int k = 0; k < K; k++)
    //{
    //    int n = k % N;                                   // номер ячейки по x (от 0)
    //    int m = (k - n) / N;
    //    if ((m < M - 1) && (n < N - 1))
    //    {
    //        fout2 << m * N + n + 1 << " " << m * N + n + 2 << " " << (m + 1) * N + n + 2 << " " << (m + 1) * N + n + 1 << endl;
    //    }
    //}for (int k = 0; k < K; k++)
    //{
    //    int n = k % N;                                   // номер ячейки по x (от 0)
    //    int m = (k - n) / N;
    //    if ((m < M - 1) && (n < N - 1))
    //    {
    //        fout2 << m * N + n + 1 << " " << m * N + n + 2 << " " << (m + 1) * N + n + 2 << " " << (m + 1) * N + n + 1 << endl;
    //    }
    //}

    
    for (int k = 0; k < N; k++)
    {
        int n = k % N;                                   // номер ячейки по x (от 0)
        int m = (k - n) / N;                             // номер ячейки по y (от 0)
        double y = y_min + m * (y_max) / (M);
        double x = x_min + n * (x_max - x_min) / (N - 1);
        double ss = 0.0;
        if (host_s[k].x > 0)
        {
            ss = host_s[k].y / pow(host_s[k].x, ggg);
        }
        fout3 << x/184.0 << " " << y/184.0 << " " << host_s[k].x << " " << host_s[k].y <<//
            " " << host_u[k].x << " " << host_u[k].y << " " << ss << endl;
    }
    cout << "TT = " << *host_TT << endl;

    fout4 << "TT = " << *host_TT << "    N = " << N  << "   M = " << M << "   K = " << K  << endl;
    fout4 << "x_min = " << x_min << " " << "x_max = " << x_max << " " << "y_min = " << y_min << " " << "y_max = " << y_max << endl;
    fout4 << "M_inf = " << M_inf << " " << "phi_0 = " << phi_0 << endl;

    int lll = 0;

  

    for (int k = 0; k < K; k++)
    {
        int n = k % N;                                   // номер ячейки по x (от 0)
        int m = (k - n) / N;                             // номер ячейки по y (от 0)
        if ((n % Nmin != 0) || (m % Nmin != 0))
        {
            continue;
        }
        lll++;

        double zav = 0.0;
        if (n > 0 && m > 0 && n < N - 1 && m < M - 1)
        {
            zav = (host_u[(m)*N + n + 1].y - host_u[(m)*N + n - 1].y) / (2 * dx) - (host_u[(m + 1) * N + n].x - host_u[(m - 1) * N + n].x) / (2 * dy);
        }

        double y = y_min + m * (y_max) / (M);
        double x = x_min + n * (x_max - x_min) / (N - 1);
        double Max = 0.0, Temp = 0.0;
        if (host_s[k].x > 0.0)
        {
            Max = sqrt((host_u[k].x * host_u[k].x + host_u[k].y * host_u[k].y) / (ggg * host_s[k].y / host_s[k].x));
            Temp = host_s[k].y / host_s[k].x;
        }
        fout5 << x/184.0 << " " << y/184.0 << " " << host_s[k].x << " " << host_s[k].y <<//
            " " << host_u[k].x << " " << host_u[k].y << " " << //
            Max << " " << Temp << " "  << zav << endl;
    }
    cout << lll << " = lll " << endl;
    cout << nn << " = nn " << endl;
    cout << mm << " = mm " << endl;

    for (int k = 0; k < nn * mm; k = k + 1)
    {
        int n = k % nn;                                   // номер ячейки по x (от 0)
        int m = (k - n) / nn;
        if ((m < mm - 1) && (n < nn - 1))
        {
            fout5 << m * nn + n + 1 << " " << m * nn + n + 2 << " " << (m + 1) * nn + n + 2 << " " << (m + 1) * nn + n + 1 << endl;
        }
    }

    fout.close();
    fout2.close();
    fout3.close();
    fout4.close();
    fout5.close();

    return 0;
}