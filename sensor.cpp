#include <stdio.h>
#include "sensor.h"
using namespace std;

Sensor::Sensor(int a1, int a2, int a3):
    a1_(a1), a2_(a2), a3_(a3), a1_0(a1), a2_0(a2), a3_0(a3) {}

double Sensor::MakeRandom() {
    int ic15 = 32768,ic10 = 1024;
    int mz = 710, my = 17784, mx = 11973;
    double xi = 9.0949470177292824E-13, c = 1.073741824E9;
    double b;
    int i13,i12,i11,ii;
    i13 = mz * a1_ + my * a2_ + mx * a3_;
    i12 = my * a1_ + mx * a2_;
    i11 = mx * a1_;
    ii = i11 / ic15;
    i12 = i12 + ii;
    a1_ = i11 - ic15 * ii;
    ii = i12 / ic15;
    i13 = i13 + ii;
    a2_ = i12 - ic15 * ii;
    a3_ = i13 % ic10;
    b = xi * (c * a3_ + ic15 * a2_ + a1_);
    return b;
}

void Sensor::Restart()
{
    this->a1_ = this->a1_0;
    this->a2_ = this->a2_0;
    this->a3_ = this->a3_0;
}

/*std::vector<Sensor> InitSensors() {
    std::vector<Sensor> sensors;
    FILE *rnd;
    rnd = fopen("rnd_Dima.dat","r");

    for (int i = 0; i < SENSORS_AMOUNT; ++i) {
        double intermediate;
        int i1, i2, i3;
        fscanf(rnd,"%lf %d %d %d",&intermediate, &i1, &i2, &i3);
        sensors.push_back(Sensor(i1, i2, i3));
	}

    return sensors;
}*/
