#pragma once

#include <vector>
#include <mutex>
/* Each sensor is determined by 3 integers for initial point. Structure of all sensors is the same. */
using namespace std;
const int SENSORS_AMOUNT = 270;

class Sensor {
public:
    Sensor(int a1, int a2, int a3);

    // Generate random number by this sensor.
    double MakeRandom();
    void Restart();
    int a1_;
    int a2_;
    int a3_;
    int a1_0;
    int a2_0;
    int a3_0;
};

//std::vector<Sensor> InitSensors();
