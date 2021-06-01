#pragma once

#include <string>
#include <vector>
#include <queue>
#include <set>
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>
#include <tuple>

namespace sng{

using strings = std::vector<std::string>;

struct int3 {
	int x, y, z;
	int3() { x = 0, y = 0, z = 0; }
	int3(int _x, int _y, int _z) { x = _x, y = _y, z = _z; }
	int operator *(int3 a) {
		return x * a.x + y * a.y + z * a.z;
	}
};

struct uint3 {
	uint32_t x, y, z;
	uint3(int _x, int _y, int _z) { x = _x, y = _y, z = _z; }
	uint3() { x = 0, y = 0, z = 0; }
	bool operator ==(uint3& a) {
		if (x == a.x && y == a.y && z == a.z)return true;
		return false;
	}
	int3 operator -(uint3& a) {
		return int3(x - a.x, y - a.y, z - a.z);
	}
	int3 operator +(uint3& a) {
		return int3(x + a.x, y + a.y, z + a.z);
	}
	int operator *(uint3 a) {
		return x * a.x + y * a.y + z * a.z;
	}
	int operator *(uint3& a) {
		return x * a.x + y * a.y + z * a.z;
	}
	static int dissq(uint3& a, uint3& b) {
		return ((int)a.x - (int)b.x) * ((int)a.x - (int)b.x) +
			((int)a.y - (int)b.y) * ((int)a.y - (int)b.y) +
			((int)a.z - (int)b.z) * ((int)a.z - (int)b.z);
	}
	static float dis(uint3& a, uint3& b) {
		return sqrt(((int)a.x - (int)b.x) * ((int)a.x - (int)b.x) +
			((int)a.y - (int)b.y) * ((int)a.y - (int)b.y) +
			((int)a.z - (int)b.z) * ((int)a.z - (int)b.z));
	}
};

struct SDF4Sort
{
	uint3 point;
	float radius;
	SDF4Sort(uint3 p, float d) {
		point = p; radius = d;
	}
	bool operator < (const SDF4Sort& a) const
	{
		return a.radius > radius;
	}
};

struct Sphere {
	uint3 core;
	float radius;
	std::vector<uint3> contactSurf;
	std::vector<int> contactSphere;
};

class manualTimer
{
	std::chrono::high_resolution_clock::time_point t0;
	double timestamp{ 0.f };
public:
	void start() { t0 = std::chrono::high_resolution_clock::now(); }
	void stop() { timestamp = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count() * 1000; }
	const double& get() { return timestamp; }
};

class SNGCalc {
#define AxisSize 128

    std::string surfacePointFile;
	const double pi = std::atan(1) * 4;
	const short axisSize = AxisSize;
	const int minSphereRadius = 0;
	const int voxelSizeMultiply = 64;
	const int maxLink = 6;
	const int outerLineAllowDistance = 9;
	const int min0_5SphereDistance = 64;
	const bool useSurfPoint = false;

    int sphereNum = 256;
	int bfsOffset[26][3] = {
		{-1,-1,-1},{0,-1,-1},{1,-1,-1},
		{-1,-1,0},{0,-1,0},{1,-1,0},
		{-1,-1,1},{0,-1,1},{1,-1,1},
		{-1,0,-1},{0,0,-1},{1,0,-1},
		{-1,0,0},
		{1,1,1},{0,1,1},{-1,1,1},
		{1,1,0},{0,1,0},{-1,1,0},
		{1,1,-1},{0,1,-1},{-1,1,-1},
		{1,0,1},{0,0,1},{-1,0,1},
		{1,0,0}
	};
	unsigned char Voxel[AxisSize][AxisSize][AxisSize];
	float sdf[AxisSize][AxisSize][AxisSize];
	std::vector<uint3> surfPoint;
	long int meanx, meany, meanz;
	int gapx, gapy, gapz;
	double area;

	std::priority_queue<SDF4Sort> priSdf;
	std::vector<SDF4Sort> spherePriority;
	std::vector<Sphere> usefulSphere;
	std::vector<Sphere> trueUsefulSphere;
	std::vector<Sphere> onlyPosSphere;
	std::vector<uint3> surfacePointForRet;
	std::vector<std::tuple<int, int, int>> links;
	std::vector<std::vector<int>> subgraphLinks;

	inline bool in_bound(uint32_t x, uint32_t y, uint32_t z) {
		return !(x >= axisSize || y >= axisSize || z >= axisSize);
	}
	inline bool in_bound(uint3& p) {
		return in_bound(p.x, p.y, p.z);
	}
	inline bool PcontactP(uint3 a, uint3 b) {
		if (fabs(a.x - b.x) <= 1 && fabs(a.y - b.y) <= 1 && fabs(a.z - b.z) <= 1)return true;
		return false;
	}
	inline int ____(int x, int y, int z) {
		return x * axisSize * axisSize + y * axisSize + z;
	}
	inline bool occur(uint3 p, std::vector<uint3>& pu) {
		for (auto& i : pu) {
			if (i == p)return true;
		}
		return false;
	}
	
	bool sphere_lineseg_intersection(uint3& p1, uint3& p2, uint3& core, int r);
	bool connect_inside(int type, uint3& p1, uint3& p2);

	void read_ply(std::string fileName);
	void calc_sdf_edt();
	void sdf_first_filter();
	void get_first_sphere();
	void select_sphere_surface();
	void link_graph(int maxlink = -1);
	void construct_multigraph();

public:
    bool sphereLessThanonlySphereNum = false;

    // Input Need Voxel File and Surface Point File, you may need to customize the read_ply function to load you voxel file
    void init(std::string voxel, std::string point);
	void start(std::string voxel, std::string point);
	// returned sphere radius is squared
	std::vector<Sphere>& get_sphere();
	std::vector<std::tuple<int, int, int>>& get_links();
	void set_sphnum(int n) {
		sphereNum = n;
	}
	SNGCalc(){
		memset(Voxel, 0, sizeof(Voxel));
        ClassMessgae();
	}
	std::string ClassMessgae() {
		return "Voxel To SDF\nVoxel:"
			+ std::to_string(axisSize) + "\tSphere:"+ std::to_string(sphereNum) + "\n";
	}
	std::tuple<int, int, int> get_gap() {
		return { gapx,gapy,gapz };
	}	
};

void outputMessage(std::string, strings);
void outputMessage(std::string, std::string);

}