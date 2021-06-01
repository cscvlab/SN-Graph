#include "SNG.hpp"
#include "edt.hpp"
#include "tinyply.h"

namespace sng{

void SNGCalc::init(std::string voxel, std::string point) {
    surfacePointFile = point;
	memset(Voxel, 0, sizeof(Voxel));
	for (auto i = 0; i < axisSize; i++) {
		for (auto j = 0; j < axisSize; j++) {
			std::fill(std::begin(sdf[i][j]), std::end(sdf[i][j]), 100000000.0);
		}
	}
    while (priSdf.size())priSdf.pop();
	surfPoint.clear();
	usefulSphere.clear();
	trueUsefulSphere.clear();
	onlyPosSphere.clear();
	links.clear();
	subgraphLinks.clear();
	spherePriority.clear();
	surfacePointForRet.clear();
	sphereLessThanonlySphereNum = false;

	read_ply(voxel);
}

bool SNGCalc::sphere_lineseg_intersection(uint3& p1, uint3& p2, uint3& core, int r) {
	double d1 = uint3::dissq(core, p1), d2 = uint3::dissq(core, p2), d3 = uint3::dissq(p2, p1);
	auto rtd1 = sqrt(d1), rtd2 = sqrt(d2), rtd3 = sqrt(d3);
	auto cos1 = d1 + d3 - d2, cos2 = d2 + d3 - d1;
	if (cos1 <= 0 || cos2 <= 0)return false;
	auto c = (rtd1 + rtd2 + rtd3) / 2;
	auto s = sqrt(c * (c - rtd1) * (c - rtd2) * (c - rtd3));
	auto h = s * 2 / rtd3;
	if (h * h < 2 * r)return true;
	return false;
}

bool SNGCalc::connect_inside(int type, uint3& p1, uint3& p2)
{
	int dx = p2.x - p1.x, dy = p2.y - p1.y, dz = p2.z - p1.z, nx, ny, nz, score = 0;
	for (auto i = 0; i < 10; i++) {
		nx = p1.x + dx / 10.0 * i; ny = p1.y + dy / 10.0 * i; nz = p1.z + dz / 10.0 * i;
		if (Voxel[nx][ny][nz] & 0x01 || (sdf[nx][ny][nz] < outerLineAllowDistance)) score++;
	}
	if (score >= 7) return false;
	return true;
}

void SNGCalc::read_ply(std::string fileName) {
	namespace tp = tinyply;
    struct double3 { double x, y, z; };
    struct uint4 {
        uint32_t x, y, z, w;
        uint32_t operator [](int i) {
            if (i == 0)return x;
            else if (i == 1)return y;
            else if (i == 2)return z;
            else if (i == 3)return w;
            else throw - 1;
        }
    };
	try {
		std::ifstream ss(fileName, std::ios::binary);
		if (ss.fail()) throw std::runtime_error("failed to open " + fileName);

		tp::PlyFile ply;
		ply.parse_header(ss);

		{
			strings msgs;
			for (auto comment : ply.get_comments()) msgs.push_back(comment);
			for (auto element : ply.get_elements())
			{
				std::string str = "element - " + element.name + " (num: " + std::to_string(element.size) + ")";
				msgs.push_back(str);
				for (auto property : element.properties) {
					std::string Sstr = "\tproperty - " + property.name + " (type: " + tinyply::PropertyTable[property.propertyType].str + ")";
					msgs.push_back(Sstr);
				}
			}
		}

		std::shared_ptr<tp::PlyData> vertices;
		std::shared_ptr<tp::PlyData> faces;
		try {
			vertices = ply.request_properties_from_element("vertex", { "x", "y", "z" });
			faces = ply.request_properties_from_element("face", { "vertex_indices" }, 4);
		}
		catch (std::exception e) {
			std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
		}

		manualTimer read_timer;

		read_timer.start();
		ply.read(ss);
		read_timer.stop();
		outputMessage("Read Time", std::to_string(read_timer.get() / 1000.f));
		const size_t numVerticesBytes = vertices->buffer.size_bytes();
		std::vector<double3> verts(vertices->count);
		std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);

		const size_t numFaceBytes = faces->buffer.size_bytes();
		std::vector<uint4> face(faces->count);
		std::memcpy(face.data(), faces->buffer.get(), numFaceBytes);

		meanx = 0, meany = 0, meanz = 0;
		read_timer.start();

		// WARNING: round may cause overflow error, we need floor here
		for (auto& pt : verts) {
			pt.x = round((pt.x += 1.0000001) *= voxelSizeMultiply + 0.5) - 1;
			pt.y = round((pt.y += 1.0000001) *= voxelSizeMultiply + 0.5) - 1;
			pt.z = round((pt.z += 1.0000001) *= voxelSizeMultiply + 0.5) - 1;
			meanx += pt.x; meany += pt.y; meanz += pt.z;
			if (pt.x >= 0 && pt.x < axisSize && pt.y>0 && pt.y < axisSize && pt.z>0 && pt.z < axisSize)
				Voxel[(int)pt.x][(int)pt.y][(int)pt.z] = 0x05;
		}
		meanx /= verts.size();
		meany /= verts.size();
		meanz /= verts.size();

		for (auto& item : face) {
			for (auto i = 0; i < 4; i++) {
				auto& p = verts[item[i]];
				if (!(p.x >= 0 && p.x < axisSize && p.y>0 && p.y < axisSize && p.z>0 && p.z < axisSize)) continue;
				if ((Voxel[(int)p.x][(int)p.y][(int)p.z] & 0x80) != 0)continue;
				Voxel[(int)p.x][(int)p.y][(int)p.z] = 0x81;
				sdf[(int)p.x][(int)p.y][(int)p.z] = 0.0;
			}
		}
		read_timer.stop();
		outputMessage("Init Time", std::to_string(read_timer.get() / 1000.f));
	}
	catch (std::exception e) {
		std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
	}
	return;
}

void SNGCalc::calc_sdf_edt() {
	manualTimer read_timer;
	read_timer.start();
	bool* labels3d = new bool[axisSize * axisSize * axisSize]();
	for (auto i = 0; i < axisSize; i++) {
		for (auto j = 0; j < axisSize; j++) {
			for (auto k = 0; k < axisSize; k++) {
				if (Voxel[i][j][k] & 0x01 /*&& !(Voxel[i][j][k] & 0x80)*/) 
					labels3d[____(i, j, k)] = true;
				else labels3d[____(i, j, k)] = false;
			}
		}
	}
	float* dt = edt::edtsq<bool>(labels3d,
		axisSize, axisSize, axisSize,
		1, 1, 1,
		true, 8);
	for (auto i = 0; i < axisSize; i++) {
		for (auto j = 0; j < axisSize; j++) {
			for (auto k = 0; k < axisSize; k++) {
				if (labels3d[____(i, j, k)] == true) sdf[i][j][k] = dt[____(i, j, k)];
			}
		}
	}

	for (auto i = 0; i < axisSize; i++) {
		for (auto j = 0; j < axisSize; j++) {
			for (auto k = 0; k < axisSize; k++) {
				if (Voxel[i][j][k] == 0x00 ) labels3d[____(i, j, k)] = true;
				else labels3d[____(i, j, k)] = false;
			}
		}
	}
	float* dt2 = edt::edtsq<bool>(labels3d, axisSize, axisSize, axisSize, 1, 1, 1, true, 8);
	for (auto i = 0; i < axisSize; i++) {
		for (auto j = 0; j < axisSize; j++) {
			for (auto k = 0; k < axisSize; k++) {
				if (labels3d[____(i, j, k)] == true) sdf[i][j][k] = dt2[____(i, j, k)];
			}
		}
	}

	read_timer.stop();
	for (auto i = 0; i < axisSize; i++) {
		for (auto j = 0; j < axisSize; j++) {
			for (auto k = 0; k < axisSize; k++) {
				if (Voxel[i][j][k] & 0x04) sdf[i][j][k] = -sdf[i][j][k];
				if (sdf[i][j][k] >= 0) continue;
				if (sdf[i][j][k] <= -minSphereRadius) {
					priSdf.push(SDF4Sort(uint3(i, j, k), -sdf[i][j][k]));
				}
			}
		}
	}

	if(minSphereRadius > 0){
		// get all surface from txt file
    	// change this part for different point
		std::fstream fs;
		fs.open(surfacePointFile);
		float x, y, z;
		for (auto i = 0; i < 2048; i++) {
			fs >> x >> z >> y;
			x = (x + 1) * voxelSizeMultiply;
			y = (y + 1) * voxelSizeMultiply;
			z = (z + 1) * voxelSizeMultiply;
			priSdf.push(SDF4Sort(uint3((int)x, (int)y, (int)z), 0.25));
			surfacePointForRet.push_back(uint3((int)x, (int)y, (int)z));
		}
		fs.close();
		// get all surface from txt file
	}

	read_timer.stop();
	outputMessage("Translate Time", std::to_string(read_timer.get() / 1000.f));
	delete[] labels3d;
	delete[] dt;
	delete[] dt2;
	return;
}

void SNGCalc::sdf_first_filter() {
	std::vector<SDF4Sort> spTempSdf;
	std::cout << priSdf.size() << std::endl;
    while (priSdf.size()) {
        SDF4Sort sphere = priSdf.top();
        priSdf.pop();
        bool badSphere = false, NGSphere = false;
        float r1 = sqrt(sphere.radius);
        if (r1 < 1) {
            Sphere sph;
            sph.core = sphere.point; sph.radius = sphere.radius;
            usefulSphere.push_back(sph);
        }
        for (auto& i : usefulSphere) {
            float r2 = sqrt(i.radius), distance = sqrt(uint3::dissq(sphere.point, i.core));
            if (r1 + r2 >= distance) {
                badSphere = true;
                break;
            }
        }
        if (badSphere)continue;
        Sphere sph;
        sph.core = sphere.point; sph.radius = sphere.radius;
        usefulSphere.push_back(sph);
    }
	std::cout << usefulSphere.size() << std::endl;
}

void SNGCalc::get_first_sphere() {
	float max_r = usefulSphere[0].radius, min_r = 1.0;
	float max_dis = 0, dis;
	std::vector<float> disList;
	uint3 center(AxisSize / 2, AxisSize / 2, AxisSize / 2);
	for (auto it = usefulSphere.begin(); it->radius > (max_r / 4); it++) {
		dis = uint3::dis(it->core, center);
		if (dis > max_dis) max_dis = dis;
		disList.push_back(dis);
	}
	max_r = sqrt(max_r);
	for (auto i = 0; i < disList.size(); i++) {
		disList[i] = sqrt(usefulSphere[i].radius) / max_r * (sqrt(usefulSphere[i].radius) - min_r) - disList[i];
	}
	auto pos = usefulSphere.begin() + std::distance(std::begin(disList), std::max_element(std::begin(disList), std::end(disList)));
	trueUsefulSphere.push_back(*pos);
	usefulSphere.erase(pos);
}

void SNGCalc::select_sphere_surface() {
    manualTimer timer;
	timer.start();
	std::sort(usefulSphere.begin(), usefulSphere.end(), [](const auto& t1, const auto& t2) {
		return t1.radius > t2.radius;
		});
	get_first_sphere();
	while (trueUsefulSphere.size() < sphereNum && usefulSphere.size()!=0) {
		float disSumTemp = 0, disSum, dis;
		Sphere nextCore;
		std::vector<Sphere>::iterator nextCoreIt;
		bool changed = false;
		for (auto it = usefulSphere.begin(); it != usefulSphere.end(); it++) {
			disSum = 512 * 512 * 512;
			for (auto& core : trueUsefulSphere) {
				dis = sqrt((it->core.x - core.core.x) * (it->core.x - core.core.x) +
					(it->core.y - core.core.y) * (it->core.y - core.core.y) +
					(it->core.z - core.core.z) * (it->core.z - core.core.z));
				if ((dis - sqrt(core.radius) + 2 * sqrt(it->radius)) < disSum) 
					disSum = dis - sqrt(core.radius) + 2 * sqrt(it->radius);
			}
			if (disSum > disSumTemp) {
				disSumTemp = disSum;
				nextCore = *it;
				nextCoreIt = it;
				changed = true;
			}
		}
		if (changed == false) {
			sphereLessThanonlySphereNum = true;
			return;
		}
        trueUsefulSphere.push_back(nextCore);
		usefulSphere.erase(nextCoreIt);
	}
	timer.stop();
	outputMessage("Core Time", std::to_string(timer.get() / 1000.f));
	int pDistance;
	std::cout << trueUsefulSphere.size() << " " << surfPoint.size() << " " << surfacePointForRet.size() << std::endl;
	return;
}

void SNGCalc::link_graph(int maxlink) {
	int maxLink = maxlink;
	if (maxlink == -1) maxLink = this->maxLink;
	using Link = std::tuple<int, int, int, float> ;
	std::vector<std::vector<Link>> tempLinks;
	float distance;
	auto comp = [](const Link& t1, const Link& t2) {
		return std::get<3>(t1) < std::get<3>(t2);
	};
	for (auto i = 0; i < trueUsefulSphere.size(); i++) {
		tempLinks.push_back({}); subgraphLinks.push_back({});
	}
	for (auto i = 0; i < trueUsefulSphere.size(); i++) {
		for (auto j = i + 1; j < trueUsefulSphere.size(); j++) {
			distance = uint3::dis(trueUsefulSphere[i].core, trueUsefulSphere[j].core)
				- sqrt(trueUsefulSphere[i].radius) - sqrt(trueUsefulSphere[j].radius);
			// link must inside, or close
			if (distance > 1) {
				auto badLink = connect_inside(0, trueUsefulSphere[i].core, trueUsefulSphere[j].core);
				if (badLink) continue;
				// link can't contact other sphere
				for (auto k = 0; k < trueUsefulSphere.size(); k++) {
					if (k == i || k == j) continue;
					badLink = sphere_lineseg_intersection(trueUsefulSphere[i].core, trueUsefulSphere[j].core,
						trueUsefulSphere[k].core, trueUsefulSphere[k].radius);
					if (badLink)break;
				}
				if (badLink) continue;
			}

			tempLinks[i].push_back(std::make_tuple(0, i, j, distance));
			tempLinks[j].push_back(std::make_tuple(0, i, j, distance));
		}
		if (tempLinks[i].size() == 0) { // if the sphere is isolated
			std::cout << "Empty Sphere ID: " << i << std::endl;
			int nearestSphereId; float shortestDis = 512 * 512;
			for (auto j = 0; j < trueUsefulSphere.size(); j++) {
				if (i == j)continue;
				auto distance = uint3::dis(trueUsefulSphere[i].core, trueUsefulSphere[j].core)
					- sqrt(trueUsefulSphere[i].radius) - sqrt(trueUsefulSphere[j].radius);
				if (distance < shortestDis) {
					nearestSphereId = j; shortestDis = distance;
				}
			}
			// CAUTION NO SURFACE POINT IN THIS SEGMENT
			tempLinks[i].push_back(std::make_tuple(0, i, nearestSphereId, distance));
			tempLinks[nearestSphereId].push_back(std::make_tuple(0, i, nearestSphereId, distance));
		}
		std::sort(tempLinks[i].begin(), tempLinks[i].end(), comp);
	}
	for (auto i = 0; i < trueUsefulSphere.size(); i++) {
		bool allSmaller = true;
		for (auto j = 0; j < std::min(maxLink, static_cast<int>(tempLinks[i].size())); j++) {
			auto [type, p1, p2, dis] = tempLinks[i][j];
			int sphere = p1;
			if (p1 == i) sphere = p2;
			if (trueUsefulSphere[p1].radius < trueUsefulSphere[p2].radius) {
				allSmaller = false; break;
			}
		}
		// if all k nearest are smaller than sphere i, find a biggest sphere
		if (allSmaller) {
			int biggestId = -1, radiusA = 0, nearestBiggerId = -1;
			float radiusB = 512 * 512;
			for (auto j = 0; j < trueUsefulSphere.size(); j++) {
				if (i == j)continue;
				if (trueUsefulSphere[j].radius > radiusA) {
					biggestId = j;
					radiusA = trueUsefulSphere[j].radius;
				}
				if (trueUsefulSphere[i].radius < trueUsefulSphere[j].radius) {
					int dis = uint3::dis(trueUsefulSphere[i].core, trueUsefulSphere[j].core);
					if (dis < radiusB) {
						nearestBiggerId = j;
						radiusB = dis;
					}
				}
			}
			if (nearestBiggerId == -1) {
				subgraphLinks[i].push_back(biggestId);
				float distance = uint3::dis(trueUsefulSphere[i].core, trueUsefulSphere[biggestId].core)
					- sqrt(trueUsefulSphere[i].radius) - sqrt(trueUsefulSphere[biggestId].radius);
				tempLinks[i].push_back(std::make_tuple(0, i, biggestId, distance));
				tempLinks[biggestId].push_back(std::make_tuple(0, i, biggestId, distance));
				subgraphLinks[biggestId].push_back(i);
			}
			else {
				subgraphLinks[i].push_back(nearestBiggerId);
				float distance = uint3::dis(trueUsefulSphere[i].core, trueUsefulSphere[nearestBiggerId].core)
					- sqrt(trueUsefulSphere[i].radius) - sqrt(trueUsefulSphere[nearestBiggerId].radius);
				tempLinks[i].push_back(std::make_tuple(0, i, nearestBiggerId, distance));
				tempLinks[nearestBiggerId].push_back(std::make_tuple(0, i, nearestBiggerId, distance));
				subgraphLinks[nearestBiggerId].push_back(i);
			}
		}
		for (auto j = 0; j < std::min(maxLink, static_cast<int>(tempLinks[i].size())); j++) {
			auto [type, p1, p2, dis] = tempLinks[i][j];
			links.push_back({ type,p1,p2 });
			subgraphLinks[p1].push_back(p2);
			subgraphLinks[p2].push_back(p1);
		}
	}
}

void SNGCalc::construct_multigraph() {
	std::set<int> spheres;
	std::vector<std::vector<int>> subgraph;
	int searchedId, subgrphId = 0;
	for (auto i = 0; i < trueUsefulSphere.size(); i++)
		spheres.insert(i);

	// split to some subgraphs
	while (spheres.size()) {
		searchedId = 0;
		subgraph.push_back({});
		auto id = *spheres.begin();
		subgraph[subgrphId].push_back(id);
		spheres.erase(id);
		while (searchedId < subgraph[subgrphId].size()) {
			for (auto i : subgraphLinks[subgraph[subgrphId][searchedId]]) {
				if (std::find(subgraph[subgrphId].begin(), subgraph[subgrphId].end(), i) == subgraph[subgrphId].end()) {
					subgraph[subgrphId].push_back(i);
					spheres.erase(i);
				}
			}
			searchedId++;
			if (spheres.size() == 0)break;
		}
		subgrphId++;
	}
	if (subgraph.size() == 1)return;

	// get most clost point pair between every two subset
	std::vector<std::tuple<int, int, int, int, int>> minSubgraphPair;
	for (auto i = 0; i < subgraph.size(); i++) {
		for (auto j = i + 1; j < subgraph.size(); j++) {
			int mindis = 1e6, dis, node1, node2;
			for (auto k1 = 0; k1 < subgraph[i].size(); k1++) {
				for (auto k2 = 0; k2 < subgraph[j].size(); k2++) {
					dis = uint3::dissq(trueUsefulSphere[subgraph[i][k1]].core, trueUsefulSphere[subgraph[j][k2]].core);
					if (dis < mindis) {
						node1 = subgraph[i][k1], node2 = subgraph[j][k2];
					}
				}
			}
			minSubgraphPair.push_back({ i, j, node1, node2, dis }); // clostest pair between i & j
			//links.push_back({ 0,node1,node2 });
		}
	}
	std::sort(minSubgraphPair.begin(), minSubgraphPair.end(), [](const auto& t1, const auto& t2) {
		return std::get<4>(t1) < std::get<4>(t2);
		});
	// link subset to a set
	std::vector<int> combinedGraph;
	int subsetNum = subgraph.size();
	for (auto i = 0; i < subgraph.size(); i++)
		combinedGraph.push_back(i);
	while (subsetNum != 1) {
		auto [subset1, subset2, node1, node2, dis] = minSubgraphPair[0];
		minSubgraphPair.erase(minSubgraphPair.begin());
		int subsetIndexPre = combinedGraph[subset2], subsetIndexAfter = combinedGraph[subset1];
		links.push_back({ 0,node1,node2 });
		for (auto i = 0; i < combinedGraph.size(); i++) {
			if (combinedGraph[i] == subsetIndexPre)
				combinedGraph[i] = subsetIndexAfter;
		}
		for (auto it = minSubgraphPair.begin(); it != minSubgraphPair.end(); ) {
			auto [subset1, subset2, node1, node2, dis] = *it;
			if (combinedGraph[subset1] == combinedGraph[subset2]) it = minSubgraphPair.erase(it);
			else it++;
			if (minSubgraphPair.size() <= 1) break;
		}
		std::sort(minSubgraphPair.begin(), minSubgraphPair.end(), [](const auto& t1, const auto& t2) {
			return std::get<4>(t1) < std::get<4>(t2);
			});
		subsetNum--;
	}
	return;
}

void SNGCalc::start(std::string voxel, std::string point) {
	init(voxel, point);
	calc_sdf_edt();
	sdf_first_filter();
	select_sphere_surface();
	if (trueUsefulSphere.size() < 2) {
		std::cout << "Voxel File Error" << std::endl;
		return;
	}
	link_graph();
	construct_multigraph();
	return;
}

void outputMessage(std::string title, strings msgs) {
	std::cout << title << ":" << std::endl;

	for (auto item : msgs) {
		std::cout << "    " << item << std::endl;
	}

	std::cout << std::endl << std::endl;
}

void outputMessage(std::string title, std::string msg) {
	std::cout << title << ": " << msg << std::endl;
}

std::vector<Sphere>& sng::SNGCalc::get_sphere()
{
	return trueUsefulSphere;
}

std::vector<std::tuple<int, int, int>>& sng::SNGCalc::get_links()
{
	return links;
}

}