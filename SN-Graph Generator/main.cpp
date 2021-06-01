#include "SNG.hpp"
#include <string>

int main(int argc, char* argv[]){
    sng::SNGCalc sngGenerator;
    sngGenerator.start("./test_data/airplane.ply", "");
    auto spheres = sngGenerator.get_sphere();
    auto links = sngGenerator.get_links();
    
    // save spheres and links by the way you prefer

    return 0;
}