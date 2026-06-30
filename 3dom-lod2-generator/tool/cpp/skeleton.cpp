#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/extrude_skeleton.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <cstdlib>

typedef CGAL::Exact_predicates_inexact_constructions_kernel     K;
typedef K::Point_2                                              Point2;
typedef K::Point_3                                              Point3;
typedef CGAL::Polygon_2<K>                                      Polygon_2;
typedef CGAL::Polygon_with_holes_2<K>                           Polygon_with_holes;

typedef CGAL::Surface_mesh<Point3>                             Mesh;


/**
 * @brief Reads polygon data from a text file formatted with exterior and holes.
 * 
 * Parses a text file containing polygon vertices organized into an exterior boundary 
 * and zero or more holes. The file format expects sections labeled "EXTERIOR", "HOLE", 
 * and "END", with vertex coordinates as pairs of doubles on separate lines.
 * 
 * Example input format:
 * EXTERIOR
 * x1 y1
 * x2 y2
 * ...
 * HOLE
 * x1 y1
 * x2 y2
 * ...
 * END
 * 
 * @param filename The path to the input text file.
 * @param exterior Vector to store points of the exterior polygon (output).
 * @param holes Vector of vectors to store points of each hole polygon (output).
 * 
 * @return true if the file was read and parsed successfully, false otherwise.
 */
bool read_polygon_data(const std::string& filename,
                       std::vector<Point2>& exterior,
                       std::vector<std::vector<Point2>>& holes)
{
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error file open: " << filename << std::endl;
        return false;
    }

    exterior.clear();
    holes.clear();

    std::string line;
    std::vector<Point2>* current_points = nullptr;

    while (std::getline(infile, line)) {
        if (line.empty()) continue;

        if (line == "EXTERIOR") {
            current_points = &exterior;
            continue;
        }
        else if (line == "HOLE") {
            holes.emplace_back();
            current_points = &holes.back();
            continue;
        }
        else if (line == "END") {
            current_points = nullptr;
            continue;
        }

        if (current_points) {
            std::istringstream iss(line);
            double x, y;
            if (!(iss >> x >> y)) {
                std::cerr << "Error point parsing: " << line << std::endl;
                return false;
            }
            current_points->emplace_back(x, y);
        }
        else {
            // continue
        }
    }

    return true;
}


/**
 * @brief Scales all vertices of a mesh.
 * 
 * Scales all vertices of a mesh by the given factors along each axis.
 * 
 * @param mesh The mesh to be scaled (modified in-place).
 * @param sx Scaling factor along the X axis.
 * @param sy Scaling factor along the Y axis.
 * @param sz Scaling factor along the Z axis.
 */
void scale_mesh(Mesh& mesh, double sx, double sy, double sz) {
    for (auto v : mesh.vertices()) {
        Point3 p = mesh.point(v);
        mesh.point(v) = Point3(p.x() * sx, p.y() * sy, p.z() * sz);
    }
}

// --- esempio di uso ---
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Use: " << argv[0] << " <input.txt> <output.ply> <max_height>" << std::endl;
        return EXIT_FAILURE;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    float height = std::atof(argv[3]);

    std::vector<Point2> exterior;
    std::vector<std::vector<Point2>> holes;

    Polygon_2 outer;
    Polygon_with_holes poly;

    if (read_polygon_data(input_path, exterior, holes)) {
        // std::cout << "EXTERIOR points: " << exterior.size() << std::endl;
        for (const auto& p : exterior)
            outer.push_back( p ) ;
        
        assert(outer.is_counterclockwise_oriented());
        poly = Polygon_with_holes( outer );

        // std::cout << "Number of holes: " << holes.size() << std::endl;
        for (size_t i = 0; i < holes.size(); ++i) {
            std::cout << "Hole " << i << " points: " << holes[i].size() << std::endl;
            Polygon_2 hole;
            hole.clear();
            for (const auto& p : holes[i])
                hole.push_back( p );
            assert(hole.is_clockwise_oriented());
            poly.add_hole( hole );
        }
    }
    else {
        std::cerr << "Error loading polygons." << std::endl;
    }

    Mesh sm;
    CGAL::extrude_skeleton(poly, sm, CGAL::parameters::maximum_height(height));

    scale_mesh(sm, 1.0, 1.0, 0.5);

    CGAL::IO::write_polygon_mesh(output_path, sm, CGAL::parameters::stream_precision(17));

    return 0;
}
