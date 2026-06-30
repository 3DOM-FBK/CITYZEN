cmake_minimum_required(VERSION 3.15)

project(ExtrudeSkeletonExample)

# Usa C++17 (necessario per alcune funzionalità di CGAL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Trova CGAL
find_package(CGAL REQUIRED)

# Trova Boost (CGAL può aver bisogno di Boost)
find_package(Boost REQUIRED)

# Aggiungi l'eseguibile
add_executable(extrude_skeleton skeleton.cpp)

# Linka CGAL e Boost all'eseguibile
target_link_libraries(extrude_skeleton CGAL::CGAL Boost::boost)
