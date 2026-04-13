/*
 * Generate planar UVs for genus-0 closed meshes by:
 * 1) removing one face to create a topological disk
 * 2) harmonic parameterization on the disk
 */

#include <SurfaceMaps/Init.hh>
#include <SurfaceMaps/Utils/IO.hh>

#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>

#include <iostream>
#include <string>

using namespace SurfaceMaps;

namespace
{
std::string get_arg(int argc, char** argv, const std::string& name, const std::string& default_value = "")
{
    for (int i = 1; i + 1 < argc; ++i)
        if (name == argv[i])
            return argv[i + 1];
    return default_value;
}

void print_usage(const char* exe_name)
{
    std::cout << "Usage:\n"
              << "  " << exe_name << " --in mesh.off --out mesh_uv.obj\n";
}
} // namespace

int main(int argc, char** argv)
{
    const std::string in_arg = get_arg(argc, argv, "--in");
    const std::string out_arg = get_arg(argc, argv, "--out");
    if (in_arg.empty() || out_arg.empty())
    {
        print_usage(argv[0]);
        return 1;
    }

    init_lib_surface_maps();

    TriMesh mesh = read_mesh(fs::path(in_arg));
    MatXd V;
    MatXi F;
    mesh_to_matrix(mesh, V, F);

    if (F.rows() < 4)
        ISM_ERROR_throw("Mesh has too few faces for planar parameterization: " << in_arg);

    // Cut open to a disk by dropping one face.
    MatXi F_disk(F.rows() - 1, 3);
    F_disk = F.topRows(F.rows() - 1);

    Eigen::VectorXi bnd;
    igl::boundary_loop(F_disk, bnd);
    if (bnd.size() < 3)
        ISM_ERROR_throw("Failed to extract boundary loop after cut: " << in_arg);

    MatXd bnd_uv;
    igl::map_vertices_to_circle(V, bnd, bnd_uv);

    MatXd V_uv;
    bool ok = igl::harmonic(V, F_disk, bnd, bnd_uv, 1, V_uv);
    if (!ok)
    {
        // Fallback to uniform Laplacian harmonic solve.
        ok = igl::harmonic(F_disk, bnd, bnd_uv, 1, V_uv);
    }
    if (!ok)
        ISM_ERROR_throw("Harmonic planar parameterization failed: " << in_arg);

    mesh.request_halfedge_texcoords2D();
    for (auto heh : mesh.halfedges())
    {
        const int vi = heh.to().idx();
        const float u = (float)V_uv(vi, 0);
        const float v = (float)V_uv(vi, 1);
        mesh.set_texcoord2D(heh, TriMesh::TexCoord2D(u, v));
    }

    write_mesh(mesh, fs::path(out_arg));
    return 0;
}

