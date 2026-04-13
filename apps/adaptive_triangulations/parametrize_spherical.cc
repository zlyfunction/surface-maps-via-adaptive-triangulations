/*
 * Generate UVs for genus-0 closed meshes via spherical embedding.
 */

#include <SurfaceMaps/Init.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/MultiRes/MultiResSphereEmbedding.hh>

#include <cmath>
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

Vec2d sphere_to_uv(const Vec3d& p_in)
{
    const Vec3d p = p_in.normalized();
    const double pi = 3.14159265358979323846;
    const double u = std::atan2(p[2], p[0]) / (2.0 * pi) + 0.5;
    const double v = std::asin(std::max(-1.0, std::min(1.0, p[1]))) / pi + 0.5;
    return Vec2d(u, v);
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
    auto embedding = multi_res_sphere_embedding(mesh);

    std::vector<Vec2d> uv_vertex(mesh.n_vertices(), Vec2d(0.0, 0.0));
    for (auto vh : mesh.vertices())
        uv_vertex[vh.idx()] = sphere_to_uv(embedding[vh]);

    mesh.request_halfedge_texcoords2D();
    for (auto heh : mesh.halfedges())
    {
        const auto uv = uv_vertex[heh.to().idx()];
        mesh.set_texcoord2D(heh, TriMesh::TexCoord2D((float)uv[0], (float)uv[1]));
    }

    write_mesh(mesh, fs::path(out_arg));
    return 0;
}

