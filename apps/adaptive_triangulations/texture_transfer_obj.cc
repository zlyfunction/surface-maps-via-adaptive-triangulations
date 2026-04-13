/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 */

#include <SurfaceMaps/Init.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/Viewer/MeshView.hh>
#include <SurfaceMaps/AdaptiveTriangulations/Helpers.hh>
#include <SurfaceMaps/AdaptiveTriangulations/InitSphereEmbeddings.hh>
#include <SurfaceMaps/AdaptiveTriangulations/LiftToSurface.hh>
#include <SurfaceMaps/AdaptiveTriangulations/OptimizeCoarseToFine.hh>

#include <glow-extras/glfw/GlfwContext.hh>
#include <glow-extras/viewer/view.hh>
#include <glow/detail/lodepng/lodepng.hh>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace SurfaceMaps;

namespace
{

bool has_flag(int argc, char** argv, const std::string& name)
{
    for (int i = 1; i < argc; ++i)
        if (name == argv[i])
            return true;
    return false;
}

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
              << "  " << exe_name << " --mesh-a a.obj --mesh-b b.obj --texture-a tex.png [options]\n\n"
              << "Options:\n"
              << "  --out-dir <path>         Output directory (default: build/output/texture_transfer_obj)\n"
              << "  --landmarks-a <path>     Optional landmarks for mesh A (.pinned)\n"
              << "  --landmarks-b <path>     Optional landmarks for mesh B (.pinned)\n"
              << "  --edit-landmarks         Open pair landmark editor before optimization\n"
              << "  --source-tex-mode <m>    mesh_uv (default) | projection\n"
              << "  --projection-dir <d>     0|1|2, used when source-tex-mode=projection (default: 1)\n"
              << "  --projection-factor <f>  UV scale for projection mode (default: 1.0)\n"
              << "  --bake-size <N>          Output texture resolution (default: 2048)\n"
              << "  --open-viewer            Open interactive viewer instead of only screenshot\n";
}

std::string shell_quote(const std::string& s)
{
    std::string out = "\"";
    for (char c : s)
    {
        if (c == '"' || c == '\\')
            out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::vector<Vec2d> representative_vertex_uvs(const TriMesh& mesh)
{
    ISM_ASSERT(mesh.has_halfedge_texcoords2D());

    std::vector<Vec2d> uvs(mesh.n_vertices(), Vec2d(0.0, 0.0));
    std::vector<bool> has_uv(mesh.n_vertices(), false);

    for (auto heh : mesh.halfedges())
    {
        auto vh = mesh.to_vertex_handle(heh);
        if (!has_uv[vh.idx()])
        {
            auto uv = mesh.texcoord2D(heh);
            uvs[vh.idx()] = Vec2d(uv[0], uv[1]);
            has_uv[vh.idx()] = true;
        }
    }

    for (auto vh : mesh.vertices())
        ISM_ASSERT(has_uv[vh.idx()]);

    return uvs;
}

std::vector<VH> pick_landmarks_from_uv(const TriMesh& mesh, const std::vector<Vec2d>& vertex_uvs)
{
    std::vector<VH> landmarks;
    std::vector<bool> used(mesh.n_vertices(), false);

    // Use canonical UV anchors. This assumes both meshes have semantically aligned UV charts.
    const std::array<Vec2d, 5> anchors = {Vec2d(0.10, 0.10), Vec2d(0.90, 0.10), Vec2d(0.90, 0.90), Vec2d(0.10, 0.90), Vec2d(0.50, 0.50)};

    for (const auto& a : anchors)
    {
        double best_dist = std::numeric_limits<double>::infinity();
        VH best_vh;
        bool found = false;

        for (auto vh : mesh.vertices())
        {
            if (used[vh.idx()])
                continue;
            const double d = (vertex_uvs[vh.idx()] - a).squaredNorm();
            if (d < best_dist)
            {
                best_dist = d;
                best_vh = vh;
                found = true;
            }
        }

        if (found)
        {
            landmarks.push_back(best_vh);
            used[best_vh.idx()] = true;
        }
    }

    ISM_ASSERT_GEQ(landmarks.size(), 2);
    return landmarks;
}

TexCoords transfer_source_uv_to_target(const MapState& map_state, int source_mesh_idx, int target_mesh_idx)
{
    ISM_ASSERT(map_state.meshes_input[source_mesh_idx].has_halfedge_texcoords2D());

    TexCoords texcoords_target(map_state.meshes_input[target_mesh_idx]);
    TriMesh mesh_embedding_T_target = embedding_to_mesh(map_state.mesh_T, map_state.embeddings_T[target_mesh_idx]);
    BSPTree bsp_T_i(mesh_embedding_T_target);

    for (auto heh : map_state.meshes_input[target_mesh_idx].halfedges())
    {
        SVH v_to = heh.to();
        SFH fh_T;
        double alpha_T, beta_T, gamma_T;
        bsp_tree_barys_face(map_state.embeddings_input[target_mesh_idx][v_to], mesh_embedding_T_target, bsp_T_i, alpha_T, beta_T, gamma_T, fh_T);
        BarycentricPoint bary_T(fh_T, alpha_T, beta_T, map_state.mesh_T);

        const Vec3d p_emb_on_ref = bary_T.interpolate(map_state.embeddings_T[source_mesh_idx], map_state.mesh_T).normalized();

        SFH fh_ref;
        double alpha_ref, beta_ref, gamma_ref;
        bsp_tree_barys_face(
                p_emb_on_ref,
                map_state.meshes_embeddings_input[source_mesh_idx],
                map_state.bsp_embeddings_input[source_mesh_idx],
                alpha_ref,
                beta_ref,
                gamma_ref,
                fh_ref);

        SHEH heh_a, heh_b, heh_c;
        handles(map_state.meshes_input[source_mesh_idx], fh_ref, heh_a, heh_b, heh_c);

        texcoords_target[heh] = map_state.meshes_input[source_mesh_idx].texcoord2D(heh_a) * alpha_ref
                + map_state.meshes_input[source_mesh_idx].texcoord2D(heh_b) * beta_ref
                + map_state.meshes_input[source_mesh_idx].texcoord2D(heh_c) * gamma_ref;
    }

    return texcoords_target;
}

TexCoords projected_source_texcoords(const TriMesh& source_mesh, int projection_dir, double texture_factor)
{
    ISM_ASSERT_GEQ(projection_dir, 0);
    ISM_ASSERT_LEQ(projection_dir, 2);

    TexCoords source_texcoords(source_mesh);
    for (auto heh : source_mesh.halfedges())
    {
        const auto p = source_mesh.point(heh.to());
        source_texcoords[heh] = Vec2d(
                p[(projection_dir + 1) % 3] * texture_factor,
                p[(projection_dir + 2) % 3] * texture_factor);
    }
    return source_texcoords;
}

TexCoords transfer_source_texcoords_to_target(
        const MapState& map_state,
        int source_mesh_idx,
        int target_mesh_idx,
        const TexCoords& source_texcoords)
{
    TexCoords texcoords_target(map_state.meshes_input[target_mesh_idx]);
    TriMesh mesh_embedding_T_target = embedding_to_mesh(map_state.mesh_T, map_state.embeddings_T[target_mesh_idx]);
    BSPTree bsp_T_i(mesh_embedding_T_target);

    for (auto heh : map_state.meshes_input[target_mesh_idx].halfedges())
    {
        SVH v_to = heh.to();
        SFH fh_T;
        double alpha_T, beta_T, gamma_T;
        bsp_tree_barys_face(map_state.embeddings_input[target_mesh_idx][v_to], mesh_embedding_T_target, bsp_T_i, alpha_T, beta_T, gamma_T, fh_T);
        BarycentricPoint bary_T(fh_T, alpha_T, beta_T, map_state.mesh_T);

        const Vec3d p_emb_on_ref = bary_T.interpolate(map_state.embeddings_T[source_mesh_idx], map_state.mesh_T).normalized();

        SFH fh_ref;
        double alpha_ref, beta_ref, gamma_ref;
        bsp_tree_barys_face(
                p_emb_on_ref,
                map_state.meshes_embeddings_input[source_mesh_idx],
                map_state.bsp_embeddings_input[source_mesh_idx],
                alpha_ref,
                beta_ref,
                gamma_ref,
                fh_ref);

        SHEH heh_a, heh_b, heh_c;
        handles(map_state.meshes_input[source_mesh_idx], fh_ref, heh_a, heh_b, heh_c);

        texcoords_target[heh] = source_texcoords[heh_a] * alpha_ref
                + source_texcoords[heh_b] * beta_ref
                + source_texcoords[heh_c] * gamma_ref;
    }
    return texcoords_target;
}

double edge_function(const Vec2d& a, const Vec2d& b, const Vec2d& c)
{
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
}

std::array<unsigned char, 4> sample_texture_bilinear_clamp_rgba(
        const std::vector<unsigned char>& image,
        unsigned width,
        unsigned height,
        const Vec2d& uv)
{
    auto clamp01 = [](double x) { return std::max(0.0, std::min(1.0, x)); };

    const double u = clamp01(uv[0]);
    const double v = clamp01(uv[1]);
    const double x = u * (double)(width - 1);
    const double y = (1.0 - v) * (double)(height - 1);

    const int x0 = std::max(0, std::min((int)width - 1, (int)std::floor(x)));
    const int y0 = std::max(0, std::min((int)height - 1, (int)std::floor(y)));
    const int x1 = std::max(0, std::min((int)width - 1, x0 + 1));
    const int y1 = std::max(0, std::min((int)height - 1, y0 + 1));
    const double tx = x - (double)x0;
    const double ty = y - (double)y0;

    auto at = [&](int ix, int iy, int c) -> double
    {
        return (double)image[4 * (iy * (int)width + ix) + c];
    };

    std::array<unsigned char, 4> out = {0, 0, 0, 255};
    for (int c = 0; c < 4; ++c)
    {
        const double c00 = at(x0, y0, c);
        const double c10 = at(x1, y0, c);
        const double c01 = at(x0, y1, c);
        const double c11 = at(x1, y1, c);
        const double c0 = c00 * (1.0 - tx) + c10 * tx;
        const double c1 = c01 * (1.0 - tx) + c11 * tx;
        const double cv = c0 * (1.0 - ty) + c1 * ty;
        out[c] = (unsigned char)std::max(0.0, std::min(255.0, std::round(cv)));
    }
    return out;
}

void dilate_texture_colors(std::vector<unsigned char>& image, int w, int h, int iterations)
{
    for (int it = 0; it < iterations; ++it)
    {
        auto src = image;
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                const int idx = 4 * (y * w + x);
                if (src[idx + 3] != 0)
                    continue;

                int cnt = 0;
                int sum_r = 0, sum_g = 0, sum_b = 0;
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        if (dx == 0 && dy == 0)
                            continue;
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h)
                            continue;
                        const int nidx = 4 * (ny * w + nx);
                        if (src[nidx + 3] == 0)
                            continue;
                        sum_r += src[nidx + 0];
                        sum_g += src[nidx + 1];
                        sum_b += src[nidx + 2];
                        cnt++;
                    }
                }

                if (cnt > 0)
                {
                    image[idx + 0] = (unsigned char)(sum_r / cnt);
                    image[idx + 1] = (unsigned char)(sum_g / cnt);
                    image[idx + 2] = (unsigned char)(sum_b / cnt);
                    image[idx + 3] = 255;
                }
            }
        }
    }
}

void bake_texture_per_pixel_via_sphere(
        const MapState& map_state,
        int source_mesh_idx,
        int target_mesh_idx,
        const fs::path& source_texture_path,
        const fs::path& out_png_path,
        int out_size)
{
    const TriMesh& target_mesh = map_state.meshes_input[target_mesh_idx];
    const TriMesh& source_mesh = map_state.meshes_input[source_mesh_idx];

    if (!target_mesh.has_halfedge_texcoords2D())
        ISM_ERROR_throw("Target mesh has no UVs. Cannot bake in target UV layout.");
    if (!source_mesh.has_halfedge_texcoords2D())
        ISM_ERROR_throw("Source mesh has no UVs. Cannot sample source texture.");

    std::vector<unsigned char> src_image_rgba;
    unsigned src_w = 0, src_h = 0;
    const unsigned decode_err = glow_lodepng::decode(src_image_rgba, src_w, src_h, source_texture_path.string(), LCT_RGBA, 8);
    if (decode_err != 0)
        ISM_ERROR_throw("Failed to decode source texture: " << source_texture_path << " (" << glow_lodepng_error_text(decode_err) << ")");

    TriMesh mesh_embedding_T_target = embedding_to_mesh(map_state.mesh_T, map_state.embeddings_T[target_mesh_idx]);
    BSPTree bsp_T_target(mesh_embedding_T_target);

    const int w = out_size;
    const int h = out_size;
    std::vector<unsigned char> out_rgba((size_t)w * (size_t)h * 4, 0);

    for (auto fh : target_mesh.faces())
    {
        HEH heh_a, heh_b, heh_c;
        handles(target_mesh, fh, heh_a, heh_b, heh_c);

        const VH va = target_mesh.to_vertex_handle(heh_a);
        const VH vb = target_mesh.to_vertex_handle(heh_b);
        const VH vc = target_mesh.to_vertex_handle(heh_c);

        const Vec2d uv_dst_a = Vec2d(target_mesh.texcoord2D(heh_a)[0], target_mesh.texcoord2D(heh_a)[1]);
        const Vec2d uv_dst_b = Vec2d(target_mesh.texcoord2D(heh_b)[0], target_mesh.texcoord2D(heh_b)[1]);
        const Vec2d uv_dst_c = Vec2d(target_mesh.texcoord2D(heh_c)[0], target_mesh.texcoord2D(heh_c)[1]);

        const Vec3d emb_a = map_state.embeddings_input[target_mesh_idx][va];
        const Vec3d emb_b = map_state.embeddings_input[target_mesh_idx][vb];
        const Vec3d emb_c = map_state.embeddings_input[target_mesh_idx][vc];

        auto uv_to_pixel = [w, h](const Vec2d& uv)
        {
            return Vec2d(uv[0] * (double)(w - 1), (1.0 - uv[1]) * (double)(h - 1));
        };

        const Vec2d p0 = uv_to_pixel(uv_dst_a);
        const Vec2d p1 = uv_to_pixel(uv_dst_b);
        const Vec2d p2 = uv_to_pixel(uv_dst_c);

        const double area = edge_function(p0, p1, p2);
        if (std::abs(area) < 1e-12)
            continue;

        const int min_x = std::max(0, (int)std::floor(std::min({p0[0], p1[0], p2[0]})));
        const int max_x = std::min(w - 1, (int)std::ceil(std::max({p0[0], p1[0], p2[0]})));
        const int min_y = std::max(0, (int)std::floor(std::min({p0[1], p1[1], p2[1]})));
        const int max_y = std::min(h - 1, (int)std::ceil(std::max({p0[1], p1[1], p2[1]})));

        for (int y = min_y; y <= max_y; ++y)
        {
            for (int x = min_x; x <= max_x; ++x)
            {
                const Vec2d p((double)x + 0.5, (double)y + 0.5);
                const double w0 = edge_function(p1, p2, p);
                const double w1 = edge_function(p2, p0, p);
                const double w2 = edge_function(p0, p1, p);

                const bool inside = (area > 0.0) ? (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) : (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
                if (!inside)
                    continue;

                const double b0 = w0 / area;
                const double b1 = w1 / area;
                const double b2 = w2 / area;

                const Vec3d p_sphere_B = (b0 * emb_a + b1 * emb_b + b2 * emb_c).normalized();

                SFH fh_T;
                double alpha_T, beta_T, gamma_T;
                bsp_tree_barys_face(p_sphere_B, mesh_embedding_T_target, bsp_T_target, alpha_T, beta_T, gamma_T, fh_T);
                BarycentricPoint bary_T(fh_T, alpha_T, beta_T, map_state.mesh_T);

                const Vec3d p_emb_on_src = bary_T.interpolate(map_state.embeddings_T[source_mesh_idx], map_state.mesh_T).normalized();

                SFH fh_src;
                double alpha_src, beta_src, gamma_src;
                bsp_tree_barys_face(
                        p_emb_on_src,
                        map_state.meshes_embeddings_input[source_mesh_idx],
                        map_state.bsp_embeddings_input[source_mesh_idx],
                        alpha_src, beta_src, gamma_src, fh_src);

                SHEH heh_src_a, heh_src_b, heh_src_c;
                handles(source_mesh, fh_src, heh_src_a, heh_src_b, heh_src_c);

                const Vec2d uv_src(
                    source_mesh.texcoord2D(heh_src_a)[0] * alpha_src
                  + source_mesh.texcoord2D(heh_src_b)[0] * beta_src
                  + source_mesh.texcoord2D(heh_src_c)[0] * gamma_src,
                    source_mesh.texcoord2D(heh_src_a)[1] * alpha_src
                  + source_mesh.texcoord2D(heh_src_b)[1] * beta_src
                  + source_mesh.texcoord2D(heh_src_c)[1] * gamma_src);

                const auto rgba = sample_texture_bilinear_clamp_rgba(src_image_rgba, src_w, src_h, uv_src);

                const int out_idx = 4 * (y * w + x);
                out_rgba[out_idx + 0] = rgba[0];
                out_rgba[out_idx + 1] = rgba[1];
                out_rgba[out_idx + 2] = rgba[2];
                out_rgba[out_idx + 3] = rgba[3];
            }
        }
    }

    fs::create_directories(out_png_path.parent_path());
    const unsigned encode_err = glow_lodepng::encode(out_png_path.string(), out_rgba, (unsigned)w, (unsigned)h, LCT_RGBA, 8);
    if (encode_err != 0)
        ISM_ERROR_throw("Failed to encode baked texture: " << out_png_path << " (" << glow_lodepng_error_text(encode_err) << ")");
}

} // namespace

int main(int argc, char** argv)
{
    const std::string mesh_a_arg = get_arg(argc, argv, "--mesh-a");
    const std::string mesh_b_arg = get_arg(argc, argv, "--mesh-b");
    const std::string texture_a_arg = get_arg(argc, argv, "--texture-a");
    if (mesh_a_arg.empty() || mesh_b_arg.empty() || texture_a_arg.empty())
    {
        print_usage(argv[0]);
        return 1;
    }

    const fs::path mesh_path_A = fs::path(mesh_a_arg);
    const fs::path mesh_path_B = fs::path(mesh_b_arg);
    const fs::path texture_path_A = fs::path(texture_a_arg);
    const fs::path output_dir = get_arg(argc, argv, "--out-dir").empty() ? (OUTPUT_PATH / "texture_transfer_obj") : fs::path(get_arg(argc, argv, "--out-dir"));
    const fs::path screenshot_dir = output_dir / "screenshots";
    const fs::path embedding_path_A = output_dir / "embedding_A.obj";
    const fs::path embedding_path_B = output_dir / "embedding_B.obj";
    const fs::path landmarks_path_A_arg = get_arg(argc, argv, "--landmarks-a");
    const fs::path landmarks_path_B_arg = get_arg(argc, argv, "--landmarks-b");
    const bool edit_landmarks = has_flag(argc, argv, "--edit-landmarks");
    const bool open_viewer = has_flag(argc, argv, "--open-viewer");
    const std::string source_tex_mode = get_arg(argc, argv, "--source-tex-mode", "mesh_uv");
    const int projection_dir = get_arg(argc, argv, "--projection-dir").empty() ? 1 : std::stoi(get_arg(argc, argv, "--projection-dir"));
    const double projection_factor = get_arg(argc, argv, "--projection-factor").empty() ? 1.0 : std::stod(get_arg(argc, argv, "--projection-factor"));
    const int bake_size = get_arg(argc, argv, "--bake-size").empty() ? 2048 : std::max(128, std::stoi(get_arg(argc, argv, "--bake-size")));

    fs::create_directories(output_dir);
    fs::create_directories(screenshot_dir);

    fs::path landmarks_path_A = landmarks_path_A_arg;
    fs::path landmarks_path_B = landmarks_path_B_arg;

    if (edit_landmarks)
    {
        if (landmarks_path_A.empty())
            landmarks_path_A = output_dir / "manual_landmarks_A.pinned";
        if (landmarks_path_B.empty())
            landmarks_path_B = output_dir / "manual_landmarks_B.pinned";

        const fs::path editor_exe = fs::absolute(fs::path(argv[0])).parent_path() / "landmark_editor_pair";
        if (!fs::exists(editor_exe))
            ISM_ERROR_throw("Cannot find landmark editor executable at " << editor_exe);

        std::stringstream cmd;
        cmd << shell_quote(editor_exe.string())
            << " --mesh-a " << shell_quote(mesh_path_A.string())
            << " --mesh-b " << shell_quote(mesh_path_B.string())
            << " --out-a " << shell_quote(landmarks_path_A.string())
            << " --out-b " << shell_quote(landmarks_path_B.string())
            << " --texture-a " << shell_quote(texture_path_A.string())
            << " --load-existing";

        ISM_INFO("Launching landmark editor...");
        const int editor_ret = std::system(cmd.str().c_str());
        if (editor_ret != 0)
            ISM_ERROR_throw("landmark_editor_pair exited with non-zero code: " << editor_ret);
    }

    if (landmarks_path_A.empty() || landmarks_path_B.empty())
    {
        ISM_INFO("No landmarks provided. Auto-generating pseudo-landmarks from UV anchors.");

        TriMesh mesh_A_in = read_mesh(mesh_path_A);
        TriMesh mesh_B_in = read_mesh(mesh_path_B);
        if (!mesh_A_in.has_halfedge_texcoords2D() || !mesh_B_in.has_halfedge_texcoords2D())
            ISM_ERROR_throw("Auto landmark generation requires UVs in both OBJ files.");

        const auto uvs_A = representative_vertex_uvs(mesh_A_in);
        const auto uvs_B = representative_vertex_uvs(mesh_B_in);
        const auto landmarks_A = pick_landmarks_from_uv(mesh_A_in, uvs_A);
        const auto landmarks_B = pick_landmarks_from_uv(mesh_B_in, uvs_B);
        ISM_ASSERT_EQ(landmarks_A.size(), landmarks_B.size());

        landmarks_path_A = output_dir / "auto_landmarks_A.pinned";
        landmarks_path_B = output_dir / "auto_landmarks_B.pinned";
        write_landmarks(landmarks_A, landmarks_path_A, {}, true);
        write_landmarks(landmarks_B, landmarks_path_B, {}, true);
    }

    glow::glfw::GlfwContext ctx;
    init_lib_surface_maps();

    MapState map_state;
    if (!init_map(
                map_state,
                {mesh_path_A, mesh_path_B},
                {landmarks_path_A, landmarks_path_B},
                {embedding_path_A, embedding_path_B},
                false))
        ISM_ERROR_throw("Map initialization failed.");

    optimize_coarse_to_fine(map_state);

    glow::SharedTexture2D texture_A = read_texture(texture_path_A);
    TexCoords uv_A_on_B(map_state.meshes_input[1]);
    if (source_tex_mode == "projection")
    {
        ISM_INFO("Using projection source texcoords. dir=" << projection_dir << ", factor=" << projection_factor);
        const TexCoords source_texcoords = projected_source_texcoords(map_state.meshes_input[0], projection_dir, projection_factor);
        uv_A_on_B = transfer_source_texcoords_to_target(map_state, 0, 1, source_texcoords);
    }
    else
    {
        uv_A_on_B = transfer_source_uv_to_target(map_state, 0, 1);
    }

    TriMesh mesh_B_with_A_uv = map_state.meshes_input[1];
    mesh_B_with_A_uv.request_halfedge_texcoords2D();
    for (auto heh : mesh_B_with_A_uv.halfedges())
    {
        const Vec2d uv = uv_A_on_B[heh];
        mesh_B_with_A_uv.set_texcoord2D(heh, TriMesh::TexCoord2D((float)uv[0], (float)uv[1]));
    }
    write_mesh(mesh_B_with_A_uv, output_dir / "B_with_A_uv.obj");

    // Render B with transferred UVs using A's texture as a quick visual sanity check.
    {
        const auto cam_pos = glow::viewer::camera_transform(tg::pos3(-1.4f, 2.1f, 0.6f), tg::pos3(0.f, 0.f, 0.f));
        auto s = screenshot_config(open_viewer, screenshot_dir / "B_with_A_texture.png", cam_pos, tg::ivec2(1920, 1080), true);
        auto v = gv::view();
        view_mesh(make_renderable(map_state.meshes_input[1], uv_A_on_B, texture_A));
    }

    // Bake a new texture image in B's original UV layout (per-pixel sphere lookup).
    const fs::path baked_tex_path = output_dir / "B_uv_baked_texture.png";
    bake_texture_per_pixel_via_sphere(
            map_state,
            0, 1,
            texture_path_A,
            baked_tex_path,
            bake_size);

    // Verification render: apply baked texture to B using B's own UVs.
    {
        TexCoords uv_B_own = tex_coords(map_state.meshes_input[1]);
        glow::SharedTexture2D baked_tex = read_texture(baked_tex_path);
        const auto cam_pos = glow::viewer::camera_transform(tg::pos3(-1.4f, 2.1f, 0.6f), tg::pos3(0.f, 0.f, 0.f));
        auto s = screenshot_config(open_viewer, screenshot_dir / "B_baked_verify.png", cam_pos, tg::ivec2(1920, 1080), true);
        auto v = gv::view();
        view_mesh(make_renderable(map_state.meshes_input[1], uv_B_own, baked_tex));
    }

    ISM_INFO("Done. Outputs written to " << output_dir);
    return 0;
}
