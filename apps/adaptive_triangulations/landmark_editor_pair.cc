/*
 * This file is part of
 * Surface Maps via Adaptive Triangulations
 * (https://github.com/patr-schm/surface-maps-via-adaptive-triangulations)
 * and is released under the MIT license.
 */

#include <SurfaceMaps/Init.hh>
#include <SurfaceMaps/Utils/IO.hh>
#include <SurfaceMaps/Utils/MeshNormalization.hh>
#include <SurfaceMaps/Viewer/MeshView.hh>
#include <SurfaceMaps/Viewer/Picking.hh>

#include <glow-extras/glfw/GlfwContext.hh>
#include <glow-extras/viewer/canvas.hh>
#include <imgui/imgui.h>

#include <algorithm>
#include <iostream>
#include <string>

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
              << "  " << exe_name << " --mesh-a a.obj --mesh-b b.obj [options]\n\n"
              << "Options:\n"
              << "  --out-a <path>         Output landmarks for A (.pinned)\n"
              << "  --out-b <path>         Output landmarks for B (.pinned)\n"
              << "  --texture-a <png>      Optional texture for A\n"
              << "  --texture-b <png>      Optional texture for B\n"
              << "  --load-existing        Load existing pinned files if present\n";
}

void draw_landmark_ids(const TriMesh& mesh, const std::vector<VH>& landmarks)
{
    auto c = gv::canvas();
    for (int i = 0; i < (int)landmarks.size(); ++i)
    {
        auto vh = landmarks[i];
        const auto p = mesh.point(vh);
        c.add_label(tg::pos3(p[0], p[1], p[2]), std::to_string(i));
    }
}

} // namespace

int main(int argc, char** argv)
{
    const std::string mesh_a_arg = get_arg(argc, argv, "--mesh-a");
    const std::string mesh_b_arg = get_arg(argc, argv, "--mesh-b");
    if (mesh_a_arg.empty() || mesh_b_arg.empty())
    {
        print_usage(argv[0]);
        return 1;
    }

    const fs::path mesh_path_a = fs::path(mesh_a_arg);
    const fs::path mesh_path_b = fs::path(mesh_b_arg);

    fs::path landmarks_path_a = get_arg(argc, argv, "--out-a");
    fs::path landmarks_path_b = get_arg(argc, argv, "--out-b");
    if (landmarks_path_a.empty())
        landmarks_path_a = mesh_path_a.parent_path() / (mesh_path_a.stem().string() + ".pinned");
    if (landmarks_path_b.empty())
        landmarks_path_b = mesh_path_b.parent_path() / (mesh_path_b.stem().string() + ".pinned");

    const fs::path texture_path_a = get_arg(argc, argv, "--texture-a");
    const fs::path texture_path_b = get_arg(argc, argv, "--texture-b");
    (void)texture_path_a;
    (void)texture_path_b;
    const bool load_existing = has_flag(argc, argv, "--load-existing");

    glow::glfw::GlfwContext ctx;
    init_lib_surface_maps();

    TriMesh mesh_a = read_mesh(mesh_path_a);
    TriMesh mesh_b = read_mesh(mesh_path_b);
    center_mesh(mesh_a);
    center_mesh(mesh_b);
    normalize_surface_area(mesh_a);
    normalize_surface_area(mesh_b);

    std::vector<VH> landmarks_a;
    std::vector<VH> landmarks_b;
    if (load_existing)
    {
        landmarks_a = read_landmarks(landmarks_path_a, LandmarkType::Keep, false);
        landmarks_b = read_landmarks(landmarks_path_b, LandmarkType::Keep, false);
        const int n = std::min((int)landmarks_a.size(), (int)landmarks_b.size());
        landmarks_a.resize(n);
        landmarks_b.resize(n);
    }

    bool edit_mesh_a = true;
    int pending_a = -1;
    bool show_ids = true;
    bool overwrite = true;
    bool auto_save_on_pair = false;

    gv::interactive([&](auto)
    {
        ImGui::Begin("Pair Landmark Editor");
        ImGui::Text("3D pick on active mesh (middle mouse).");
        ImGui::Text("Workflow: pick A -> pick B to form one pair.");
        ImGui::Separator();

        ImGui::Text("Mesh A: %s", mesh_path_a.string().c_str());
        ImGui::Text("Mesh B: %s", mesh_path_b.string().c_str());
        ImGui::Text("Out A: %s", landmarks_path_a.string().c_str());
        ImGui::Text("Out B: %s", landmarks_path_b.string().c_str());

        ImGui::Separator();
        ImGui::Text("Pair count: %d", (int)landmarks_a.size());
        if (pending_a >= 0)
            ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1), "Pending A vertex id: %d (now click B)", pending_a);
        else
            ImGui::TextColored(ImVec4(0.5f, 0.9f, 1.0f, 1), "Active mesh: %s (middle mouse picks)", edit_mesh_a ? "A" : "B");

        ImGui::Checkbox("Show landmark IDs", &show_ids);
        ImGui::Checkbox("Overwrite files", &overwrite);
        ImGui::Checkbox("Auto-save on each completed pair", &auto_save_on_pair);

        if (ImGui::Button("Edit mesh A"))
            edit_mesh_a = true;
        ImGui::SameLine();
        if (ImGui::Button("Edit mesh B"))
            edit_mesh_a = false;

        if (ImGui::Button("Undo last pair"))
        {
            if (!landmarks_a.empty())
            {
                landmarks_a.pop_back();
                landmarks_b.pop_back();
            }
            pending_a = -1;
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear all"))
        {
            landmarks_a.clear();
            landmarks_b.clear();
            pending_a = -1;
        }

        if (ImGui::Button("Save landmarks now"))
        {
            write_landmarks(landmarks_a, landmarks_path_a, {}, overwrite);
            write_landmarks(landmarks_b, landmarks_path_b, {}, overwrite);
        }

        ImGui::Text("Middle click: pick on active mesh. Left drag: orbit camera.");
        ImGui::Text("Right click: cancel pending A.");
        ImGui::End();

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
            pending_a = -1;

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle))
        {
            if (pending_a >= 0)
            {
                const VH vh_b = pick_vertex(mesh_b);
                if (vh_b.is_valid())
                {
                    landmarks_a.push_back(VH(pending_a));
                    landmarks_b.push_back(vh_b);
                    pending_a = -1;
                    edit_mesh_a = true;

                    if (auto_save_on_pair)
                    {
                        write_landmarks(landmarks_a, landmarks_path_a, {}, true);
                        write_landmarks(landmarks_b, landmarks_path_b, {}, true);
                    }
                }
            }
            else if (edit_mesh_a)
            {
                const VH vh_a = pick_vertex(mesh_a);
                if (vh_a.is_valid())
                {
                    pending_a = vh_a.idx();
                    edit_mesh_a = false;
                }
            }
            else
            {
                // We enforce A->B pairing to keep order clear.
            }
        }

        auto g = gv::grid();
        {
            auto v = gv::view();
            view_caption(edit_mesh_a ? "Active: A" : "Active: B");
            if (edit_mesh_a)
            {
                view_mesh(mesh_a, Color(0.9, 0.9, 0.9, 1.0));
                view_landmarks(mesh_a, landmarks_a, WidthScreen(10.0));
                if (show_ids)
                    draw_landmark_ids(mesh_a, landmarks_a);
            }
            else
            {
                view_mesh(mesh_b, Color(0.9, 0.9, 0.9, 1.0));
                view_landmarks(mesh_b, landmarks_b, WidthScreen(10.0));
                if (show_ids)
                    draw_landmark_ids(mesh_b, landmarks_b);
            }
            if (pending_a >= 0)
                view_landmarks(mesh_a, std::vector<VH>{VH(pending_a)}, WidthScreen(16.0));
        }
    });

    // Final save on exit.
    write_landmarks(landmarks_a, landmarks_path_a, {}, true);
    write_landmarks(landmarks_b, landmarks_path_b, {}, true);

    return 0;
}
