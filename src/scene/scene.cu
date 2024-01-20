#include "scene/scene.hpp"

#include "scene/sphere.hpp"
#include "math/ray.hpp"

#include "hit_details.hpp"

#include <climits>

namespace scene
{
    scene_t::scene_t()
    {
        spheres = (sphere_t**)malloc(sizeof(sphere_t*) * max_sphere_count);
        materials = (material::material_t**)malloc(sizeof(material::material_t*) * max_material_count);
    }

    scene_t::~scene_t()
    {
        for (int i = 0; i < num_materials; i++)
        {
            free(materials[i]);
        }
        free(materials);

        for (int i = 0; i < num_spheres; i++)
        {
            free(spheres[i]);
        }

        free(spheres);
    }

    void scene_t::add_sphere(sphere_t& sphere)
    {
        if (num_spheres == max_sphere_count)
        {
            std::cout << "Not adding sphere to scene due to exceeding max sphere count.\n";
            return;
        }

        spheres[num_spheres++] = new sphere_t(sphere) ;
    }
        
    uint32_t scene_t::add_material(material::material_t* mat)
    {
        if (num_materials == max_material_count)
        {
            std::cout << "Not adding material to scene due to exceeding max material count.\n";
            return 0; 
        }

        materials[num_materials++] = (mat);
        return num_materials - 1;
    }

    __device__ hit_details_t scene_t::ray_hit(const math::ray_t& ray) const
    {
        // The value of t is slightly greater than 0 because of shadow acne.
        // There are situations where due to floating point precision problems, we may have 
        // a case where the ray param at t is not the exact value, causing the intersection point to be 
        // beneath the surface rather than on it. This will cause the ray to continuously intersect the surface of the sphere,
        // causing continous intersections results in false - shadowing.
        // By setting min_t to be greater than 0, this problem can be resolved.
        float min_t = 0.01f;
        float max_t = 3.40282347e+38F;

        hit_details_t ray_hit_details{};
        bool ray_hit_object_in_scene = false;

        for (int i = 0; i < num_spheres; i++)
        {
            const sphere_t* sphere = spheres[i];
            const auto t = sphere->hit_by_ray(ray, min_t, max_t);
            if (t != -1.0f)
            {
                // Fill hit_details struct.
                ray_hit_details.ray_param_t = t;
                ray_hit_details.point_of_intersection = ray.at(t);

                math::float3 normal = (ray_hit_details.point_of_intersection - sphere->center) / sphere->radius;

                // To find if the ray hit a back face or front face.
                // If the angle between normal and ray direction is greater than 90, then
                // the ray hit a front face.
                const auto normal_and_ray_direction_dot_product = math::float3::dot(ray.direction.normalize(), normal);
                if (normal_and_ray_direction_dot_product < 0.0f)
                {
                    ray_hit_details.back_face = false;
                    ray_hit_details.normal = normal;
                }
                else
                {
                    ray_hit_details.back_face = true;
                    ray_hit_details.normal = normal * -1.0f;
                }

                ray_hit_details.material_index =sphere->mat_index; 

                max_t = t;

                ray_hit_object_in_scene = true;
            }
        }

        if (!ray_hit_object_in_scene)
        {
            ray_hit_details.ray_param_t = -1.0f;
        } 

        return ray_hit_details;
    }
}
