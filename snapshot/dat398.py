import warp as wp
from newton._src.solvers.vbd.particle_vbd_kernels import *
wp.set_module_options({"enable_backward": False})

@wp.kernel
def apply_planar_lanes398(
    # inputs
    pos: wp.array[wp.vec3],
    displacement_in: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    gamma: float,
    truncation_t_out: wp.array[float],
):
    """Truncate particle displacements against soft self-contact division planes, one thread group per EE or VT pair."""
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // 32
    t_id_current_primitive = t_id % 32

    # process edge-edge collisions
    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        collision_count = get_edge_colliding_edges_count(collision_info, primitive_id)
        while collision_buffer_counter < collision_count:
            e2_idx = collision_info.edge_colliding_edges[2 * (collision_buffer_offset + collision_buffer_counter) + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]

                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]

                e2_v1 = edge_indices[e2_idx, 2]
                e2_v2 = edge_indices[e2_idx, 3]

                e2_v1_pos = pos[e2_v1]
                e2_v2_pos = pos[e2_v2]

                delta_e2_v1 = displacement_in[e2_v1]
                delta_e2_v2 = displacement_in[e2_v2]

                # n points to the edge 1 side
                valid_plane, n, d, separation_eps = create_edge_edge_division_plane_closest_pt(
                    e1_v1_pos,
                    delta_e1_v1,
                    e1_v2_pos,
                    delta_e1_v2,
                    e2_v1_pos,
                    delta_e2_v1,
                    e2_v2_pos,
                    delta_e2_v2,
                )

                # Without a strict separator at the reference (the edges intersect within
                # float precision) DAT cannot tell which side is safe, so the pair is left
                # unconstrained for this pass: freezing it would pin both edges at the
                # reference indefinitely. Contact forces separate it and a later detection
                # certifies a separator.
                if valid_plane:
                    t = planar_truncation_t(e1_v1_pos, delta_e1_v1, n, d, gamma, separation_eps)
                    if t < 1.0:
                        wp.atomic_min(truncation_t_out, e1_v1, t)
                    t = planar_truncation_t(e1_v2_pos, delta_e1_v2, n, d, gamma, separation_eps)
                    if t < 1.0:
                        wp.atomic_min(truncation_t_out, e1_v2, t)
                    t = planar_truncation_t(e2_v1_pos, delta_e2_v1, -n, d, gamma, separation_eps)
                    if t < 1.0:
                        wp.atomic_min(truncation_t_out, e2_v1, t)
                    t = planar_truncation_t(e2_v2_pos, delta_e2_v2, -n, d, gamma, separation_eps)
                    if t < 1.0:
                        wp.atomic_min(truncation_t_out, e2_v2, t)
            collision_buffer_counter += 32

    # process vertex-triangle collisions
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id

        colliding_particle_pos = pos[particle_idx]
        colliding_particle_displacement = displacement_in[particle_idx]

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        collision_count = get_vertex_colliding_triangles_count(collision_info, primitive_id)
        while collision_buffer_counter < collision_count:
            tri_idx = collision_info.vertex_colliding_triangles[
                (collision_buffer_offset + collision_buffer_counter) * 2 + 1
            ]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                t1 = pos[tri_a]
                t2 = pos[tri_b]
                t3 = pos[tri_c]
                delta_t1 = displacement_in[tri_a]
                delta_t2 = displacement_in[tri_b]
                delta_t3 = displacement_in[tri_c]

                valid_plane, n, d, separation_eps = create_vertex_triangle_division_plane_closest_pt(
                    colliding_particle_pos,
                    colliding_particle_displacement,
                    t1,
                    delta_t1,
                    t2,
                    delta_t2,
                    t3,
                    delta_t3,
                )

                # Same policy as the edge-edge pass: a zero-gap or intersecting pair has no
                # strict separator and is left unconstrained for this pass.
                if valid_plane:
                    t = planar_truncation_t(
                        colliding_particle_pos, colliding_particle_displacement, n, d, gamma, separation_eps
                    )
                    if t < 1.0:
                        wp.atomic_min(truncation_t_out, particle_idx, t)
                    t = planar_truncation_t(t1, delta_t1, -n, d, gamma, separation_eps)
                    if t < 1.0:
                        wp.atomic_min(truncation_t_out, tri_a, t)
                    t = planar_truncation_t(t2, delta_t2, -n, d, gamma, separation_eps)
                    if t < 1.0:
                        wp.atomic_min(truncation_t_out, tri_b, t)
                    t = planar_truncation_t(t3, delta_t3, -n, d, gamma, separation_eps)
                    if t < 1.0:
                        wp.atomic_min(truncation_t_out, tri_c, t)

            collision_buffer_counter += 32

    # Don't forget to do the final truncation based on the maximum displacement allowance!
