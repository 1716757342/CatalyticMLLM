# import torch


# def init_edge_rot_mat(edge_distance_vec):
#     edge_vec_0 = edge_distance_vec
#     edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

#     # Make sure the atoms are far enough apart
#     #assert torch.min(edge_vec_0_distance) < 0.0001
#     if torch.min(edge_vec_0_distance) < 0.0001:
#         print(
#             "Error edge_vec_0_distance: {}".format(
#                 torch.min(edge_vec_0_distance)
#             )
#         )
        
#     norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

#     edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
#     edge_vec_2 = edge_vec_2 / (
#         torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
#     )
#     # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
#     # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
#     edge_vec_2b = edge_vec_2.clone()
#     edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
#     edge_vec_2b[:, 1] = edge_vec_2[:, 0]
#     edge_vec_2c = edge_vec_2.clone()
#     edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
#     edge_vec_2c[:, 2] = edge_vec_2[:, 1]
#     vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
#         -1, 1
#     )
#     vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
#         -1, 1
#     )

#     vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
#     edge_vec_2 = torch.where(
#         torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
#     )
#     vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
#     edge_vec_2 = torch.where(
#         torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
#     )

#     vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
#     # Check the vectors aren't aligned
#     assert torch.max(vec_dot) < 0.99

#     norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
#     norm_z = norm_z / (
#         torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
#     )
#     norm_z = norm_z / (
#         torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
#     )
#     norm_y = torch.cross(norm_x, norm_z, dim=1)
#     norm_y = norm_y / (
#         torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
#     )

#     # Construct the 3D rotation matrix
#     norm_x = norm_x.view(-1, 3, 1)
#     norm_y = -norm_y.view(-1, 3, 1)
#     norm_z = norm_z.view(-1, 3, 1)

#     edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
#     edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

#     return edge_rot_mat.detach()
# File: Qwen2.5-VL-main/equiformer_v2_all/nets/equiformer_v2/edge_rot_mat.py

import torch

def init_edge_rot_mat(edge_distance_vec: torch.Tensor) -> torch.Tensor:
    """
    Calculates the rotation matrix that aligns the z-axis with the given edge vector.

    This implementation uses Rodrigues' rotation formula for numerical stability and
    correctly handles the edge cases where the edge vector is collinear with the z-axis.

    Args:
        edge_distance_vec (torch.Tensor): A tensor of shape (num_edges, 3) representing
                                          the vectors connecting pairs of atoms.

    Returns:
        torch.Tensor: A tensor of shape (num_edges, 3, 3) representing the rotation
                      matrices.
    """
    # 1. Normalize the input edge vectors to get unit vectors.
    # Add a small epsilon to the denominator to prevent division by zero for zero-length vectors.
    epsilon = 1e-10
    edge_vec_norm = torch.linalg.norm(edge_distance_vec, dim=1, keepdim=True)
    edge_vec_unit = edge_distance_vec / (edge_vec_norm + epsilon)

    # 2. Define the canonical vector to be rotated (the z-axis).
    z_axis = torch.zeros_like(edge_vec_unit)
    z_axis[:, 2] = 1.0

    # 3. Calculate the rotation axis (k) and the cosine of the rotation angle (c).
    # The rotation axis is the cross product of the two vectors.
    k = torch.cross(z_axis, edge_vec_unit, dim=1)
    k_norm = torch.linalg.norm(k, dim=1, keepdim=True)
    
    # The cosine of the angle is the dot product.
    c = torch.sum(z_axis * edge_vec_unit, dim=1)

    # 4. Handle the edge cases of collinear vectors.
    # Create a mask for edges that are nearly parallel or anti-parallel to the z-axis.
    collinear_mask = (k_norm < epsilon).squeeze()

    # Create the skew-symmetric matrix (K) from the rotation axis vector.
    K = torch.zeros(edge_vec_unit.shape[0], 3, 3, device=edge_vec_unit.device, dtype=edge_vec_unit.dtype)
    K[:, 0, 1] = -k[:, 2]
    K[:, 1, 0] = k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 2, 0] = -k[:, 1]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 1] = k[:, 0]

    # 5. Use Rodrigues' rotation formula for the general case (non-collinear vectors).
    # R = I + K + K^2 * (1 - c) / (s^2), where s = ||k|| = sin(theta)
    I = torch.eye(3, device=edge_vec_unit.device, dtype=edge_vec_unit.dtype).expand(edge_vec_unit.shape[0], -1, -1)
    
    # =======================================================
    # === Final core fix: ensure the broadcast shapes are compatible ===
    # =======================================================
    # Squeeze k_norm to 1D, calculate the factor also in 1D.
    factor_1d = (1.0 - c) / (k_norm.squeeze(-1)**2 + epsilon)
    # Reshape the 1D factor to (num_edges, 1, 1) for broadcasting.
    factor = factor_1d.view(-1, 1, 1)
    
    rot_mat = I + K + torch.bmm(K, K) * factor

    # 6. Apply corrections for the special cases (collinear vectors).
    if torch.any(collinear_mask):
        # For anti-parallel vectors (c ≈ -1), the rotation is 180 degrees around any
        # perpendicular axis. We choose a rotation around the y-axis for consistency.
        rot_180 = torch.eye(3, device=edge_vec_unit.device, dtype=edge_vec_unit.dtype)
        rot_180[0, 0] = -1
        rot_180[2, 2] = -1
        
        # Use the identity matrix for parallel vectors (c ≈ 1) and the 180-degree
        # rotation matrix for anti-parallel vectors.
        # `c < 0` distinguishes between the anti-parallel and parallel cases.
        rot_mat[collinear_mask] = torch.where(
            c[collinear_mask].view(-1, 1, 1) < 0,
            rot_180,
            I[0]
        )
    
    # Unlike the original implementation, we DO NOT detach the rotation matrix.
    # The gradient needs to flow through this operation back to the coordinates.
    return rot_mat