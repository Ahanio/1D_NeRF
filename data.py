import torch
import numpy as np


def unisurf_data_gen(x_range, objects, num_rays, num_samples_per_ray, step_size=0.01):
    x = torch.linspace(x_range[0], x_range[1], num_rays * 10).float().unsqueeze(-1)

    # pick random points that lay outside of objects
    x = x[torch.randperm(x.shape[0])]
    for ob in objects:
        x = x[torch.logical_not(torch.logical_and(x > ob[0], x < ob[1]))]
    x = x[:num_rays]
    origins = x  # [n,]
    origins[0] = 0.5  # 0.2  ## For visualization purposes

    directions = torch.Tensor(np.random.choice([-1, 1], num_rays)).float()  # [n,]
    directions[0] = 1  ## For visualization purposes

    # sample points along the ray. The number of points is num_samples_per_ray
    # step size between the points is step_size
    ray_points = origins.unsqueeze(-1) + directions.unsqueeze(-1) * (
        torch.arange(0, step_size * num_samples_per_ray, step_size)[
            :num_samples_per_ray
        ]
    )

    depths = find_gt_depth_per_ray(x_range, objects, origins, directions)

    depths = torch.Tensor(depths)
    # return origins, directions, depths
    gt_boundary = origins + directions * depths
    delta = torch.abs(torch.diff(ray_points, dim=1, append=ray_points[:, -2:-1]))
    assert len(delta[delta < 0]) == 0
    return origins.unsqueeze(-1), ray_points, delta, depths, gt_boundary


def rays_generator(
    x_range,
    objects,
    num_rays,
    num_samples_per_ray,
    non_uniform=True,
    fix_step_size=False,
    step_size=0.01,
):
    x = torch.linspace(x_range[0], x_range[1], num_rays * 10).float().unsqueeze(-1)

    # pick random points that lay outside of objects
    x = x[torch.randperm(x.shape[0])]
    for ob in objects:
        x = x[torch.logical_not(torch.logical_and(x > ob[0], x < ob[1]))]
    x = x[:num_rays]
    origins = x  # [n,]
    origins[0] = 0.5  # 0.2  ## For visualization purposes

    directions = torch.Tensor(np.random.choice([-1, 1], num_rays)).float()  # [n,]
    directions[0] = 1  ## For visualization purposes

    t_max = torch.zeros_like(origins)
    t_max[directions > 0] = x_range[1] - origins[directions > 0] + 0.5
    t_max[directions < 0] = origins[directions < 0] - x_range[0] + 0.5
    t_min = 0

    ray_points = origins.unsqueeze(-1) + directions.unsqueeze(-1) * (
        torch.linspace(0, 1, num_samples_per_ray) * (t_max - t_min).unsqueeze(-1)
        + t_min
    )
    if non_uniform:
        random_shift = (
            torch.rand(*ray_points.shape)
            * (t_max - t_min)[:, None]
            / ray_points.shape[-1]
        )
        ray_points += random_shift
        ray_points[:, 0] = origins

    if fix_step_size and not non_uniform:
        ray_points = origins.unsqueeze(-1) + directions.unsqueeze(-1) * (
            torch.arange(0, step_size * num_samples_per_ray, step_size)[
                :num_samples_per_ray
            ]
        )

    depths = find_gt_depth_per_ray(x_range, objects, origins, directions)

    depths = torch.Tensor(depths)
    # return origins, directions, depths
    gt_boundary = origins + directions * depths
    delta = torch.abs(torch.diff(ray_points, dim=1, append=ray_points[:, -2:-1]))
    assert len(delta[delta < 0]) == 0
    return origins.unsqueeze(-1), ray_points, delta, depths, gt_boundary


def find_gt_depth_per_ray(x_range, objects, origins, directions):
    # for each ray find the depth along that ray where it intersects with the object.
    # If the ray does not intersect with any object, then the depth is the boundary of the scene

    depths = []
    num_rays = len(origins)
    bounds = torch.Tensor(objects).reshape(-1).sort()[0]
    for i in range(num_rays):
        cur_depth = None
        if directions[i] > 0:
            closest = bounds[bounds > origins[i]]
            if len(closest) != 0:
                cur_depth = closest.min() - origins[i]
            else:
                cur_depth = x_range[1] - origins[i]
        elif directions[i] < 0:
            closest = bounds[bounds < origins[i]]
            if len(closest) != 0:
                cur_depth = origins[i] - closest.max()
            else:
                cur_depth = origins[i] - x_range[0]

        depths.append(cur_depth)
    return depths


def shoot_one_ray(origin, direction, num_samples, x_range, step_size=None):
    t_max = x_range[1] - origin + 0.5 if direction > 0 else origin - x_range[0] + 0.5
    if step_size is None:
        t_min = 0
        ray_points = origin + direction * (
            torch.linspace(0, 1, num_samples) * (t_max - t_min) + t_min
        )
    else:
        num_samples = int(
            t_max / step_size
        )
        ray_points = torch.arange(
            0, step_size * num_samples, step_size
        ) * direction + origin
    delta = torch.abs(torch.diff(ray_points, dim=0, append=ray_points[-2:-1]))
    return ray_points[:, None], delta
