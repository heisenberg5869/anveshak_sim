def update_grid_map(grid_map, lidar_scan, rover_pose, rover_velocity):

    # Motion gating
    if abs(rover_velocity.angular_z) > 0.5:
        return grid_map

    for point in lidar_scan:
        world_pt = transform_to_world(point, rover_pose)
        distance = calculate_distance(rover_pose.position, world_pt)

        # Distance based confidence
        confidence = max(0.1, 1.0 - distance / max_range)

        # Wall preservation
        for cell in line(rover_pose.position, world_pt)[:-1]:
            update = confidence * 0.3

            if grid_map[cell].occupancy > 0.6:  # wall
                current_angle = atan2(cell.y - rover_pose.y, cell.x - rover_pose.x)
                angle_diff = abs(current_angle - grid_map[cell].detection_angle)
                update *= 1.0 if angle_diff > 30 else 0.1

            grid_map[cell].occupancy -= update
            grid_map[cell].occupancy = clamp(grid_map[cell].occupancy, 0.0, 1.0)

        # Mark hit point as occupied
        hit_cell = world_to_grid(world_pt)
        grid_map[hit_cell].occupancy += confidence * 0.7
        grid_map[hit_cell].occupancy = clamp(grid_map[hit_cell].occupancy, 0.0, 1.0)

    return grid_map