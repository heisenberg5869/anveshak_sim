def handle_loop_closure(current_pose_id, current_features, pose_graph):
    """
    Inputs:
    - current_pose_id: Index of the newest node in the graph
    - current_features: Visual descriptors or LiDAR scan of the current location
    - pose_graph: The data structure containing all Poses (nodes) and Constraints (edges)
    """

    # Detecting Loop-Closure Candidates
    # We don't compare against every single past frame. Use spatial KD-Tree to find likely matches.
    candidates = feature_database.query(current_features, max_results=5)
    
    for candidate_id in candidates:
        # Skip frames that are too close in time (we already have odometry for those)
        if abs(current_pose_id - candidate_id) < TEMPORAL_THRESHOLD:
            continue
            
        # Validating a Match
        success, relative_transform, confidence = geometric_registration(pose_graph.get_data(candidate_id), current_features)

        if success and confidence > ACCEPTANCE_THRESHOLD:
            
            # Adding a Constraint
            new_constraint = Constraint(from_id = current_pose_id, to_id = candidate_id, transform = relative_transform, information_matrix = compute_info_matrix(confidence))
            pose_graph.add_edge(new_constraint)
            
            # Performing Global Optimization
            optimizer = LevenbergMarquardtOptimizer(pose_graph)
            
            optimized_poses = optimizer.solve(max_iterations=10)
            
            # Update the global map with corrected positions
            pose_graph.update_all_poses(optimized_poses)
            
            # Stop after finding one solid match to prevent over-optimization
            break