#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Float32MultiArray
import math
from utils import *
import torch
#from your_msgs.msg import PedestrianArray  # Replace with your actual pedestrian message type

class GnnSocialForcePlanner:
    def __init__(self):
        rospy.init_node('social_force_planner')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model("gnn_model.pt")
        self.model.eval()
        self.robot_features = np.array([0.,0.,0.,0.,1.])

        self.robot_pos = np.array([0.0, 0.0])
        self.robot_vel = np.array([0.0, 0.0])
        self.pedestrians = []
        self.pedestrians2 = []

        rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_cb)
        rospy.Subscriber("/cmd_vel", Twist, self.robot_vel_cb)
        rospy.Subscriber("/dynamic_object2", Float32MultiArray, self.pedestrians_cb) # /dynamic_object for lidar only; /dynamic_object2 for sensor fusion
        rospy.Subscriber("/dynamic_object2", Float32MultiArray, self.pedestrians2_cb) # for ground truth
        rospy.Subscriber("/move_base_flex/global_planner/GlobalPlanner/plan", Path, self.path_cb) # λ = 0.5
        #rospy.Subscriber("/move_base_flex/dwa_local_planner/DWAPlannerROS/global_plan", Path, self.path_cb) # λ = 2.0

        self.path_pub = rospy.Publisher("/social_force_path", Path, queue_size=1)
        self.marker_pub = rospy.Publisher("/social_force_marker", Marker, queue_size=1)
        rospy.Timer(rospy.Duration(0.5), lambda event: self.publish_marker(2.5,0.0))

        self.msg_cnt = 0.0
        self.msg_num = 3600 #one people: 2400/800 # two people: 3600/450
        self.robot_velocity = 0.4
        self.collision_radius = 1.0

        self.path_length_dwa_total = 0.0
        self.path_length_dwa_cnt = 0.0
        self.path_length_social_total = 0.0
        self.path_length_social_cnt = 0.0

        self.path_irregularity_dwa_total = 0.0
        self.path_irregularity_dwa_cnt = 0.0
        self.path_irregularity_social_total = 0.0
        self.path_irregularity_social_cnt = 0.0

        self.closest_distance_dwa_total = 0.0
        self.closest_distance_dwa_cnt = 0.0
        self.closest_distance_social_total = 0.0
        self.closest_distance_social_cnt = 0.0

        self.min_ttc_dwa_total = 0.0
        self.min_ttc_dwa_cnt = 0.0
        self.min_ttc_social_total = 0.0
        self.min_ttc_social_cnt = 0.0

    def robot_pose_cb(self, msg):
        self.robot_pos = np.array([msg.pose.position.x, msg.pose.position.y])

    def robot_vel_cb(self, msg):
        self.robot_vel = np.array([msg.linear.x, msg.linear.y])

    def pedestrians_cb(self, msg):
        self.pedestrians = []
        
        rows = msg.layout.dim[0].size if len(msg.layout.dim) > 0 else 0
        cols = msg.layout.dim[1].size if len(msg.layout.dim) > 1 else 0
        for i in range(rows):
            row = msg.data[i * cols : (i + 1) * cols]
            self.pedestrians.append(row)
        #rospy.loginfo("Received matrix (%d x %d):", rows, cols)
        #for r in self.pedestrians:
        #    rospy.loginfo(str(r))
    
    def pedestrians2_cb(self, msg):
        self.pedestrians2 = []
        
        rows = msg.layout.dim[0].size if len(msg.layout.dim) > 0 else 0
        cols = msg.layout.dim[1].size if len(msg.layout.dim) > 1 else 0
        for i in range(rows):
            row = msg.data[i * cols : (i + 1) * cols]
            self.pedestrians2.append(row)

    def path_cb(self, msg):
        new_path = Path()
        new_path.header = msg.header

        pose_num = len(msg.poses)
        for i in range(pose_num):
            pose_curr = msg.poses[i]
            pos_curr = np.array([pose_curr.pose.position.x, pose_curr.pose.position.y])
            pose_next = msg.poses[i+1] if i < (pose_num-1) else msg.poses[i]
            pos_next = np.array([pose_next.pose.position.x, pose_next.pose.position.y])
            pos_delta = pos_next-pos_curr
            #print(pos_next,pos_curr,i,pose_num)
            vel = self.robot_velocity*pos_delta/np.linalg.norm(pos_delta) if (i < (pose_num-1) and np.linalg.norm(pos_delta)!=0.) else np.array([0., 0.])
            force = self.predict_offset(pos_curr,vel)
            #force = self.compute_social_force(pos)
            offset_pos = pos_curr + force

            new_pose = PoseStamped()
            new_pose.header = pose_curr.header
            new_pose.pose.position.x = offset_pos[0]
            new_pose.pose.position.y = offset_pos[1]
            new_pose.pose.position.z = pose_curr.pose.position.z
            new_pose.pose.orientation = pose_curr.pose.orientation
            new_path.poses.append(new_pose)



        self.path_pub.publish(new_path)

        path_length_dwa = self.calculate_path_length(msg.poses)
        if path_length_dwa!=0.0:
            self.path_length_dwa_total+=path_length_dwa
            self.path_length_dwa_cnt+=1
        path_length_social = self.calculate_path_length(new_path.poses)
        if path_length_social!=0.0:
            self.path_length_social_total+=path_length_social
            self.path_length_social_cnt+=1
        #print("path irregularity dwa: ", path_length_dwa, "path irregularity social: ", path_length_social)
        
        path_irregularity_dwa = self.calculate_path_irregularity(msg.poses)
        if path_irregularity_dwa!=math.inf:
            self.path_irregularity_dwa_total+=path_irregularity_dwa
            self.path_irregularity_dwa_cnt+=1
        path_irregularity_social = self.calculate_path_irregularity(new_path.poses)
        if path_irregularity_social!=math.inf:
            self.path_irregularity_social_total+=path_irregularity_social
            self.path_irregularity_social_cnt+=1
        #print("path irregularity dwa: ", path_irregularity_dwa, "path irregularity social: ", path_irregularity_social)

        closest_distance_dwa = self.calculate_closest_distance(msg.poses)
        if closest_distance_dwa!=math.inf:
            self.closest_distance_dwa_total+=closest_distance_dwa
            self.closest_distance_dwa_cnt+=1
        closest_distance_social = self.calculate_closest_distance(new_path.poses)
        if closest_distance_social!=math.inf:
            self.closest_distance_social_total+=closest_distance_social
            self.closest_distance_social_cnt+=1
        #print(closest_distance_dwa, closest_distance_social)
        
        min_ttc_dwa = self.calculate_min_ttc(msg.poses,self.robot_velocity,self.collision_radius)
        if min_ttc_dwa!=math.inf:
            self.min_ttc_dwa_total+=min_ttc_dwa
            self.min_ttc_dwa_cnt+=1
        min_ttc_social = self.calculate_min_ttc(new_path.poses,self.robot_velocity,self.collision_radius)
        if min_ttc_social!=math.inf:
            self.min_ttc_social_total+=min_ttc_social
            self.min_ttc_social_cnt+=1
        #print(min_ttc_dwa, min_ttc_social)

        

        self.msg_cnt=self.msg_cnt+1
        #print(self.msg_cnt)
        if self.msg_cnt==self.msg_num:
            print("path_length_dwa: ", self.path_length_dwa_total/self.path_length_dwa_cnt)
            print("path_length_social: ", self.path_length_social_total/self.path_length_social_cnt)
            print("path_irregularity_dwa: ", self.path_irregularity_dwa_total/self.path_irregularity_dwa_cnt)
            print("path_irregularity_social: ", self.path_irregularity_social_total/self.path_irregularity_social_cnt)
            print("closest_distance_dwa: ", self.closest_distance_dwa_total/self.closest_distance_dwa_cnt)
            print("closest_distance_social: ", self.closest_distance_social_total/self.closest_distance_social_cnt)
            if self.min_ttc_dwa_cnt !=0:
                print("min_ttc_dwa: ", self.min_ttc_dwa_total/self.min_ttc_dwa_cnt, " count: ", self.min_ttc_dwa_cnt)
            else:
                print("min_ttc_dwa: ", math.inf, " count: ", self.min_ttc_dwa_cnt)
            if self.min_ttc_social_cnt !=0:
                print("min_ttc_social: ", self.min_ttc_social_total/self.min_ttc_social_cnt, " count: ", self.min_ttc_social_cnt)
            else:
                print("min_ttc_social: ", math.inf, " count: ", self.min_ttc_social_cnt)
    
    def publish_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"   # Or "odom" depending on your global planner frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "social_force"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3   # diameter of the sphere
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # red, fully opaque
        self.marker_pub.publish(marker)

    def compute_social_force(self, robot_pos):
        total_force = np.zeros(2)
        for ped in self.pedestrians:
            #print(ped)
            ped_pos = np.array([ped[0], ped[1]])
            ped_vel = np.array([ped[2], ped[3]])
            diff = robot_pos - (ped_pos + ped_vel)
            dist = np.linalg.norm(diff)
            if dist < 0.001:
                continue
            direction = diff / dist
            force_mag = 1.0 * np.exp(-dist / 0.5)  # λ = 0.5 for global_planner, 2.0 for dwa_global
            total_force += force_mag * direction
        return total_force
    
    def predict_offset(self, pos, vel):
        self.robot_features[:2] = pos.copy()
        self.robot_features[2:4] = vel.copy()
        input = self.robot_features.reshape(1, -1)
        for ped in self.pedestrians:
            ped_features = np.array([ped[0],ped[1],ped[2],ped[3],0.])
            #print(input)
            #print(ped_features)
            input = np.vstack((input, ped_features))
        x_node = preprocess_frame_to_node_features(input)
        edge_index, edge_attr = build_bidirectional_star(x_node)
        data = Data(
            x=torch.from_numpy(x_node),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            )
        with torch.no_grad():
            y_hat = self.model(data.to(self.device)).cpu().numpy().reshape(-1)
        return y_hat
    
    def calculate_path_length(self, poses):
        n = len(poses)
        if n<2:
            return 0.0
        total = 0.0
        for i in range(1, n):
            p0 = poses[i-1].pose.position
            p1 = poses[i].pose.position
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            total += math.hypot(dx, dy)
        return total
    
    def calculate_path_irregularity(self, poses, eps_len: float = 1e-9):
        n = len(poses)
        if n < 2:
            return 0.0

        # Extract (x, y)
        pts = []
        for ps in poses:
            p = ps.pose.position
            pts.append((float(p.x), float(p.y)))

        # Chord (start -> end)
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        chord = math.hypot(x1 - x0, y1 - y0)

        # Sum absolute heading changes over non-degenerate segments
        total_abs_turn = 0.0
        prev_h = None
        for i in range(1, n):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            if math.hypot(dx, dy) < eps_len:
                continue  # skip zero-length segments
            h = math.atan2(dy, dx)
            if prev_h is not None:
                delta_h = h - prev_h
                # change angle
                while delta_h <= -math.pi:
                    delta_h += 2 * math.pi
                while delta_h > math.pi:
                    delta_h -= 2 * math.pi
                total_abs_turn += abs(delta_h)
            prev_h = h
        irregularity = (total_abs_turn / chord) if chord >= eps_len else 0.0
        return irregularity

    def calculate_closest_distance(self, poses, eps_len: float = 1e-9):
        if not self.pedestrians2 or len(poses) < 2:
            return math.inf

        # Extract 2D points from the path
        pts = []
        for ps in poses:
            p = ps.pose.position
            pts.append((float(p.x), float(p.y)))

        # Precompute valid segment midpoints
        midpoints = []  # (mx, my)
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            if math.hypot(x1 - x0, y1 - y0) < eps_len:
                continue  # skip degenerate segment
            mx = 0.5 * (x0 + x1)
            my = 0.5 * (y0 + y1)
            midpoints.append((mx, my))

        if not midpoints:
            return math.inf

        best_dist = math.inf
        best_ped_xy = (None, None)
        best_mid_xy = (None, None)

        # Scan pedestrians vs. segment midpoints
        for ped in self.pedestrians2:
            px, py = float(ped[0]), float(ped[1])
            for mx, my in midpoints:
                d = math.hypot(px - mx, py - my)
                if d < best_dist:
                    best_dist = d
                    best_ped_xy = (px, py)
                    best_mid_xy = (mx, my)

        return best_dist

    def calculate_min_ttc(self, poses, v_robot, collision_radius, eps_len: float = 1e-9):
        # Basic checks
        if not poses or len(poses) < 2 or v_robot <= 0.0 or not self.pedestrians2:
            return math.inf

        # Extract 2D points
        pts = []
        for ps in poses:
            p = ps.pose.position
            pts.append((float(p.x), float(p.y)))

        # Build segment midpoints and arc-length to each midpoint
        midpoints = []  # (seg_idx, mx, my, s_to_mid)
        s_cursor = 0.0
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy)
            if L < eps_len:
                continue  # skip degenerate segment
            mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            s_to_mid = s_cursor + 0.5 * L
            midpoints.append((mx, my, s_to_mid))
            s_cursor += L

        if not midpoints:
            return math.inf

        best_ttc = math.inf
        best_xy = (None, None)
        min_gap = math.inf

        # Scan each segment midpoint at its arrival time
        for mx, my, s_to_mid in midpoints:
            t_mid = s_to_mid / v_robot  # robot arrival time at this midpoint
            # Evaluate all pedestrians at t_mid
            for ped in self.pedestrians2:
                px, py = float(ped[0]), float(ped[1])
                vx, vy = float(ped[2]), float(ped[3])
                ptx = px + vx * t_mid
                pty = py + vy * t_mid
                d = math.hypot(ptx - mx, pty - my)
                if d < min_gap:
                    min_gap = d
                if d <= collision_radius and t_mid < best_ttc:
                    best_ttc = t_mid
                    best_xy = (mx, my)

        return best_ttc

if __name__ == '__main__':
    try:
        planner = GnnSocialForcePlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
