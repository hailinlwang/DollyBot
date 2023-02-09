#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag

#Map Handling Functions
def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np

def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    def check_if_duplicate(self, point):
        
        # Assuming that a node's ID is its index in the node list
        for node in self.nodes:
            if point[0] == node.point[0] and point[1] == node.point[1] :
                return True

        return False

    #Functions required for RRT
    def sample_map_space(self):
        # Return an [x,y] coordinate to drive the robot towards
        # HINT: You may choose a subset of the map space to sample in

        # Continue to take sample points until we find one that isn't duplicate
        while True:

            print(self.bounds)

            # Get bounding range for map
            x, y, theta = self.map_settings_dict['origin']

            # Generate random values for x and y
            point = np.random.rand(2, 1)
            point[0, 0] = (np.abs(x)-x)*point[0, 0] + x
            point[1, 0] = (np.abs(y)-y)*point[1, 0] + y

            if not self.check_if_duplicate(point):
                break

        return point
    
    def closest_node(self, point):

        closest_dist = 9999
        closest_ind = 0

        # Assuming that a node's ID is its index in the node list
        for i, node in enumerate(self.nodes):

            # Get distant from current point to node point
            dist = np.sqrt((point[0, 0] - node.point[0])**2 + (point[1, 0] - node.point[1])**2)

            # Update if necessary
            if dist < closest_dist:
                closest_dist = dist
                closest_ind = i

        return closest_ind 
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]

        # Get trajectory for first timestep
        vel, rot_vel = self.robot_controller(node_i, point_s)
        robot_traj = self.trajectory_rollout(vel, rot_vel)

        # Multiply robot_traj by rotation matrix to go to world frame, then add to node_i
        theta = - node_i[2]
        rotm = np.eye((3))
        rotm[0, :] = [np.cos(theta), -np.sin(theta), 0]
        rotm[1, :] = [np.sin(theta), np.cos(theta), 0]

        robot_traj = node_i + rotm@robot_traj

        # Iteratively call robot_controller and trajectory_rollout until at point_s
        threshold = 0.01
        tog = 0
        cnt = 0
        
        while (tog == 0) and (cnt < 50) :

            # Get new trajectory in robot frame
            new_vel, new_rot_vel = self.robot_controller(robot_traj[:, -1], point_s)
            new_pts = self.trajectory_rollout(new_vel, new_rot_vel)

            # Transform new pts to world frame
            theta = - robot_traj[2, -1]
            rotm = np.eye((3))
            rotm[0, :] = [np.cos(theta), -np.sin(theta), 0]
            rotm[1, :] = [np.sin(theta), np.cos(theta), 0]
            new_pts_wf = rotm@new_pts + np.expand_dims(robot_traj[:, -1], axis=1)

            # Collision check new_pt
            rr, cc = self.points_to_robot_circle(new_pts_wf)

            col = self.check_collision(rr, cc)

            if col:
                return True, []
                
            # If not occupied add new pts to path
            robot_traj = np.hstack((robot_traj, new_pts_wf))

            # Check if new point is close enough to point_s
            if np.linalg.norm(robot_traj[1:2, robot_traj.shape[1] - 1]) < threshold:
                tog = 1
            
            cnt += 1
        
        return False, robot_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        p_lin = 0.1
        p_ang = 0.1

        dx = point_s[0] - node_i[0]
        dy = point_s[1] - node_i[1]

        theta_d = np.arctan2(dy,dx)
        theta_i = node_i[2]

        while theta_d < 0:
            theta_d += 2*np.pi
        while theta_i < 0:
            theta_i += 2*np.pi

        dtheta = theta_d - node_i[2]
        if dtheta > np.pi:
            dtheta -= 2*np.pi

        lin_vel = p_lin * np.sqrt(dx**2 + dy**2)

        ang_vel = p_ang * dtheta

        return lin_vel, ang_vel
    
    def trajectory_rollout(self,vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        num_substeps = 10
        timestep = 1
        
        # Initialize empty trajectory
        traj = np.zeros((3, num_substeps))
        vel_vec = np.zeros((3, 1))
        vel_vec[0, 0] = vel
        vel_vec[2, 0] = rot_vel
        theta = 0
        # vel_vec = np.array([[vel, 0, rot_vel]]).T

        # Generate time stamps of sub steps
        substep_size = timestep / num_substeps
        rotm = np.eye((3))

        # Run through each substep
        for i in range(1, num_substeps):
            # Build rotation matrix

            rotm[0, :] = [np.cos(theta), -np.sin(theta), 0]
            rotm[1, :] = [np.sin(theta), np.cos(theta), 0]
            
            traj[:, i] = traj[:, i-1] + np.reshape(rotm@vel_vec*substep_size, (3,))
            # print(vel_vec)
            theta = traj[2, i]
            vel_vec[0,0] = vel * np.cos(theta)
            vel_vec[1,0] = vel * np.sin(theta)

        return traj
    
    def point_to_cell(self, point):
        
        # Get origin values (coordinates of left bottom corner in m)
        x_origin, y_origin, theta_origin = self.map_settings_dict['origin']
        resolution = self.map_settings_dict['resolution']

        # TESTING
        # point = np.array([[np.abs(x_origin)], [np.abs(y_origin)]])

        # Instantiate conversion array
        points = np.zeros_like(point)

        # Convert metric points to be measured from upper left corner
        x = point[0, :] + abs(x_origin)
        y = -1*(point[1, :] - abs(y_origin))

        # Scale points
        y_map, x_map = self.map_shape

        points[0, :] = (x / (abs(x_origin)*2)) * (x_map - 1)
        points[1, :] = (y / (abs(y_origin)*2)) * (y_map - 1)

        # points[0, :] = x / resolution
        # points[1, :] = y / resolution

        points = points.astype(int)

        # TESTING
        # print(points)
        # data = mpimg.imread("../maps/willowgarageworld_05res.png")
        # plt.plot(points[0,0], points[1,0], 'o', color="green")
        # plt.imshow(data)
        # plt.show()

        return points

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function

        robot_locations_rr = list()
        robot_locations_cc = list()

        for p in range(points.shape[1]):

            pixel = self.point_to_cell(np.expand_dims(points[:, p], axis=1))

            rr, cc = disk((pixel[0, 0], pixel[1, 0]), self.robot_radius)
            robot_locations_rr.append(rr)
            robot_locations_cc.append(cc)

        return robot_locations_rr, robot_locations_cc
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    def check_collision(self, rr, cc):

        if 0 in self.occupancy_map[rr, cc]:
            return True

        return False

    #Planner Functions
    def rrt_planning(self):

        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        while True: 

            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            # Simulate driving the robot towards the point from the closest node
            # When trajectory_o is returned, assume in INERTIAL FRAME {I}
            coll, trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            # If at any point along the trajectory is a collision, cannot add point as node
            if not coll:
                curr_cost = self.nodes[closest_node_id].cost + self.num_substeps
                self.nodes.append(Node(trajectory_o[:, -1].reshape((3,1)), closest_node_id, curr_cost))
                self.nodes[closest_node_id].children_ids.append(len(self.nodes) - 1)

                # Check to see if we're within
                dist = np.sqrt((trajectory_o[0, -1] - self.goal_point[0, 0])**2 + (trajectory_o[1, -1] - self.goal_point[1, 0])**2)

                if dist < self.stopping_dist:
                    break

            print(len(self.nodes))

            if len(self.nodes) > 150:
                data = mpimg.imread("../maps/willowgarageworld_05res.png")
                pts = self.nodes[0].point
                for i in range(1, len(self.nodes)):
                    pts = np.hstack((pts, np.expand_dims(self.nodes[i].point, axis=1)))

                print(pts[0, :], pts[1, :])
                plt.plot(pts[0, :], pts[1, :], 'o', color="green")
                plt.imshow(data)
                plt.show()

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    nodes = path_planner.rrt_planning()

    # nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    print(node_path_metric)

    # #Leftover test functions
    # np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
