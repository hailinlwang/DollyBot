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

    #Functions required for RRT
    def sample_map_space(self):
        # Return an [x,y] coordinate to drive the robot towards
        # HINT: You may choose a subset of the map space to sample in

        # Continue to take sample points until we find one that isn't duplicate
        while True:

            # Get bounding range for map
            bounding_x, bounding_y = self.bounds[0, :], self.bounds[1, :]

            # Generate random values for x and y
            point = np.random.rand(2, 1)
            point[1, 0] = (bounding_x[1]-bounding_x[0])*point[1, 0] + bounding_x[0]
            point[0, 0] = (bounding_y[1]-bounding_y[0])*point[0, 0] + bounding_y[0]

            if !check_if_duplicate(point)
                break

        return point
    
    def check_if_duplicate(self, point):
        
        # Assuming that a node's ID is its index in the node list
        for node in self.nodes:
            if point == node.point[0:2, 0]:
                return True

        return False
    
    def closest_node(self, point):

        closest_dist = 9999
        closest_ind = 0

        # Assuming that a node's ID is its index in the node list
        for i, node in enumerate(self.nodes):

            # Get distant from current point to node point
            dist = np.sqrt((point[0, 0] - node.point[0, 0])**2 + (point[1, 0] - node.point[1, 0])**2)

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
        print("TO DO: Implment a method to simulate a trajectory given a sampled point")

        vel, rot_vel = self.robot_controller(node_i, point_s)

        robot_traj = self.trajectory_rollout(vel, rot_vel)

        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        print("TO DO: Implement a control scheme to drive you towards the sampled point")
        return 0, 0
    
    def trajectory_rollout(self, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        print("TO DO: Implement a way to rollout the controls chosen")

        # Initialize empty trajectory
        traj = np.zeros((3, self.num_substeps))

        # Generate time stamps of sub steps
        substep_size = self.timestep / self.num_substeps

        # Run through each substep
        for i in range(2, self.num_substeps-1):
            # Get current q and p
            q = traj[:, i-1]
            theta = q[2, 0]
            p = np.array([[vel], [rot_vel]])
            
            # Get q_dot
            G_q = np.zeros((3, 2))
            G_q[:, 0] = [np.cos(theta), np.sin(theta), 0]
            G_q[:, 0] = [0, 0, 1]
            q_dot = G_q@p
            
            # Get next step
            traj[:, i] = traj[:, i-1] + q_dot*substep_size

        return traj
    
    def point_to_cell(selfs, point):

        # Get origin values (coordinates of left bottom corner in m)
        x_origin, y_origin, theta_origin = self.map_settings_dict['origin']
        resolution = self.map_settings_dict['resolution']

        # Instantiate conversion array
        points = np.zeros_like(point)

        # Convert metric points to be measured from upper left corner
        x = point[0, :] + abs(x_origin)
        y = -1*point[1, :] + abs(y_origin)

        # Scale points
        # y_map, x_map = self.map_shape

        # points[0, :] = (y / (abs(y_origin)*2)) * y_map
        # points[1, :] = (x / (abs(x_origin)*2)) * x_map

        points[0, :] = y / resolution
        points[1, :] = x / resolution

        return points

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        print("TO DO: Implement a method to get the pixel locations of the robot path")
        return [], []
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

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        while True: 
            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            # Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            # If at any point along the trajectory is a collision, cannot add point to node
            collision = False

            for i in trajectory_o.shape[1]:

                waypt = trajectory_o.shape[:, i]

                waypt_ind = point_to_cell(waypt[0:2, 0])

                if self.occupancy_map[waypt_ind]:
                    collision = True
                    break

            if !collision:
                # Question: what is first index of trajectory_o
                curr_cost = self.nodes[closest_node_id].cost + self.num_substeps
                self.nodes.append(Node(trajectory_o[:, 0], closest_node_id, curr_cost))
                self.nodes[closest_node_id].children_ids.append(len(self.nodes) - 1)

            #Check if goal has been reached
            if point == self.goal_point:
                break

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

    path_planner.point_to_cell(goal_point)

    # nodes = path_planner.rrt_star_planning()
    # node_path_metric = np.hstack(path_planner.recover_path())

    # #Leftover test functions
    # np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
