#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.draw import disk
from scipy.linalg import block_diag
import random

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
        print(self.bounds)
        #Robot information
        self.robot_radius = 0.25 #m
        self.vel_max = 0.8 #m/s (Feel free to change!)
        self.rot_vel_max = 1.82 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point.reshape((2,1)) #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 0.5 #s
        self.num_substeps = 10

        #Planning storage
        start_pt = np.array([[0],[-19],[0]])
        # start_pt = np.array([[40.5],[25],[0]])
        print(start_pt.shape)
        self.nodes = [Node(start_pt, -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        # self.window = pygame_utils.PygameWindow(
        #     "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    def check_if_duplicate(self, point):
        
        # Assuming that a node's ID is its index in the node list
        for node in self.nodes:

            if np.linalg.norm(point[:2] - node.point[:2]) < 0.5:
            #if point[0] == node.point[0] and point[1] == node.point[1] :
                return True

        return False

    #Functions required for RRT
    def sample_map_space(self, num_samples):
        # Return an [x,y] coordinate to drive the robot towards
        # HINT: You may choose a subset of the map space to sample in
        bounding_x, bounding_y = self.bounds[0, :], self.bounds[1, :]
        closest_id = self.closest_node(self.goal_point)
        closest_point = self.nodes[closest_id].point
        exp = 0

        if num_samples % 5 == 0 and num_samples != 0:
            closest_point = np.array([[40.5],[25],[0]])

        # if len(self.nodes) != 1:
        if num_samples > 20:
            exp = num_samples*0.05
        #     closest_id = random.randint(round(len(self.nodes)/2), len(self.nodes)-1)
        #     closest_point = self.nodes[closest_id].point

        x_rng = 2 + (exp*0.25)
        y_rng = 2.5 + exp
        xmin = max(closest_point[0] - x_rng,bounding_x[0]) 
        xmax = min(closest_point[0] + x_rng,bounding_x[1]) 
        ymin = max(closest_point[1] - y_rng,bounding_y[0]) 
        ymax = min(closest_point[1] + y_rng,bounding_y[1]) 
    
        ret_point = np.empty((2,1))
        print("Closest point is: ", closest_point)

        while True:
            point = np.random.rand(2, 1)
            ret_point[0, 0] = (xmax-xmin - 2*self.robot_radius)*point[0, 0] + xmin + self.robot_radius
            ret_point[1, 0] = (ymax-ymin - 2*self.robot_radius)*point[1, 0] + ymin + self.robot_radius

            if not self.check_if_duplicate(ret_point):
                break
    # else:

        #     #lower_threshold = 1
        #     upper_threshold = 5
        #     # Continue to take sample points until we find one that isn't duplicate
        #     ret_point = np.empty((2,1))

        #     while True:
                
        #         # Get bounding range for map
        #         bounding_x, bounding_y = self.bounds[0, :], self.bounds[1, :]
        #         # Generate random values for x and y
        #         point = np.random.rand(2, 1)
        #         ret_point[0, 0] = (bounding_x[1]-bounding_x[0] - 2*self.robot_radius)*point[0, 0] + bounding_x[0] + self.robot_radius
        #         ret_point[1, 0] = (bounding_y[1]-bounding_y[0] - 2*self.robot_radius)*point[1, 0] + bounding_y[0] + self.robot_radius

        #         if (np.linalg.norm(self.closest_node(ret_point)-ret_point) <  upper_threshold):
        #             if not self.check_if_duplicate(ret_point):
        #                 break
        
        return ret_point
    
    def closest_node(self, point):

        closest_dist = 9999
        closest_ind = 0

        # Assuming that a node's ID is its index in the node list
        for i, node in enumerate(self.nodes):

            # Get distant from current point to node point
            # dist = np.linalg.norm(point[:2]-node.point[:2])
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
        threshold = 0.50
        tog = 0
        cnt = 0
        
        while (tog == 0) and (cnt < 1000) :

            # Get new trajectory in robot frame
            new_vel, new_rot_vel = self.robot_controller(robot_traj[:, -1], point_s)
            new_pts = self.trajectory_rollout(new_vel, new_rot_vel)

            # Transform new pts to world frame
            theta = robot_traj[2,-1]
            rotm = np.eye((3))
            rotm[0, :] = [np.cos(theta), -np.sin(theta), 0]
            rotm[1, :] = [np.sin(theta), np.cos(theta), 0]
            new_pts_wf = rotm@new_pts + np.expand_dims(robot_traj[:, -1],axis=1)
            robot_traj = np.hstack((robot_traj, new_pts_wf))

            # Collision check new_pt
            rr,cc = self.points_to_robot_circle(new_pts_wf)
            col = self.check_collision(cc,rr)

            if (col == 1):
                return []
            # for pix in circle_pixels:
            #     if self.occupancy_map[pix[0], pix[1]] < 10:
            #         return []
                
            # If not occupied add new pts to path

            # Check if new point is close enough to point_s
            if np.linalg.norm(robot_traj[:2,-1]-point_s[:2].reshape((1,2))) < threshold:
            # if np.linalg.norm(robot_traj[1:2, robot_traj.shape[1]-1]) < threshold:
                tog = 1
            cnt += 1
        #print(cnt)
        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        # This controller determines the velocities that will nominally move the robot from node i to node s
        # Max velocities should be enforced
        # If heading error get it in right direction
        # The move laterally
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

        if dtheta > 0.1:
            ang_vel = p_ang * dtheta
            if ang_vel > self.rot_vel_max:
                ang_vel = self.rot_vel_max
            lin_vel = 0

        else:
            lin_vel = p_lin * np.sqrt(dx**2 + dy**2)
            if lin_vel > self.vel_max:
                lin_vel = self.vel_max

            ang_vel = p_ang * dtheta
            if ang_vel > self.rot_vel_max:
                ang_vel = self.rot_vel_max

        return lin_vel,ang_vel
    
    def trajectory_rollout(self,vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")
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
        # Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        # point is a 2 by N matrix of points of interest
        # input: point (2xN array)        - points of interest expressed in origin (map) reference frame {I}
        # output: map_indices (2xN array) - points converted to indices wrt top left
        
        # convert from map reference frame {I} to bottom-left ref frame {B}
        # position vector: r_B = r_I + r_BI = r_I - r_IB (offset vector from yaml file)

        x_B = point[0] - self.map_settings_dict["origin"][0] 
        y_B = point[1] - self.map_settings_dict["origin"][1]

        # need to convert to index by dividing by resolution (*1/0.05 = *20)
        height = self.map_shape[1]*self.map_settings_dict["resolution"]          # map height in meters
        x_idx = (x_B/self.map_settings_dict["resolution"]).astype(int)
        
        # print("self.map_shape[1]: ", self.map_shape[1])
        # print("Height: ", height)

        y_idx = ((y_B)/self.map_settings_dict["resolution"]).astype(int)  # y_B is wrt bottom left, while y_idx is wrt top left
        map_indices = np.vstack((x_idx,y_idx))

        
        return map_indices
    
    def point_to_cell_multi(self, point):

        x_B = point[0,:] - self.map_settings_dict["origin"][0] 
        y_B = point[1,:] - self.map_settings_dict["origin"][1]

        # need to convert to index by dividing by resolution (*1/0.05 = *20)
        height = self.map_shape[1]*self.map_settings_dict["resolution"]          # map height in meters
        x_idx = (x_B/self.map_settings_dict["resolution"]).astype(int)
        
        # print("self.map_shape[1]: ", self.map_shape[1])
        # print("Height: ", height)

        y_idx = ((y_B)/self.map_settings_dict["resolution"]).astype(int)  # y_B is wrt bottom left, while y_idx is wrt top left
        map_indices = np.vstack((x_idx,y_idx))
        
        return map_indices

    def points_to_robot_circle(self, points):
        # Convert a series of [x,y] points to robot map footprints for collision detection
        # Hint: The disk function is included to help you with this function
        points_idx = self.point_to_cell(points)         # convert to occupancy grid indexes (pixels)
        pixel_radius = self.robot_radius*20         # robot radius in pixels
        footprint = [[],[]]

        for j in range(len(points_idx[0])):
            rr, cc = disk((points_idx[0,j], points_idx[1,j]), pixel_radius, shape=(1600,1600))
            footprint = np.hstack((footprint,np.vstack((rr,cc))))
        
        return footprint
    #Note: If you have corr
    #RRT* specific functions

    def check_collision(self, rr, cc):
        if min(rr) < 1 or max(rr) >= self.occupancy_map.shape[0]:
            return True
        if min(cc) < 1 or max(cc) >= self.occupancy_map.shape[1]:
            return True
        # black = [0,0,0,1]
        black = 0
        if black in self.occupancy_map[rr.astype(int), cc.astype(int)].tolist():
            return True
        return False
    

    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        new_traj = self.simulate_trajectory(node_i, point_f)
        return new_traj
    
    def cost_to_come(self,trajectory):
        cost = 0
        for i in range(1,len(trajectory)):
            cost += np.linalg.norm(trajectory[i,:2] - trajectory[i-1,:2])
        return cost
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        for child_id in self.nodes[node_id].children_ids:
            traj = self.connect_node_to_point(self.nodes[node_id].point,self.nodes[child_id].point[:2])
            cost = self.cost_to_come(traj)
            self.nodes[child_id].cost = self.nodes[node_id].cost + cost
            self.update_children(child_id)
        return

    #Planner Functions
    def rrt_planning(self):
         #This function performs RRT for the given map and robot  
        print("bounds are: ", self.bounds)
        num_iters = 10000
        data = mpimg.imread("../maps/willowgarageworld_05res.png")
        plt.imshow(data)
        i = 0
        tree = np.zeros((2,1))
        # pixels_per_m = s
        while i < num_iters:
        # for i in range(num_iters):
            # Sample
            print(i)

            trajectory_o = []

            num_samples = 0

            while len(trajectory_o) == 0:

                new__sampled_point = self.sample_map_space(num_samples)

                print("Sampled point at", new__sampled_point)

                num_samples += 1

                # Closest Node
                current_parent_id = self.closest_node(new__sampled_point) 
                print(self.nodes[current_parent_id].point)

                # Simulate trajectory
                trajectory_o = self.simulate_trajectory(self.nodes[current_parent_id].point, new__sampled_point[:2])

                print(trajectory_o)
            
            # Assuming that we are getting the final part of the path to add to the graph
            
            if len(trajectory_o) != 0:
                i+=1
                print('Added node', new__sampled_point[0,0], " , ", new__sampled_point[1,0])

                new_point = trajectory_o[:,-1].reshape((3,1))
                pix = self.point_to_cell(new_point)
                footprint = self.points_to_robot_circle(new_point)
                # Plotting point
                plt.scatter(footprint[0], footprint[1], color='r')
                plt.scatter(pix[0], pix[1], marker='o')
                tree = np.hstack((tree, pix))
                tree = np.hstack((tree, footprint))
                if i%20 == 0:
                    plt.scatter(tree[0,:], tree[1,:])
                    plt.imshow(data)
                    plt.show()

                # Add node from trajectory_ o to graph
                path_cost = self.cost_to_come(trajectory_o) + self.nodes[current_parent_id].cost
                new_node = Node(new_point, current_parent_id, path_cost)
                self.nodes.append(new_node)
                
                # Add this node Id to parent node's children
                new_id = len(self.nodes)-1
                self.nodes[current_parent_id].children_ids.append(new_id)
                # print('Initial parent {} for node {}'.format(current_parent_id,new_id))

                # Check if at goal
                # print("new point shape is: ",new_point[:2].shape)
                if np.linalg.norm(new_point[:2] - self.goal_point.reshape((2,1))) < self.stopping_dist:
                    print('At goal!')
                    break

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot  
        num_iters = 10000
        data = mpimg.imread("../maps/willowgarageworld_05res.png")
        i = 0
        # pixels_per_m = s
        while i < num_iters:
        # for i in range(num_iters):
            # Sample
            #print(i)
            trajectory_o = []

            num_samples = 0

            while len(trajectory_o) == 0:

                new_point = self.sample_map_space()
                
                num_samples += 1
                # footprint = self.points_to_robot_circle(new_point)
                # plt.scatter(footprint[0], footprint[1], color='r')
                # plt.scatter(pix[0], pix[1], marker='o')

                # Closest Node
                current_parent_id = self.closest_node(new_point) 

                # Simulate trajectory
                trajectory_o = self.simulate_trajectory(self.nodes[current_parent_id].point, new_point[:2])
            
            # Assuming that we are getting the final part of the path to add to the graph
            
            if len(trajectory_o) != 0:
                i+=1
                print('Added node', new_point)

                new_point = trajectory_o[:,-1].reshape((3,1))
                pix = self.point_to_cell(new_point)
                footprint = self.points_to_robot_circle(new_point)
                plt.scatter(footprint[0], footprint[1], color='r')
                plt.scatter(pix[0], pix[1], marker='o')


                # Add node from trajectory_ o to graph
                path_cost = self.cost_to_come(trajectory_o) + self.nodes[current_parent_id].cost
                new_node = Node(new_point, current_parent_id, path_cost)
                self.nodes.append(new_node)
                

                # Add this node Id to parent node's children
                new_id = len(self.nodes)-1
                self.nodes[current_parent_id].children_ids.append(new_id)
                print('Initial parent {} for node {}'.format(current_parent_id,new_id))
        
                # Find shortest CollisionFree path to all near nodes
                # By iterating through all it should rewire all nodes within ball of radius
                for id, node in enumerate(self.nodes):
                    if id == new_id or id == current_parent_id:
                        continue
                    # Check if the node is in the ball radius
                    if np.linalg.norm(node.point[:2]-new_point[:2]) < self.ball_radius():

                        # Get a path between this node and the trajectory_end_pt
                        trajectory_test = self.simulate_trajectory(node.point,self.nodes[new_id].point[:2])

                        if len(trajectory_test) != 0:
                            # If cost is also lower than original then rewire
                            trajectory_test_end_pt = trajectory_test[:,-1]
                            test_path_cost = self.cost_to_come(trajectory_test)
                            new_cost = self.nodes[id].cost + test_path_cost
                            old_cost = self.nodes[new_id].cost
                            if new_cost < old_cost:
                                print('Node {} current parent {} new parent {}'.format(new_id,current_parent_id,id))
                                # Update new path cost
                                self.nodes[new_id].cost = new_cost

                                self.nodes[new_id].point = trajectory_test_end_pt.reshape((3,1))

                                # Rewire node by deleting old node in graph
                                self.nodes[new_id].parent_id = id
                                
                                # Add this node Id to new parent node's children
                                self.nodes[id].children_ids.append(new_id)

                                # Remove the node from old old parents id
                                self.nodes[current_parent_id].children_ids.remove(new_id)
                                
                                # Update the new parent id
                                current_parent_id = id
                                
                # Update all nodes in ball radius
                for id, node in enumerate(self.nodes):
                    if id == new_id or id == current_parent_id:
                        continue
                    # Check if the node is in the ball radius
                    new_id = len(self.nodes)-1
                    if np.linalg.norm(node.point[:2]-new_point[:2]) < self.ball_radius():

                        # Get a path between this node and the trajectory_end_pt
                        trajectory_test = self.simulate_trajectory(self.nodes[new_id].point,node.point)

                        # Check if the rewired path is collision free
                        if len(trajectory_test) != 0:

                            # If cost is also lower than original then rewire
                            trajectory_test_end_pt = trajectory_test[:,-1]
                            test_path_cost = self.cost_to_come(trajectory_test)
                            new_cost = self.nodes[new_id].cost + test_path_cost
                            old_cost = self.nodes[id].cost

                            if new_cost < old_cost:
                                print('Adding new child {} to node {}'.format(id,new_id))
                                old_parent_id = self.nodes[id].parent_id
                                # Update new path cost
                                self.nodes[id].cost = new_cost

                                # Rewire node by deleting old node in graph
                                self.nodes[id].parent_id = new_id
                                
                                # Add this node Id to new parent node's children
                                self.nodes[new_id].children_ids.append(id)

                                # Remove the node from old old parents id
                                self.nodes[old_parent_id].children_ids.remove(id)

                                # Update children
                                self.update_children(id)
                #closest_id = self.nodes[self.closest_node(self.goal_point.reshape((2,1)))].point[:2,0]
                print(new_point[:2])
                print(np.linalg.norm(new_point[:2,-1] - self.goal_point.reshape(2,1)))
            #Check for early end
                if np.linalg.norm(new_point[:2] - self.goal_point.reshape((1,2))) < self.stopping_dist:
                    print('At goal!')
                    break
        plt.imshow(data)
        plt.show()
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

    random.seed(26)

    #robot information
    goal_point = np.array([42, 26]) #m
    # goal_point = np.array([0, 0]) #m

    stopping_dist = 1 #m

    data = mpimg.imread('../maps/willowgarageworld_05res.png')        
    plt.imshow(data)

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    goal_point_coords = path_planner.point_to_cell(goal_point)
    print("goal_point_coords: ", goal_point_coords)
    plt.scatter(goal_point_coords[0], goal_point_coords[1], marker='x', color='g')
    plt.show()

    # nodes = path_planner.rrt_star_planning()
    nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())
    print(node_path_metric)

    
    if len(node_path_metric) != 0:
        # Plot point
        for i in range(0,len(node_path_metric[0]),1):
            pt = node_path_metric[:,i].reshape((3,1))

            pix = path_planner.point_to_cell(pt)
            footprint = path_planner.points_to_robot_circle(pt)
            plt.scatter(footprint[0], footprint[1], color='r')
            plt.scatter(pix[0], pix[1], marker='o')
    data = mpimg.imread('../maps/willowgarageworld_05res.png')        
    plt.imshow(data)
    plt.show()
        # #Plot paths
        # for i in range(1,len(node_path_metric),1):
        #     start = node_path_metric(i-1).reshape((3,1))
        #     end = node_path_metric(i).reshape((3,1))
        #     traj = path_planner.connect_node_to_point(start,end[:2])

        #     for j in range(0,int(len(traj[0])),5):
        #         traj_pix = path_planner.point_to_cell(traj[0:2,j])
        #         plt.scatter(traj_pix[0], traj_pix[1], color='g', marker='1')

    # #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)

    # node_i = np.array([[0, 0, 2.5]]).T
    # point_s = goal_point  
    # final_trajectory = path_planner.simulate_trajectory(node_i, point_s)
    # print("Final Trajectory:", final_trajectory)
    # print("shape", np.shape(final_trajectory))

    # for i, val in enumerate(node_path_metric.T):
    #     if i%10 != 0:
    #         continue
    #     path_planner.window.add_se2_pose(val, length=8, color=(0, 0, 255))
    
    # #path_planner.window.add_se2_pose(node_i.flatten(), length=12, color=(255, 0, 0))

    # # Ensures that pygame window does not close unless keyboard exit (CTRL+C)
    # # running = True
    # # while running:
    # #     for event in pygame.event.get():
    # #         if event.type == pygame.QUIT:
    # #             running = False


if __name__ == '__main__':
    main()