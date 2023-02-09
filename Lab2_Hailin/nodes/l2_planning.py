#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
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

            # Get bounding range for map
            bounding_x, bounding_y = self.bounds[0, :], self.bounds[1, :]

            # Generate random values for x and y
            point = np.random.rand(2, 1)
            point[1, 0] = (bounding_x[1]-bounding_x[0])*point[1, 0] + bounding_x[0]
            point[0, 0] = (bounding_y[1]-bounding_y[0])*point[0, 0] + bounding_y[0]

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
        print("TO DO: Implment a method to simulate a trajectory given a sampled point")

        # Get trajectory for first timestep
        vel, rot_vel = self.robot_controller(node_i, point_s)
        robot_traj = self.trajectory_rollout(vel, rot_vel)

        # Multiply robot_traj by rotation matrix to go to world frame, then add to node_i
        theta = - node_i[2,0]
        rotm = np.eye((3))
        rotm[0, :] = [np.cos(theta), -np.sin(theta), 0]
        rotm[1, :] = [np.sin(theta), np.cos(theta), 0]

        robot_traj = node_i + rotm@robot_traj

        # Iteratively call robot_controller and trajectory_rollout until at point_s
        threshold = 0.10
        tog = 0
        cnt = 0
        
        while (tog == 0) and (cnt < 50) :

            # Get new trajectory in robot frame
            new_vel, new_rot_vel = self.robot_controller(robot_traj[:, cnt], point_s)
            new_pts = self.trajectory_rollout(new_vel, new_rot_vel)

            # Transform new pts to world frame
            theta = - robot_traj[2,cnt]
            rotm = np.eye((3))
            rotm[0, :] = [np.cos(theta), -np.sin(theta), 0]
            rotm[1, :] = [np.sin(theta), np.cos(theta), 0]
            new_pts_wf = rotm@robot_traj + robot_traj[:, cnt]
            robot_traj = np.hstack(robot_traj, new_pts_wf)

            # Collision check new_pt
            # circle_pixels = self.points_to_robot_circle(new_pts)
            # col = self.check_collision(circle_pixels)
            # if (col == 1):
            #     return []
            # for pix in circle_pixels:
            #     if self.occupancy_map[pix[0], pix[1]] < 10:
            #         return []
                
            # If not occupied add new pts to path
            robot_traj = np.hstack(robot_traj, new_pts_wf)

            # Check if new point is close enough to point_s
            if np.linalg.norm(robot_traj[1:2, robot_traj.shape[1]]) < threshold:
                tog = 1
        
        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        # This controller determines the velocities that will nominally move the robot from node i to node s
        # Max velocities should be enforced
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

        return lin_vel,ang_vel
    
    def trajectory_rollout(self, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        print("TO DO: Implement a way to rollout the controls chosen")

        # Initialize empty trajectory
        traj = np.zeros((3, self.num_substeps))
        vel = np.array([[vel, 0, rot_vel]]).T

        # Generate time stamps of sub steps
        substep_size = timestep / num_substeps
        rotm = np.eye((3))

        # Run through each substep
        for i in range(1, self.num_substeps-1):
            # Build rotation matrix
            theta = traj[2, i]
            rotm[0, :] = [np.cos(theta), -np.sin(theta), 0]
            rotm[1, :] = [np.sin(theta), np.cos(theta), 0]
            
            traj[:, i] = traj[:, i-1] + np.reshape(rotm@vel*substep_size, (3,))

        return traj
        
        
    def point_to_cell(self, point):

        # Get origin values (coordinates of left bottom corner in m)
        x_origin, y_origin, theta_origin = self.map_settings_dict['origin']
        resolution = self.map_settings_dict['resolution']

        # Instantiate conversion array
        points = np.zeros_like(point)

        # Convert metric points to be measured from upper left corner
        x = point[0, :] + abs(x_origin)
        y = -1*point[1, :] + abs(y_origin)

        points[0, :] = y / resolution
        points[1, :] = x / resolution

        return points

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function

        # For each point that robot is expected draw a circle
        # Then then draw a circle with all pixel locations that robot covers
        robot_locations_rr = list()
        robot_locations_cc = list()

        for p in points:
            
            pixel = self.point_to_cell(p)
            rr, cc = disk((pixel[0], pixel[1]), self.robot_radius)
            robot_locations_rr.append(rr)
            robot_locations_cc.append(cc)


        return robot_locations_rr, robot_locations_cc

    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions

    def check_collision(self, rr, cc):

        if 0 in self.occupancy_map[rr, cc]:
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
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(trajectory):
        cost = 0

        # For each point in trajectory
        for i in range(1,len(trajectory)):
            cost += np.linalg.norm(trajectory[i,:2] - trajectory[i-1,:2])
        return cost
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        for child in node_id.children_ids:
            traj = self.connect_node_to_point(node_id.point,child.point[:2])
            cost = self.cost_to_come(traj)
            child.cost = node_id.cost + cost
            self.update_children(child)
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            print("TO DO: Check for collisions and add safe points to list of nodes.")
            
            #Check if goal has been reached
            print("TO DO: Check if at goal point.")
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot  
        num_iters = 50

        for i in range(num_iters): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point) #TODO what is this ID function doing, currently assuming the point is being used as id

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")
            rr, cc = self.points_to_robot_circle(trajectory_o)

            
            # Assuming that we are getting the final part of the path to add to the graph
            trajectory_end_pt = point

            if not(self.check_collision(rr, cc)): # TODO get the obstacles
                # Add node from trajectory_ o to graph
                path_cost = self.cost_to_come(trajectory_o)
                new_node = Node(trajectory_end_pt, closest_node_id, path_cost)
                self.nodes.append(new_node)
                # Add this node Id to parent node's children
                new_id = len(self.nodes)-1
                self.nodes[closest_node_id].children_ids.append(new_id)


        
                # Find shortest CollisionFree path to all near nodes
                # By iterating through all it should rewire all nodes within ball of radius
                ## TODO do we need to manually rewire closest_node_id after iterating through all nodes in ball radius

                current_parent_id = closest_node_id
                for id, node in enumerate(self.nodes):
                    # Check if the node is in the ball radius
                    if np.linalg.norm(node.point-trajectory_end_pt) < self.ball_radius():

                        # Get a path between this node and the trajectory_end_pt
                        trajectory_test = self.simulate_trajectory(node.point,self.nodes[trajectory_end_pt].point)
                        rr_test, cc_test = self.points_to_robot_circle(trajectory_test)
                        if not(self.check_collision(rr_test, cc_test)):

                            # If cost is also lower than original then rewire
                            trajectory_test_end_pt = trajectory_test[-1]
                            test_path_cost = self.cost_to_come(trajectory_test_end_pt)
                            if test_path_cost < path_cost:

                                # Update new path cost
                                path_cost = test_path_cost

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
                    # Check if the node is in the ball radius
                    new_id = len(self.nodes)-1
                    if np.linalg.norm(node.point-trajectory_end_pt) < self.ball_radius():

                        # Get a path between this node and the trajectory_end_pt
                        trajectory_test = self.simulate_trajectory(self.nodes[new_id].point,node.point)
                        rr_test, cc_test = self.points_to_robot_circle(trajectory_test)

                        # Check if the rewired path is collision free
                        if (rr_test, cc_test) not in obstacles:

                            # If cost is also lower than original then rewire
                            trajectory_test_end_pt = trajectory_test[-1]
                            test_path_cost = self.cost_to_come(trajectory_test)
                            new_cost = self.nodes[trajectory_end_pt].cost + test_path_cost
                            old_cost = self.nodes[id].cost

                            if new_cost < old_cost:
                                current_parent_id = self.nodes[id].parent_id
                                # Update new path cost
                                self.nodes[id].cost = new_cost

                                # Rewire node by deleting old node in graph
                                self.nodes[id].parent_id = new_id
                                
                                # Add this node Id to new parent node's children
                                self.nodes[new_id].children_ids.append(id)

                                # Remove the node from old old parents id
                                self.nodes[current_parent_id].children_ids.remove(id)

                                # Update children
                                self.update_children(id)


            #Check for early end
            if np.linalg.norm(trajectory_end_pt[:2] - self.goal_point) < self.stopping_dist:
                break

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
    nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
