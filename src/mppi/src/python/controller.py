#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from gymnasium import spaces
import queue

# Global Variables for state of robot (read from odom)
y = None
x = None
theta = None

# Extracts the turtlebot state from the pose
def get_pose(data):
    global x
    global y
    global theta
    
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    quats = data.pose.pose.orientation
    r, p, yaw = euler_from_quaternion([quats.x, quats.y, quats.z, quats.w])
    theta = yaw

# Global Variables for the map and its informaion
map = None
map_res = None
map_width = None
map_height = None
map_x = None
map_y = None

# Extracts map data
def get_map(data):
    global map
    global map_res
    global map_x
    global map_y
    global map_width
    global map_height

    map = data.data
    map_res = data.info.resolution
    map_width = data.info.width
    map_height = data.info.height
    map_x = data.info.origin.position.x
    map_y = data.info.origin.position.y

# Gets the indices of the map that correspond to the given x, y state
def state_to_map(x1, y1):

    col = round((x1-map_x)/(map_res))
    row = round((y1-map_y)/(map_res))
    return row, col

# Global variables for Lidar scan
angle_min = None
angle_max = None
angle_increment = None
ranges = None

def get_scan(data):
    global angle_min
    global angle_max
    global angle_increment
    global ranges
    
    angle_min = data.angle_min
    angle_max = data.angle_max
    angle_increment = data.angle_increment
    ranges = data.ranges

# Checks whether the given state values are occupied or not depending on the lidar scan
def scan_occupancy(x1, y1):
    dist = np.linalg.norm([x1-x, y1-y])
    
    a = np.arctan2(y1-y, x1-x)
    angle = (a + 2 * np.pi) % (2 * np.pi)

    index = round(angle/angle_increment)

    r = ranges[index]

    if dist > (r - 0.7):
        # print(str(a) + "gonna hit in " + str(dist))
        return 1000
    if dist < r or r is np.inf:
        return 1



#Runs the MPPI Control
def control():
    # Setup the publisher to command the velocity
    pub = rospy.Publisher('/cmd_vel', Twist ,queue_size=100)

    # Subscribes to the odom to get the pose estimate
    rospy.Subscriber('/odom', Odometry, get_pose)
    # Subscribes to the slam map to get the occupancy grid if you need it
    # rospy.Subscriber('/map', OccupancyGrid, get_map)
    # Subcribes to the lidar scan 
    rospy.Subscriber('/scan', LaserScan, get_scan)

    rate = rospy.Rate(1) # 10hz
    unicyle = Unicycle(0.0,0.22, -2.84, 2.84)
    mppi = MPPI(unicyle)

    # Setting up the standard message, these values will always be 0
    msg = Twist()
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0

    goals = queue.Queue()
    goals.put([-2,3.8])
    # goals.put([-3,3])
    goals.put([-6,3])
    goals.put([-6.5,1.0])
    goal_pos = goals.get()
    print("New Goal: " + str(goal_pos))

    while not rospy.is_shutdown():
        # Making sure we get all variables we need
        global_vars = [x, y, theta] #, map_width, map_height, map_x, map_y]
        
        if None not in global_vars:
            state = np.array([x,y, theta])
            
            within_goal = np.linalg.norm([x-goal_pos[0], y - goal_pos[1]]) < 0.75
            if within_goal:
                # Stop and get new goal, then continue next loop
                msg.linear.x = 0.0
                msg.angular.z = 0.0

                pub.publish(msg)
                
                if not goals.empty():
                    goal_pos = goals.get()
                    print("New Goal: " + str(goal_pos))
                    rate.sleep()
                else:
                    msg.linear.x = 0.0
                    msg.angular.z = 0.0
                    pub.publish(msg)
                    print("No More Goals")
                    rate.sleep()              

            action = mppi.get_action(state, goal_pos)

            # The variables to control the robot
            msg.linear.x = action[0]
            msg.angular.z = action[1]

            pub.publish(msg)
            rate.sleep()

        rate.sleep()


class Unicycle():
    """Discrete-time unicycle kinematic model for 2D robot simulator."""

    def __init__(self, v_min=0, v_max=1, w_min=-2 * np.pi, w_max=2 * np.pi):
        self.action_space = spaces.Box(
            np.array([v_min, w_min]),
            np.array([v_max, w_max]),
            shape=(2,),
            dtype=float,
        )
        super().__init__()

    def step(
        self, current_state: np.ndarray, action: np.ndarray, dt: float = 0.5
    ) -> np.ndarray:
        """Move 1 timestep forward w/ kinematic model, x_{t+1} = f(x_t, u_t)"""
        # current_state = np.array([x, y, theta])
        # action = np.array([vx, vw])

        # clip the action to be within the control limits
        clipped_action = np.clip(
            action, self.action_space.low, self.action_space.high
        )

        current_state = current_state.reshape((-1, 3))
        clipped_action = clipped_action.reshape((-1, 2))
        next_state = np.empty_like(current_state)

        next_state[:, 0] = current_state[:, 0] + dt * clipped_action[
            :, 0
        ] * np.cos(current_state[:, 2])
        next_state[:, 1] = current_state[:, 1] + dt * clipped_action[
            :, 0
        ] * np.sin(current_state[:, 2])
        next_state[:, 2] = current_state[:, 2] + dt * clipped_action[:, 1]

        next_state = next_state.squeeze()

        return next_state
    
class MPPI:
    def __init__(
        self,
        motion_model= Unicycle(), 
        max_rolls=50, # The max # of rollouts to consider each step
        alpha=0.1, # The hyperparameter of the weighted sum equation
    ):
        self.motion_model = motion_model
        self.max_rolls = max_rolls 
        self.alpha = alpha
        self.L = 10 # The # of states to look ahead into
        
        # The control limits
        self.v_max = 0.07 # Forces TurtleBot to move at half the max speed
        self.w_max = 2*np.pi
        
        self.nom = np.array([self.v_max,0]) # the current nominal control sequence, initially goes forward
        self.control_seq = [] # The best control sequence stored over time
        self.range = self.w_max/3 # The range in radians for the random control pertubation
        
        self.rolls = [] # The next actions to consider as the current rollouts
        self.trajectories = [] # The trajectory for each rollout
        self.costs = [] # The cost for each trajectory
        
        # The start point of the model
        self.initial_state = None
        
        # The current state of the model
        self.state = np.array([0,0,0])
        

    def get_action(self, state: np.ndarray, goal_pos: np.ndarray):
        # Sets the starting point if there is none
        if(self.initial_state is None):
            self.initial_state = state
            self.plot_init_state = True
        
        self.state = state
        
        # Gets possible rollouts/trajectories and costs them
        self.get_rolls()
        self.cost_rolls(goal_pos)
        
        # Updates the best rollout of the controller based on weighted sums
        self.update_nom()
        return self.nom

    def update_nom(self):
        # Gets the denomenator used in the weighted sum
        den = 0
        for cost in self.costs:
            e = np.exp(-cost/self.alpha)
            den += e
        if not den ==0:
            nums = []
            for i in range(len(self.rolls)):
                roll = self.rolls[i][1]
                cost = self.costs[i]
                e = np.exp(-cost/self.alpha)
                nums.append(roll*e)
            
            new_angle = 0
            for num in nums:
                new_angle += (num/den)
            
            new_nom = np.array([self.v_max, new_angle])
            self.nom = new_nom
            self.control_seq.append(new_nom)
        
    # Gets the possible rollouts and trajectories from the current state
    def get_rolls(self):
        self.rolls = []
        
        # Get uniform random angles within the range 
        angles = np.random.uniform(-1*self.range, self.range, self.max_rolls)
        
        # Adds the action to our rollouts and clips it based on unicycle values
        for angle in angles:
            action = np.array([self.v_max, angle])
            self.rolls.append(action)
             
        # Calculates the trajectory of states for each rollout
        self.trajectories = []  
        for roll in self.rolls:
            current_state = self.state
            trajectory = [current_state]
            for l in range(self.L):
                next_state = self.motion_model.step(current_state, roll)
                trajectory.append(next_state)
                current_state = next_state
            self.trajectories.append(trajectory)
        
    # Costs each trajectory    
    def cost_rolls(self, goal_pos: np.ndarray):
        self.costs = []
        
        for traj in self.trajectories:
            cost = 0
            for state in traj:
                new_x = state[0]
                new_y = state[1]
                
                # If you want to use the map's occupancy grid instead of the lidar
                # row, col = state_to_map(new_x, new_y)
                # index = map_width*row + col

                # For each state in the current trajectory, 
                # add high cost if they hit a wall 
                # if info > 0:
                #     cost += 10000
                # elif info == -1:
                #     cost += 1
                # else:
                #     cost += 1 # Otherwise 1

                # Adds the euclidean distance of each state to the given goal
                euc_cost = np.linalg.norm([new_x - goal_pos[0],new_y - goal_pos[1]])
                cost += euc_cost

                info = scan_occupancy(new_x, new_y)
                cost += info
  
  
            
            self.costs.append(cost)


    
if __name__ == '__main__':
    try:
        rospy.init_node('controller', anonymous=True)
        control()
    except rospy.ROSInterruptException:
        pass
