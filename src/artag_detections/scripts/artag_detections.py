#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
from apriltag_ros.msg import AprilTagDetectionArray
import numpy as np
from pyquaternion import Quaternion

class artag_detections_handler():
    def __init__(self) -> None:
        # Initialize ROS node
        rospy.init_node('tag_pose_publisher', anonymous=False)
        
        # TF buffer and listener for transforming between frames
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        
        # Subscriber for AprilTag detections
        self.tag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.tag_callback)
        
        # Dictionary to store detected tags and their poses
        self.closed_tags = {}
        
        # Frame names
        self.camera_frame = 'camera'
        self.map_frame = 'map'
        
        # Transformation from camera to map frame
        self.TMC = None
        
        # TF broadcaster for publishing tag frames
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Maximum distance for considering detected tags
        self.camera_max_detect_distance = 3.5
        
    def tag_callback(self, tags):
        # Process AprilTag detections
        num_tags = len(tags.detections)
        if num_tags == 0:
            return
        
        for i in range(num_tags):
            tag_id = int(tags.detections[i].id[0])
            x, y, z = tags.detections[i].pose.pose.pose.position.x, tags.detections[i].pose.pose.pose.position.y, tags.detections[i].pose.pose.pose.position.z
            q = Quaternion(w=tags.detections[i].pose.pose.pose.orientation.w, x=tags.detections[i].pose.pose.pose.orientation.x,
                           y=tags.detections[i].pose.pose.pose.orientation.y, z=tags.detections[i].pose.pose.pose.orientation.z)

            if z < self.camera_max_detect_distance:
                translation_vector = np.array([x, y, z, 1])
                rotation_matrix = q.rotation_matrix
                TCA = np.vstack([np.hstack([rotation_matrix, translation_vector[:3].reshape(3, 1)]), [0, 0, 0, 1]])

                if self.TMC is None:
                    print('[WARNING]: Map to camera transform not available')
                    return

                TMA = np.dot(self.TMC, TCA)

                if tag_id in self.closed_tags:
                    print('UPDATING TAG:', tag_id)
                    L = 0.95
                    self.closed_tags[tag_id] = L * self.closed_tags[tag_id] + (1 - L) * TMA
                else:
                    print('FOUND NEW TAG:', tag_id)
                    self.closed_tags[tag_id] = TMA

    def broadcast_tag_frames(self):
        for tag_id, TMA in self.closed_tags.items():
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = self.map_frame
            t.child_frame_id = f'TAG_{tag_id}'
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = TMA[:3, 3]
            q = Quaternion(matrix=TMA[:3, :3])
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q.x, q.y, q.z, q.w
            self.tf_broadcaster.sendTransform(t)

    def update_camera_to_map_transform(self):
        try:
            trans = self.tfBuffer.lookup_transform(self.map_frame, self.camera_frame, rospy.Time(0))
            q = Quaternion(w=trans.transform.rotation.w, x=trans.transform.rotation.x, 
                           y=trans.transform.rotation.y, z=trans.transform.rotation.z)
            rotation_matrix = q.rotation_matrix
            translation_vector = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z, 1])
            self.TMC = np.vstack([np.hstack([rotation_matrix, translation_vector[:3].reshape(3, 1)]), [0, 0, 0, 1]])
        except Exception as e:
            print("Transform from MAP to Camera not found:", str(e))

def main():
    rospy.init_node('tag_pose_publisher', anonymous=False)
    adh = artag_detections_handler()
    rate = rospy.Rate(60)
    
    while not rospy.is_shutdown():
        adh.update_camera_to_map_transform()
        adh.broadcast_tag_frames()
        rate.sleep()

if __name__ == '__main__':
    main()
