#!/usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
from pyquaternion import Quaternion
from geometry_msgs.msg import TransformStamped
from apriltag_ros.msg import AprilTagDetectionArray

class artag_corrections_handler:
    def __init__(self) -> None:
        # Initialize ROS node
        rospy.init_node('transformation_calculator', anonymous=True)

        # TF2 buffer and listener for transforming between frames
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # TF2 broadcaster to update transformations dynamically
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Subscribe to the AprilTag detections
        self.apriltag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.apriltag_callback)

        # Load extrinsics from file
        self.lidar_to_camera_extrinsics = self.load_extrinsics("lidar_to_camera_extrinsics.txt")

        # Dictionary to store and update detected tags
        self.detected_tags = {}

        # Map frame
        self.map_frame = 'map'

    def load_extrinsics(self, file_path):
        try:
            with open(file_path, 'r') as file:
                line = file.readline().strip()
                qx, qy, qz, qw, tx, ty, tz = map(float, line.split())
                return {
                    'rotation': [qw, qx, qy, qz],  # Quaternion order is w, x, y, z
                    'translation': [tx, ty, tz]
                }
        except Exception as e:
            rospy.logerr(f"Failed to load extrinsics from {file_path}: {e}")
            return None

    def apriltag_callback(self, data):
        # Process each detected tag
        for detection in data.detections:
            tag_id = detection.id[0]
            tag_transform = detection.pose.pose.pose  # This is the PoseStamped under the tag's frame

            # Transform tag detections from camera to lidar frame
            if self.lidar_to_camera_extrinsics:
                camera_to_tag_transform = self.pose_to_transform_matrix(tag_transform)
                lidar_to_tag_transform = np.dot(self.create_transform_matrix(self.lidar_to_camera_extrinsics['rotation'], self.lidar_to_camera_extrinsics['translation']), camera_to_tag_transform)

                # Optionally transform from lidar to map if required
                self.update_tag_in_map(tag_id, lidar_to_tag_transform)

    def pose_to_transform_matrix(self, pose):
        trans = [pose.position.x, pose.position.y, pose.position.z]
        rot = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        rotation_quaternion = Quaternion(rot)
        rotation_matrix = rotation_quaternion.rotation_matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = trans
        return transform_matrix

    def update_tag_in_map(self, tag_id, lidar_to_tag_transform):
        # Convert lidar to tag transform to map if transformation exists
        try:
            lidar_to_map_tf = self.tf_buffer.lookup_transform(self.map_frame, 'lidar_frame', rospy.Time(0), rospy.Duration(1.0))
            lidar_to_map_matrix = self.transform_to_matrix(lidar_to_map_tf.transform)
            tag_to_map_transform = np.dot(lidar_to_map_matrix, lidar_to_tag_transform)
            self.broadcast_transform(tag_id, tag_to_map_transform, self.map_frame)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Failed to transform from lidar to map frame: {}".format(e))

    def transform_to_matrix(self, transform):
        q = [transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z]
        t = [transform.translation.x, transform.translation.y, transform.translation.z]
        rotation_matrix = Quaternion(q).rotation_matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = t
        return transform_matrix

    def broadcast_transform(self, tag_id, transform_matrix, frame_id):
        msg = TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.child_frame_id = f'tag_{tag_id}'
        msg.transform.translation.x = transform_matrix[0, 3]
        msg.transform.translation.y = transform_matrix[1, 3]
        msg.transform.translation.z = transform_matrix[2, 3]
        q = Quaternion(matrix=transform_matrix)
        msg.transform.rotation.x = q.x
        msg.transform.rotation.y = q.y
        msg.transform.rotation.z = q.z
        msg.transform.rotation.w = q.w
        self.tf_broadcaster.sendTransform(msg)

if __name__ == '__main__':
    try:
        artag_corrections_handler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
