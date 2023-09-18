#include <csignal>
#include <iostream>
#include <map>
#include <vector>
#include <numeric>

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "std_msgs/Empty.h"
#include "image_geometry/pinhole_camera_model.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Vector3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <image_transport/image_transport.h>

using namespace std;
using namespace sensor_msgs;
using namespace cv;

image_transport::Publisher result_img_pub_;
#define SSTR(x) static_cast<std::ostringstream&>(std::ostringstream() << std::dec << x).str()
#define ROUND2(x) std::round(x * 100) / 100
#define ROUND3(x) std::round(x * 1000) / 1000

bool camera_model_computed = false;
bool show_detections;
float marker_size;
image_geometry::PinholeCameraModel camera_model;
Mat distortion_coefficients;
Matx33d intrinsic_matrix;
Ptr<aruco::DetectorParameters> detector_params;
Ptr<cv::aruco::Dictionary> dictionary;
string marker_tf_prefix;
int blur_window_size = 7;
int image_fps = 30;
int image_width = 640;
int image_height = 480;
bool enable_blur = true;

int num_detected = 10;
int min_prec_value = 80;
map<int, vector<int>> ids_hashmap;

void int_handler(int x) {
    if (show_detections) {
        cv::destroyAllWindows();
    }
    ros::shutdown();
    exit(0);
}

tf2::Vector3 cv_vector3d_to_tf_vector3(const Vec3d &vec) {
    return {vec[0], vec[1], vec[2]};
}

tf2::Quaternion cv_vector3d_to_tf_quaternion(const Vec3d &rotation_vector) {
    Mat rotation_matrix;
    auto ax = rotation_vector[0], ay = rotation_vector[1], az = rotation_vector[2];
    auto angle = sqrt(ax * ax + ay * ay + az * az);
    auto cosa = cos(angle * 0.5);
    auto sina = sin(angle * 0.5);
    auto qx = ax * sina / angle;
    auto qy = ay * sina / angle;
    auto qz = az * sina / angle;
    auto qw = cosa;
    tf2::Quaternion q;
    q.setValue(qx, qy, qz, qw);
    return q;
}

tf2::Transform create_transform(const Vec3d &tvec, const Vec3d &rotation_vector) {
    tf2::Transform transform;
    transform.setOrigin(cv_vector3d_to_tf_vector3(tvec));
    transform.setRotation(cv_vector3d_to_tf_quaternion(rotation_vector));
    return transform;
}

void callback_camera_info(const CameraInfoConstPtr &msg) {
    if (camera_model_computed) {
        return;
    }
    camera_model.fromCameraInfo(msg);
    camera_model.distortionCoeffs().copyTo(distortion_coefficients);
    intrinsic_matrix = camera_model.intrinsicMatrix();
    camera_model_computed = true;
    ROS_INFO("Camera model is computed");
}

void callback(const ImageConstPtr &image_msg) {
    if (!camera_model_computed) {
        ROS_INFO("Camera model is not computed yet");
        return;
    }

    string frame_id = "camera_link";  // Set your base frame as "base_link"
    auto image = cv_bridge::toCvShare(image_msg)->image;
    cv::Mat display_image(image);

    // Smooth the image to improve detection results
    if (enable_blur) {
        GaussianBlur(image, image, Size(blur_window_size, blur_window_size), 0, 0);
    }

    // Detect the markers
    vector<int> ids;
    vector<vector<Point2f>> corners, rejected;
    aruco::detectMarkers(image, dictionary, corners, ids, detector_params, rejected);

    // Show image if no markers are detected
    if (ids.empty()) {
        cv::putText(display_image, "No markers found", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 3);
        if (show_detections) {
            if (result_img_pub_.getNumSubscribers() > 0) {
                result_img_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", display_image).toImageMsg());
            }
            auto key = waitKey(1);
            if (key == 27) {
                ROS_INFO("ESC pressed, exiting the program");
                ros::shutdown();
            }
        }
        return;
    }

    // Compute poses of markers
    vector<Vec3d> rotation_vectors, translation_vectors;
    aruco::estimatePoseSingleMarkers(corners, marker_size, intrinsic_matrix, distortion_coefficients, rotation_vectors, translation_vectors);

    // Draw marker poses
    if (show_detections) {
        aruco::drawDetectedMarkers(display_image, corners, ids);

        if (result_img_pub_.getNumSubscribers() > 0) {
            cv::putText(display_image, "" + SSTR(image_width) + "x" + SSTR(image_height) + "@" + SSTR(image_fps) + "FPS m. size: " + SSTR(marker_size) + " m" + " blur: " + SSTR(blur_window_size), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 255, 0), 2);

            for (int i = 0; i < ids.size(); i++) {
                vector<int> current_vector(num_detected);
                current_vector = ids_hashmap[ids[i]];
                int num_detections = std::accumulate(current_vector.begin(), current_vector.end(), 0);
                double prec = (double) num_detections / num_detected * 100;

                if (prec >= min_prec_value) {
                    Vec3d distance_z_first = translation_vectors[i];
                    double distance_z = ROUND3(distance_z_first[2]);
                    cv::putText(display_image, "id: " + SSTR(ids[i]) + " z dis: " + SSTR(distance_z) + " m  " + SSTR(ROUND2(prec)) + " %", cv::Point(10, 70 + i * 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, CV_RGB(0, 255, 0), 2);
                    result_img_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", display_image).toImageMsg());
                }
            }
        }

        auto key = waitKey(1);
        if (key == 27) {
            ROS_INFO("ESC pressed, exiting the program");
            ros::shutdown();
        }
    }

    // Publish TFs for each of the markers
    static tf2_ros::TransformBroadcaster br;
    auto stamp = ros::Time::now();

    // Create and publish tf message for each marker
    for (auto i = 0; i < rotation_vectors.size(); ++i) {
        auto translation_vector = translation_vectors[i];

        // Add +90 degrees of pitch to the orientation
        tf2::Quaternion rotation_quaternion;
        rotation_quaternion.setRPY(0, 1.57079632679, 0);

        auto transform = create_transform(translation_vector, rotation_vectors[i]);
        transform.setRotation(rotation_quaternion);

        // Adjust the translation to match the camera's orientation
        tf2::Matrix3x3 rotation_matrix(transform.getRotation());
        tf2::Vector3 rotated_translation = rotation_matrix * tf2::Vector3(translation_vector[0], translation_vector[1], translation_vector[2]);

        // Set the adjusted translation
        transform.setOrigin(rotated_translation);

        geometry_msgs::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = frame_id;
        stringstream ss;
        ss << marker_tf_prefix << ids[i];
        tf_msg.child_frame_id = ss.str();
        tf_msg.transform.translation.x = transform.getOrigin().getX();
        tf_msg.transform.translation.y = transform.getOrigin().getY();
        tf_msg.transform.translation.z = transform.getOrigin().getZ();
        tf_msg.transform.rotation.x = transform.getRotation().getX();
        tf_msg.transform.rotation.y = transform.getRotation().getY();
        tf_msg.transform.rotation.z = transform.getRotation().getZ();
        tf_msg.transform.rotation.w = transform.getRotation().getW();
        br.sendTransform(tf_msg);
    }
}

int main(int argc, char **argv) {
    map<string, aruco::PREDEFINED_DICTIONARY_NAME> dictionary_names;
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_4X4_50", aruco::DICT_4X4_50));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_4X4_100", aruco::DICT_4X4_100));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_4X4_250", aruco::DICT_4X4_250));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_4X4_1000", aruco::DICT_4X4_1000));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_5X5_50", aruco::DICT_5X5_50));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_5X5_100", aruco::DICT_5X5_100));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_5X5_250", aruco::DICT_5X5_250));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_5X5_1000", aruco::DICT_5X5_1000));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_6X6_50", aruco::DICT_6X6_50));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_6X6_100", aruco::DICT_6X6_100));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_6X6_250", aruco::DICT_6X6_250));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_6X6_1000", aruco::DICT_6X6_1000));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_7X7_50", aruco::DICT_7X7_50));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_7X7_100", aruco::DICT_7X7_100));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_7X7_250", aruco::DICT_7X7_250));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_7X7_1000", aruco::DICT_7X7_1000));
    dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_ARUCO_ORIGINAL", aruco::DICT_ARUCO_ORIGINAL));

    signal(SIGINT, int_handler);

    ros::init(argc, argv, "aruco_detector_ocv");
    ros::NodeHandle nh("~");
    string rgb_topic, rgb_info_topic, dictionary_name;
    nh.param("camera", rgb_topic, string("/kinect2/hd/image_color_rect"));
    nh.param("camera_info", rgb_info_topic, string("/kinect2/hd/camera_info"));
    nh.param("show_detections", show_detections, true);
    nh.param("tf_prefix", marker_tf_prefix, string("marker"));
    nh.param("marker_size", marker_size, 0.09f);
    nh.param("enable_blur", enable_blur, true);
    nh.param("blur_window_size", blur_window_size, 7);
    nh.param("image_fps", image_fps, 30);
    nh.param("image_width", image_width, 640);
    nh.param("image_height", image_height, 480);
    nh.param("num_detected", num_detected, 50);
    nh.param("min_prec_value", min_prec_value, 80);

    detector_params = aruco::DetectorParameters::create();
    detector_params->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
    nh.param("dictionary_name", dictionary_name, string("DICT_6X6_250"));
    nh.param("aruco_adaptiveThreshWinSizeStep", detector_params->adaptiveThreshWinSizeStep, 4);
    int queue_size = 10;

    dictionary = aruco::getPredefinedDictionary(dictionary_names[dictionary_name]);
    ROS_DEBUG("%f", marker_size);

    if (show_detections) {
        // namedWindow("markers", cv::WINDOW_KEEPRATIO);
    }
    ros::Subscriber rgb_sub = nh.subscribe(rgb_topic.c_str(), queue_size, callback);
    ros::Subscriber rgb_info_sub = nh.subscribe(rgb_info_topic.c_str(), queue_size, callback_camera_info);

    image_transport::ImageTransport it(nh);
    result_img_pub_ = it.advertise("/result_img", 1);

    ros::spin();
    return 0;
}


//tf만 바뀐듯
// #include <csignal>
// #include <iostream>
// #include <map>
// #include <vector>
// #include <numeric>

// #include "ros/ros.h"
// #include "sensor_msgs/Image.h"
// #include "sensor_msgs/CameraInfo.h"
// #include "std_msgs/Empty.h"
// #include "image_geometry/pinhole_camera_model.h"
// #include "tf2_ros/transform_broadcaster.h"
// #include "tf2/LinearMath/Vector3.h"
// #include "tf2/LinearMath/Quaternion.h"
// #include "tf2/LinearMath/Transform.h"
// #include "cv_bridge/cv_bridge.h"
// #include <opencv2/highgui.hpp>
// #include <opencv2/aruco.hpp>
// #include <opencv2/imgproc.hpp>
// #include <image_transport/image_transport.h>

// using namespace std;
// using namespace sensor_msgs;
// using namespace cv;

// image_transport::Publisher result_img_pub_;
// #define SSTR(x) static_cast<std::ostringstream&>(std::ostringstream() << std::dec << x).str()
// #define ROUND2(x) std::round(x * 100) / 100
// #define ROUND3(x) std::round(x * 1000) / 1000

// bool camera_model_computed = false;
// bool show_detections;
// float marker_size;
// image_geometry::PinholeCameraModel camera_model;
// Mat distortion_coefficients;
// Matx33d intrinsic_matrix;
// Ptr<aruco::DetectorParameters> detector_params;
// Ptr<cv::aruco::Dictionary> dictionary;
// string marker_tf_prefix;
// int blur_window_size = 7;
// int image_fps = 30;
// int image_width = 640;
// int image_height = 480;
// bool enable_blur = true;

// int num_detected = 10;
// int min_prec_value = 80;
// map<int, vector<int>> ids_hashmap;

// void int_handler(int x) {
//     if (show_detections) {
//         cv::destroyAllWindows();
//     }
//     ros::shutdown();
//     exit(0);
// }

// tf2::Vector3 cv_vector3d_to_tf_vector3(const Vec3d &vec) {
//     return {vec[0], vec[1], vec[2]};
// }

// tf2::Quaternion cv_vector3d_to_tf_quaternion(const Vec3d &rotation_vector) {
//     Mat rotation_matrix;
//     auto ax = rotation_vector[0], ay = rotation_vector[1], az = rotation_vector[2];
//     auto angle = sqrt(ax * ax + ay * ay + az * az);
//     auto cosa = cos(angle * 0.5);
//     auto sina = sin(angle * 0.5);
//     auto qx = ax * sina / angle;
//     auto qy = ay * sina / angle;
//     auto qz = az * sina / angle;
//     auto qw = cosa;
//     tf2::Quaternion q;
//     q.setValue(qx, qy, qz, qw);
//     return q;
// }

// tf2::Transform create_transform(const Vec3d &tvec, const Vec3d &rotation_vector) {
//     tf2::Transform transform;
//     transform.setOrigin(cv_vector3d_to_tf_vector3(tvec));

//     // Use a fixed orientation (roll, pitch, yaw = 0)
//     tf2::Quaternion fixed_rotation(0, 0, 0, 1);
//     transform.setRotation(fixed_rotation);

//     return transform;
// }

// void callback_camera_info(const CameraInfoConstPtr &msg) {
//     if (camera_model_computed) {
//         return;
//     }
//     camera_model.fromCameraInfo(msg);
//     camera_model.distortionCoeffs().copyTo(distortion_coefficients);
//     intrinsic_matrix = camera_model.intrinsicMatrix();
//     camera_model_computed = true;
//     ROS_INFO("Camera model is computed");
// }

// void callback(const ImageConstPtr &image_msg) {
//     if (!camera_model_computed) {
//         ROS_INFO("Camera model is not computed yet");
//         return;
//     }

//     string frame_id = "camera_link";  // Set your base frame as "base_link"
//     auto image = cv_bridge::toCvShare(image_msg)->image;
//     cv::Mat display_image(image);

//     // Smooth the image to improve detection results
//     if (enable_blur) {
//         GaussianBlur(image, image, Size(blur_window_size, blur_window_size), 0, 0);
//     }

//     // Detect the markers
//     vector<int> ids;
//     vector<vector<Point2f>> corners, rejected;
//     aruco::detectMarkers(image, dictionary, corners, ids, detector_params, rejected);

//     // Show image if no markers are detected
//     if (ids.empty()) {
//         cv::putText(display_image, "No markers found", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 3);
//         if (show_detections) {
//             if (result_img_pub_.getNumSubscribers() > 0) {
//                 result_img_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", display_image).toImageMsg());
//             }
//             auto key = waitKey(1);
//             if (key == 27) {
//                 ROS_INFO("ESC pressed, exiting the program");
//                 ros::shutdown();
//             }
//         }
//         return;
//     }

//     // Compute poses of markers
//     vector<Vec3d> rotation_vectors, translation_vectors;
//     aruco::estimatePoseSingleMarkers(corners, marker_size, intrinsic_matrix, distortion_coefficients, rotation_vectors, translation_vectors);

//     // Draw marker poses
//     if (show_detections) {
//         aruco::drawDetectedMarkers(display_image, corners, ids);

//         if (result_img_pub_.getNumSubscribers() > 0) {
//             cv::putText(display_image, "" + SSTR(image_width) + "x" + SSTR(image_height) + "@" + SSTR(image_fps) + "FPS m. size: " + SSTR(marker_size) + " m" + " blur: " + SSTR(blur_window_size), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 255, 0), 2);

//             for (int i = 0; i < ids.size(); i++) {
//                 vector<int> current_vector(num_detected);
//                 current_vector = ids_hashmap[ids[i]];
//                 int num_detections = std::accumulate(current_vector.begin(), current_vector.end(), 0);
//                 double prec = (double) num_detections / num_detected * 100;

//                 if (prec >= min_prec_value) {
//                     Vec3d distance_z_first = translation_vectors[i];
//                     double distance_z = ROUND3(distance_z_first[2]);
//                     cv::putText(display_image, "id: " + SSTR(ids[i]) + " z dis: " + SSTR(distance_z) + " m  " + SSTR(ROUND2(prec)) + " %", cv::Point(10, 70 + i * 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, CV_RGB(0, 255, 0), 2);
//                     result_img_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", display_image).toImageMsg());
//                 }
//             }
//         }

//         auto key = waitKey(1);
//         if (key == 27) {
//             ROS_INFO("ESC pressed, exiting the program");
//             ros::shutdown();
//         }
//     }

//     // Publish TFs for each of the markers
//     static tf2_ros::TransformBroadcaster br;
//     auto stamp = ros::Time::now();

//     // Create and publish tf message for each marker
//     for (auto i = 0; i < rotation_vectors.size(); ++i) {
//         auto translation_vector = translation_vectors[i];
        
//         // Declare fixed_rotation here
//         tf2::Quaternion fixed_rotation(0, 0, 0, 1);
        
//         auto transform = create_transform(translation_vector, rotation_vectors[i]); // Use rotation_vectors[i]

//         // Use the fixed rotation for orientation
//         transform.setRotation(fixed_rotation);

//         geometry_msgs::TransformStamped tf_msg;
//         tf_msg.header.stamp = stamp;
//         tf_msg.header.frame_id = frame_id;
//         stringstream ss;
//         ss << marker_tf_prefix << ids[i];
//         tf_msg.child_frame_id = ss.str();
//         tf_msg.transform.translation.x = transform.getOrigin().getX();
//         tf_msg.transform.translation.y = transform.getOrigin().getY();
//         tf_msg.transform.translation.z = transform.getOrigin().getZ();
//         tf_msg.transform.rotation.x = transform.getRotation().getX();
//         tf_msg.transform.rotation.y = transform.getRotation().getY();
//         tf_msg.transform.rotation.z = transform.getRotation().getZ();
//         tf_msg.transform.rotation.w = transform.getRotation().getW();
//         br.sendTransform(tf_msg);
//     }

//     // Rotate vector
//     if (num_detected > 0) {
//         for (auto it = ids_hashmap.begin(); it != ids_hashmap.end(); it++) {
//             vector<int> current_vector(num_detected);
//             current_vector = it->second;
//             rotate(current_vector.begin(), current_vector.end() - 1, current_vector.end());
//             it->second = current_vector;
//         }

//         for (auto it = ids_hashmap.begin(); it != ids_hashmap.end(); it++) {
//             bool current_id_was_found = false;
//             for (int j = 0; j < ids.size(); j++) {
//                 if ((ids[j] == it->first) && (it->second.size() > 1)) {
//                     current_id_was_found = true;
//                     ids.erase(ids.begin() + j);
//                 }
//             }
//             vector<int> current_vector(num_detected);
//             current_vector = it->second;
//             current_vector[0] = 0;
//             if (current_id_was_found) {
//                 current_vector[0] = 1;
//             }
//             it->second = current_vector;
//         }

//         for (int i = 0; i < ids.size(); i++) {
//             std::map<int, vector<int>>::iterator ittt = ids_hashmap.begin();
//             vector<int> tmpp(num_detected, 0);
//             tmpp[0] = 1;
//             ids_hashmap.insert(make_pair(ids[i], tmpp));
//         }

//         for (auto it = ids_hashmap.begin(); it != ids_hashmap.end(); it++) {
//             vector<int> tmp(num_detected, 0);
//             tmp = it->second;

//             if (it->second.size() == 0) {
//                 vector<int> tmpe(num_detected, 0);
//                 tmpe[0] = 1;
//                 it->second = tmpe;
//             }
//         }
//     }
// }

// int main(int argc, char **argv) {
//     map<string, aruco::PREDEFINED_DICTIONARY_NAME> dictionary_names;
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_4X4_50", aruco::DICT_4X4_50));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_4X4_100", aruco::DICT_4X4_100));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_4X4_250", aruco::DICT_4X4_250));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_4X4_1000", aruco::DICT_4X4_1000));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_5X5_50", aruco::DICT_5X5_50));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_5X5_100", aruco::DICT_5X5_100));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_5X5_250", aruco::DICT_5X5_250));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_5X5_1000", aruco::DICT_5X5_1000));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_6X6_50", aruco::DICT_6X6_50));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_6X6_100", aruco::DICT_6X6_100));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_6X6_250", aruco::DICT_6X6_250));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_6X6_1000", aruco::DICT_6X6_1000));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_7X7_50", aruco::DICT_7X7_50));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_7X7_100", aruco::DICT_7X7_100));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_7X7_250", aruco::DICT_7X7_250));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_7X7_1000", aruco::DICT_7X7_1000));
//     dictionary_names.insert(pair<string, aruco::PREDEFINED_DICTIONARY_NAME>("DICT_ARUCO_ORIGINAL", aruco::DICT_ARUCO_ORIGINAL));

//     signal(SIGINT, int_handler);

//     ros::init(argc, argv, "aruco_detector_ocv");
//     ros::NodeHandle nh("~");
//     string rgb_topic, rgb_info_topic, dictionary_name;
//     nh.param("camera", rgb_topic, string("/kinect2/hd/image_color_rect"));
//     nh.param("camera_info", rgb_info_topic, string("/kinect2/hd/camera_info"));
//     nh.param("show_detections", show_detections, true);
//     nh.param("tf_prefix", marker_tf_prefix, string("marker"));
//     nh.param("marker_size", marker_size, 0.09f);
//     nh.param("enable_blur", enable_blur, true);
//     nh.param("blur_window_size", blur_window_size, 7);
//     nh.param("image_fps", image_fps, 30);
//     nh.param("image_width", image_width, 640);
//     nh.param("image_height", image_height, 480);
//     nh.param("num_detected", num_detected, 50);
//     nh.param("min_prec_value", min_prec_value, 80);

//     detector_params = aruco::DetectorParameters::create();
//     detector_params->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
//     nh.param("dictionary_name", dictionary_name, string("DICT_6X6_250"));
//     nh.param("aruco_adaptiveThreshWinSizeStep", detector_params->adaptiveThreshWinSizeStep, 4);
//     int queue_size = 10;

//     dictionary = aruco::getPredefinedDictionary(dictionary_names[dictionary_name]);
//     ROS_DEBUG("%f", marker_size);

//     if (show_detections) {
//         // namedWindow("markers", cv::WINDOW_KEEPRATIO);
//     }
//     ros::Subscriber rgb_sub = nh.subscribe(rgb_topic.c_str(), queue_size, callback);
//     ros::Subscriber rgb_info_sub = nh.subscribe(rgb_info_topic.c_str(), queue_size, callback_camera_info);

//     image_transport::ImageTransport it(nh);
//     result_img_pub_ = it.advertise("/result_img", 1);

//     ros::spin();
//     return 0;
// }