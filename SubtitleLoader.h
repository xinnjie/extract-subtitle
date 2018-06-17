//
// Created by capitalg on 4/12/18.
//

#ifndef TEXT_DETECT_VIDEOLOADER_H
#define TEXT_DETECT_VIDEOLOADER_H
#include <string>
#include <opencv2/videoio.hpp>
#include <exception>


class SubtitleLoader {
public:
    explicit SubtitleLoader(std::string file_path, double interval);
    cv::Mat load();
    cv::VideoCapture capt;
    void set_frame_id(int id);
    int get_frame_id();

    int interval, lower, upper, total_frames_count, current_frame, video_width, video_height;

};


class OutOfFrames : std::exception {
};


#endif //TEXT_DETECT_VIDEOLOADER_H
