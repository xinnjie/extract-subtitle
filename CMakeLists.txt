cmake_minimum_required(VERSION 3.9)
project(text_detect)
set(CMAKE_CXX_STANDARD 11)


# OpenCV
find_package(OpenCV REQUIRED)
#Boost
set(Boost_USE_STATIC_LIBS        ON)
FIND_PACKAGE(Boost COMPONENTS system  filesystem REQUIRED)

file( GLOB_RECURSE TextDetectionFiles
        "TextDetection/*"
        )

set(pic_clean_files PicClean.cpp PicClean.h)
set(video_loader_files SubtitleLoader.cpp SubtitleLoader.h)


#add_executable(text_detect not_used/main.cpp ${TextDetectionFiles})
#target_link_libraries(text_detect ${OpenCV_LIBS} ${Boost})
#
#
#
#add_executable(demo_detect lab/detect.cpp)
#target_link_libraries(demo_detect ${OpenCV_LIBS} )
#
#
#add_executable(demo_video lab/video_demo.cpp)
#target_link_libraries(demo_video ${OpenCV_LIBS} )
#
#add_executable(try_video not_used/locate_subtitle.cpp ${TextDetectionFiles}  PicClean.cpp PicClean.h)
#target_link_libraries(try_video ${OpenCV_LIBS}  ${Boost_LIBRARIES})



#add_executable(lena  tutorial_code/imgProc/Smoothing/Smoothing.cpp)
#target_link_libraries(lena ${OpenCV_LIBS} )

#
#message(${Boost_LIBRARIES})
#message(${Boost})
#message(${OpenCV_LIBS} )


#add_executable(better_text not_used/better_text.cpp)
#target_link_libraries(better_text ${OpenCV_LIBS})
#
#
#add_executable(clean_text not_used/clean_test.cpp)
#target_link_libraries(clean_text ${OpenCV_LIBS})

#add_executable(try_swt ${TextDetectionFiles} not_used/run_swt.cpp ${video_loader_files})
#target_link_libraries(try_swt ${OpenCV_LIBS})


add_executable(extract_subtiltes ${pic_clean_files} ${video_loader_files} extract_subtitles.cpp)
target_link_libraries(extract_subtiltes ${OpenCV_LIBS} ${Boost_LIBRARIES})

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
