#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

project(sample_cudamapper)

# TODO VI: I don't think I need to include the data dir
get_property(cudamapper_data_include_dir GLOBAL PROPERTY cudamapper_data_include_dir)
include_directories(${cudamapper_data_include_dir})

add_executable(${PROJECT_NAME}
               sample_cudamapper.cpp
               )

target_link_libraries(${PROJECT_NAME}
                      cudamapper
                      )

install(TARGETS ${PROJECT_NAME}
            DESTINATION samples)