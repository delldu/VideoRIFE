/************************************************************************************
***
***	Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, Tue 02 Apr 2024 03:49:53 PM CST
***
************************************************************************************/

#include "rife.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

#include <sys/stat.h> // for chmod()


int video_slow_predict(VideoSlowNetwork *slow_net, char *input_file, char *second_file, char *output_file)
{
    TENSOR *input1_tensor, *input2_tensor, *output_tensor;

    CheckPoint("input_file = %s, second_file = %s", input_file, second_file);

    // Loading content tensor and it's segment tensor
    {
        input1_tensor = tensor_load_image(input_file, 0 /*input_with_alpha*/);
        check_tensor(input1_tensor);

        input2_tensor = tensor_load_image(second_file, 0 /*input_with_alpha*/);
        check_tensor(input2_tensor);

    }

    // Blender input1_tensor & style_tensor
    {
        TENSOR *timestep = tensor_create(input1_tensor->batch, 1, input1_tensor->height, input1_tensor->width);
        check_tensor(timestep);
        tensor_clamp_(timestep, 0.5, 0.5);
        
        output_tensor = slow_net->forward(input1_tensor, input2_tensor, timestep);
        check_tensor(output_tensor);
        tensor_destroy(input2_tensor);
        tensor_destroy(input1_tensor);
        tensor_destroy(timestep);

        tensor_show("-------------- output_tensor", output_tensor);


        TENSOR *xxxx_test;

        // xxxx_test = slow_net->net.get_output_tensor("xlist0");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** xlist0", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = slow_net->net.get_output_tensor("xlist1");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** xlist1", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = slow_net->net.get_output_tensor("xlist2");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** xlist2", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = slow_net->net.get_output_tensor("xlist3");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** xlist3", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = slow_net->net.get_output_tensor("cnet");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** cnet", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = slow_net->net.get_output_tensor("net");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** net", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = slow_net->net.get_output_tensor("inp");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** inp", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }


        xxxx_test = slow_net->net.get_output_tensor("x_flow");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_flow", xxxx_test);
            tensor_destroy(xxxx_test);
        }

        xxxx_test = slow_net->net.get_output_tensor("x_corr");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_corr", xxxx_test);
            tensor_destroy(xxxx_test);
        }

        xxxx_test = slow_net->net.get_output_tensor("x_motion_feat");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_motion_feat", xxxx_test);
            tensor_destroy(xxxx_test);
        }

        xxxx_test = slow_net->net.get_output_tensor("x_inp");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_inp", xxxx_test);
            tensor_destroy(xxxx_test);
        }


        xxxx_test = slow_net->net.get_output_tensor("x_net");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_net", xxxx_test);
            tensor_destroy(xxxx_test);
        }

        xxxx_test = slow_net->net.get_output_tensor("x_delta_flow");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_delta_flow", xxxx_test);
            tensor_destroy(xxxx_test);
        }





        xxxx_test = slow_net->net.get_output_tensor("m");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** m", xxxx_test);
            tensor_destroy(xxxx_test);
        }

        xxxx_test = slow_net->net.get_output_tensor("update_mask");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** update_mask", xxxx_test);
            tensor_destroy(xxxx_test);
        }


        // xxxx_test = slow_net->net.get_output_tensor("m");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** m", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = slow_net->net.get_output_tensor("up_mask");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** up_mask", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }


        // xxxx_test = slow_net->net.get_output_tensor("BatchBasicEncoder");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** BatchBasicEncoder", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }


        // tensor_saveas_image(output_tensor, 0 /*batch 0*/, output_file);
        // chmod(output_file, 0644);

        tensor_destroy(output_tensor);
    }

    return RET_OK;
}
