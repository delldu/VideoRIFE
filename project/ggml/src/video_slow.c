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

    {
        input1_tensor = tensor_load_image(input_file, 0 /*input_with_alpha*/);
        check_tensor(input1_tensor);

        input2_tensor = tensor_load_image(second_file, 0 /*input_with_alpha*/);
        check_tensor(input2_tensor);
    }

    {
        TENSOR *timestep = tensor_create(input1_tensor->batch, 1, input1_tensor->height, input1_tensor->width);
        check_tensor(timestep);
        tensor_clamp_(timestep, 0.5, 0.5);

        output_tensor = slow_net->forward(input1_tensor, input2_tensor, timestep);
        check_tensor(output_tensor);
        tensor_destroy(input2_tensor);
        tensor_destroy(input1_tensor);
        tensor_destroy(timestep);

        // TENSOR *xxxx_test;
        // xxxx_test = slow_net->net.get_output_tensor("I1");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** I1", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        tensor_saveas_image(output_tensor, 0 /*batch 0*/, output_file);
        chmod(output_file, 0644);

        tensor_destroy(output_tensor);
    }

    return RET_OK;
}
