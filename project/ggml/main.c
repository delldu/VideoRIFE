/************************************************************************************
***
*** Copyright 2021-2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, 2021年 11月 22日 星期一 14:33:18 CST
***
************************************************************************************/

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ggml_engine.h>
#include <nimage/tensor.h>

#include "rife.h"

#define DEFAULT_DEVICE 1
#define DEFAULT_OUTPUT "output"

int video_slow_predict(VideoSlowNetwork *slow_net, char *input_file, char *second_file, char *output_file);

static void video_slow_help(char* cmd)
{
    printf("Usage: %s [option] image_files\n", cmd);
    printf("    -h, --help                   Display this help, version %s.\n", ENGINE_VERSION);
    printf("    -d, --device <no>            Set device (0 -- cpu, 1 -- cuda0, 2 -- cuda1, ..., default: %d)\n", DEFAULT_DEVICE);
    printf("    -s, --second <filename>      Set second image file.\n");
    printf("    -o, --output                 output dir, default: %s.\n", DEFAULT_OUTPUT);

    exit(1);
}

int main(int argc, char** argv)
{
    int optc;
    int option_index = 0;
    int device_no = DEFAULT_DEVICE;
    char *second_file = NULL;
    char* output_dir = (char*)DEFAULT_OUTPUT;

    char *p, output_filename[1024];

    struct option long_opts[] = {
        { "help", 0, 0, 'h' },
        { "device", 1, 0, 'd' },
        { "second", 1, 0, 's' },
        { "output", 1, 0, 'o' },
        { 0, 0, 0, 0 }

    };

    if (argc <= 1)
        video_slow_help(argv[0]);


    while ((optc = getopt_long(argc, argv, "h d: s: o:", long_opts, &option_index)) != EOF) {
        switch (optc) {
        case 'd':
            device_no = atoi(optarg);
            break;
        case 's':
            second_file = optarg;
            break;
        case 'o':
            output_dir = optarg;
            break;
        case 'h': // help
        default:
            video_slow_help(argv[0]);
            break;
        }
    }

    // client
    if (optind == argc) // no input image, nothing to do ...
        return 0;

    if (second_file == NULL) {
        printf("Please input second file.");

        video_slow_help(argv[0]);
        return -1;
    }

    VideoSlowNetwork slow_net;
    // int network
    {
        slow_net.init(device_no);
    }

    for (int i = optind; i < argc; i++) {
        p = strrchr(argv[i], '/');
        if (p != NULL) {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, p + 1);
        } else {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, argv[i]);
        }

        video_slow_predict(&slow_net, argv[i], second_file, output_filename);
    }

    // free network ...
    {
        slow_net.exit();
    }

    return 0;
}
