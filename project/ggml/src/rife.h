#ifndef __RIFE__H__
#define __RIFE__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#include <vector>

#pragma GCC diagnostic ignored "-Wformat-truncation"
// x = ggml_cont(ctx, x);
// ggml_set_name(x, "x");
// ggml_set_output(x);

struct Conv2dReLU {
    int in_planes;
    int out_planes;

    struct Conv2d conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = in_planes;
        conv.out_channels = out_planes;
        conv.kernel_size = {3, 3};
        conv.stride = { 2, 2 };
        conv.padding = { 1, 1 };
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        conv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);
        return x;
    }
};

struct Head {
    struct Conv2d cnn0;
    struct Conv2d cnn1;
    struct Conv2d cnn2;
    struct ConvTranspose2d cnn3;

    void create_weight_tensors(struct ggml_context* ctx) {
        cnn0.in_channels = 3;
        cnn0.out_channels = 32;
        cnn0.kernel_size = {3, 3};
        cnn0.stride = { 2, 2 };
        cnn0.padding = { 1, 1 };
        cnn0.create_weight_tensors(ctx);

        cnn1.in_channels = 32;
        cnn1.out_channels = 32;
        cnn1.kernel_size = {3, 3};
        cnn1.stride = { 1, 1 };
        cnn1.padding = { 1, 1 };
        cnn1.create_weight_tensors(ctx);

        cnn2.in_channels = 32;
        cnn2.out_channels = 32;
        cnn2.kernel_size = {3, 3};
        cnn2.stride = { 1, 1 };
        cnn2.padding = { 1, 1 };
        cnn2.create_weight_tensors(ctx);

        cnn3.in_channels = 32;
        cnn3.out_channels = 8;
        cnn3.kernel_size = 4;
        cnn3.stride = 2;
        cnn3.padding = 1;
        cnn3.output_padding = 1;
        cnn3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "cnn0.");
        cnn0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "cnn1.");
        cnn1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "cnn2.");
        cnn2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "cnn3.");
        cnn3.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
      x = cnn0.forward(ctx, x);
      x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);

      x = cnn1.forward(ctx, x);
      x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);

      x = cnn2.forward(ctx, x);
      x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);

      x = cnn3.forward(ctx, x);

    	return x;
    }
};

struct ResConv {
    int c;

    struct Conv2d conv;
    ggml_tensor_t *beta;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = c;
        conv.out_channels = c;
        conv.kernel_size = {3, 3};
        conv.stride = { 1, 1 };
        conv.padding = { 1, 1 };
        conv.create_weight_tensors(ctx);

        beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, c, 1);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);

        ggml_format_name(beta, "%s%s", prefix, "beta");
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // return self.relu(self.conv(x) * self.beta + x)
        ggml_tensor_t *y = conv.forward(ctx, x);
        y = ggml_mul(ctx, y, beta);
        x = ggml_add(ctx, y, x);
        x = ggml_leaky_relu(ctx, x, 0.2, true/*inplace*/);

        return x;
    }
};


struct IFBlockFirst {
    int in_planes;
    int c;

    struct Conv2dReLU conv_0;
    struct Conv2dReLU conv_1;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv_0.in_planes = in_planes;
        conv_0.out_planes = c/2;
        conv_0.create_weight_tensors(ctx);

        conv_1.in_planes = c/2;
        conv_1.out_planes = c;
        conv_1.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        conv_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "1.");
        conv_1.setup_weight_names(s);                
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv_0.forward(ctx, x);
        x = conv_1.forward(ctx, x);
        return x;
    }
};

struct IFBlockLast {
    int c;

    struct ConvTranspose2d conv_0;
    struct PixelShuffle shuf;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv_0.in_channels = c;
        conv_0.out_channels = 4*6;
        conv_0.kernel_size = 4;
        conv_0.stride = 2;
        conv_0.padding = 1;
        conv_0.create_weight_tensors(ctx);

        shuf.upscale_factor = 2;
        shuf.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        conv_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "1.");
        shuf.setup_weight_names(s);                
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv_0.forward(ctx, x);
        x = shuf.forward(ctx, x);
        return x;
    }
};

struct IFBlock {
    int in_planes;
    int c;

    struct IFBlockFirst conv0;
    struct ResConv convblocks[8];
    struct IFBlockLast lastconv;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv0.in_planes = in_planes;
        conv0.c = c;
        conv0.create_weight_tensors(ctx);

        for (int i = 0; i < 8; i++) {
            convblocks[i].c = c;
            convblocks[i].create_weight_tensors(ctx);
        }

        lastconv.c = c;
        lastconv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv0.");
        conv0.setup_weight_names(s);

        for (int i = 0; i < 8; i++) {
            snprintf(s, sizeof(s), "%sconvblock.%d.", prefix, i);
            convblocks[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "lastconv.");
        lastconv.setup_weight_names(s);
    }

    ggml_tensor_t* resize(struct ggml_context* ctx, ggml_tensor_t* x, float scale) {
        if (fabsf(scale - 1.0) < 1e-5)
            return x;

        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];
        W = int(W * scale);
        H = int(H * scale);
        x = ggml_upscale_ext(ctx, x, W, H, C, B);

        return x;
    }

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* start_flow, float scale) {
        // x = resize(x, 1.0 / scale)

        // """flow is not None"""
        // if flow is not None:
        //     flow = resize(flow, 1.0 / scale) / scale
        //     x = torch.cat((x, flow), dim=1)

        // feat = self.conv0(x)
        // feat = self.convblock(feat)
        // feat = self.lastconv(feat)
        // feat = resize(feat, scale)
        // flow = feat[:, 0:4] * scale
        // mask = feat[:, 4:5]
        // return flow, mask
        std::vector<ggml_tensor_t *> flow_mask_list;
        x = resize(ctx, x, 1.0/scale);
  
        if (start_flow != NULL) {
            start_flow = resize(ctx, start_flow, 1.0/scale);
            start_flow = ggml_scale(ctx, start_flow, 1.0/scale);
            x = ggml_concat(ctx, x, start_flow, 2/*dim on channel*/);
        }
        ggml_tensor_t *feat, *flow, *mask;
        feat = conv0.forward(ctx, x);
        for (int i = 0; i < 8; i++) {
            feat = convblocks[i].forward(ctx, feat);
        }

        feat = lastconv.forward(ctx, feat);
        feat = resize(ctx, feat, scale);
        // # tensor [feat] size: [1, 6, 1056, 1056], min: -10.310724, max: 2.952527, mean: -0.450135

        flow = ggml_nn_slice(ctx, feat, 2 /*dim on C*/, 0, 4, 1/*step*/);
        flow = ggml_scale(ctx, flow, scale);
        mask = ggml_nn_slice(ctx, feat, 2 /*dim on C*/, 4, 5, 1/*step*/);

        flow_mask_list.push_back(flow);
        flow_mask_list.push_back(mask);

      	return flow_mask_list;
    }
};

struct IFNet : GGMLNetwork {
    int MAX_H = 2048;
    int MAX_W = 2048;
    int MAX_TIMES = 32;

    struct IFBlock block0;
    struct IFBlock block1;
    struct IFBlock block2;
    struct IFBlock block3;

    struct Head encode;

    void create_weight_tensors(struct ggml_context* ctx) {
        block0.in_planes = 7 + 16;
        block0.c = 192;
        block0.create_weight_tensors(ctx);

        block1.in_planes = 8 + 4 + 16;
        block1.c = 128;
        block1.create_weight_tensors(ctx);

        block2.in_planes = 8 + 4 + 16;
        block2.c = 96;
        block2.create_weight_tensors(ctx);

        block3.in_planes = 8 + 4 + 16;
        block3.c = 64;
        block3.create_weight_tensors(ctx);

        encode.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "block0.");
        block0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "block1.");
        block1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "block2.");
        block2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "block3.");
        block3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "encode.");
        encode.setup_weight_names(s);
    }

    ggml_tensor_t* resize_pad(struct ggml_context* ctx, ggml_tensor_t* x) {
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];

        if (H > MAX_H || W > MAX_W) { // need resize ?
            float s = (float)MAX_H/H;
            if (s < (float)MAX_W/W) {
                s = (float)MAX_W/W;
            }
            int SH = s * H; // new width
            int SW = s * W; // new height
            x = ggml_interpolate(ctx, x, 1 /*dim on H */, SH);
            x = ggml_interpolate(ctx, x, 0 /*dim on W */, SW);
        }

        // Need pad ?        
        W = (int)x->ne[0];
        H = (int)x->ne[1];
        int r_pad = (MAX_TIMES - (W % MAX_TIMES)) % MAX_TIMES;
        int l_pad = r_pad/2; r_pad = r_pad - l_pad;
        int b_pad = (MAX_TIMES - (H % MAX_TIMES)) % MAX_TIMES;
        int t_pad = b_pad/2; b_pad = b_pad - t_pad;

        if (l_pad > 0 || r_pad > 0 || t_pad > 0 || b_pad > 0) {
            x = ggml_replication_pad2d(ctx, x, l_pad, r_pad, t_pad, b_pad);
        }

        return x;
    }

    ggml_tensor_t* make_grid(struct ggml_context* ctx, int B, int H, int W) {
        // hg = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        // vg = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        // # hg.size() -- [1, 1, 1056, 1056]
        // # vg.size() -- [1, 1, 1056, 1056]

        // grid = torch.cat([hg, vg], dim=1)
        ggml_tensor_t* x = ggml_grid_mesh(ctx, B, H, W, 1/*norm*/);
        x = ggml_scale(ctx, x, 2.0);
        x = ggml_add_constant(ctx, x, -1.0);

        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [2, W, H, B] -> [W, H, 2, B]
        return x;
    }

    ggml_tensor_t* warp(struct ggml_context* ctx, ggml_tensor_t *feat, ggml_tensor_t *flow, ggml_tensor_t *grid) {
        int W = (int)feat->ne[0];
        int H = (int)feat->ne[1];

        ggml_tensor_t *flow_x = ggml_nn_slice(ctx, flow, 2/*dim on C*/, 0, 1, 1/*step*/);
        ggml_tensor_t *flow_y = ggml_nn_slice(ctx, flow, 2/*dim on C*/, 1, 2, 1/*step*/);
        flow_x = ggml_scale(ctx, flow_x, 2.0f/(W - 1.0));
        flow_y = ggml_scale(ctx, flow_y, 2.0f/(H - 1.0));
        flow = ggml_concat(ctx, flow_x, flow_y, 2/*dim*/);

        ggml_tensor_t *g = ggml_add(ctx, grid, flow);
        g = ggml_permute(ctx, g, 1, 2, 0, 3); // [W, H, 2, B] -> [2, W, H, B]

        return ggml_grid_sample(ctx, feat, g);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        float scale_list[4] = {8.0, 4.0, 2.0, 1.0 };
        ggml_tensor_t *I1, *I2, *timestep;
        std::vector<ggml_tensor_t *>flow_mask_list;

        GGML_UNUSED(argc);
        I1 = argv[0];
        I2 = argv[1];
        timestep = argv[2];

        int W2 = (int)I1->ne[0];
        int H2 = (int)I1->ne[1];
        int C2 = (int)I1->ne[2];
        int B2 = (int)I1->ne[3];

        I1 = resize_pad(ctx, I1);
        I2 = resize_pad(ctx, I2);
        timestep = resize_pad(ctx, timestep);
        // tensor [I1] size: [1, 3, 1024, 1024], min: 0.0, max: 1.0, mean: 0.483912
        // tensor [I2] size: [1, 3, 1024, 1024], min: 0.0, max: 1.0, mean: 0.484398
        // tensor [timestep] size: [1, 1, 1024, 1024], min: 0.5, max: 0.5, mean: 0.5

        int W = (int)I1->ne[0];
        int H = (int)I1->ne[1];
        // int C = (int)I1->ne[2];
        int B = (int)I1->ne[3];

        ggml_tensor_t *F1, *F2, *xx, *flow, *mask, *W_F1, *W_F2, *W_I1, *W_I2;
        F1 = encode.forward(ctx, I1);
        F2 = encode.forward(ctx, I2);
        // tensor [F1] size: [1, 8, 1024, 1024], min: -2.54316, max: 2.194751, mean: -0.084768
        // tensor [F2] size: [1, 8, 1024, 1024], min: -2.54316, max: 2.194751, mean: -0.075074

        xx = ggml_cat(ctx, 5, I1, I2, F1, F2, timestep, 2/*dim on channel*/);
        // tensor [xx] size: [1, 23, 1024, 1024], min: -2.54316, max: 2.194751, mean: 0.092443

        // flow, mask = self.block0.forward(xx, None, scale=scale_list[0])
        flow_mask_list = block0.forward(ctx, xx, NULL /*flow*/, scale_list[0]);
        flow = flow_mask_list[0];
        mask = flow_mask_list[1];
        // tensor [flow] size: [1, 4, 1024, 1024], min: -10.025298, max: 11.566859, mean: 0.010115
        // tensor [mask] size: [1, 1, 1024, 1024], min: -1.444131, max: 1.61599, mean: 0.053681

        W_I1 = I1;
        W_I2 = I2;
        ggml_tensor_t *grid = make_grid(ctx, B, H, W);
        // tensor [grid] size: [1, 2, 1024, 1024], min: -1.0, max: 1.0, mean: 0.0

        struct IFBlock *blocks[] = { &block1, &block2, &block1 };
        ggml_tensor_t *flow1, *flow2, *fd;
        for (int i = 0; i < 3; i++) {
            flow1 = ggml_nn_slice(ctx, flow, 2/*dim*/, 0, 2, 1/*step*/);
            flow2 = ggml_nn_slice(ctx, flow, 2/*dim*/, 2, 4, 1/*step*/);

            W_F1 = warp(ctx, F1, flow1, grid);
            W_F2 = warp(ctx, F2, flow2, grid);

            xx = ggml_cat(ctx, 6, W_I1, W_I2, W_F1, W_F2, timestep, mask, 2/*dim*/);

            flow_mask_list = blocks[i]->forward(ctx, xx, flow, scale_list[i + 1]);
            fd = flow_mask_list[0];
            mask = flow_mask_list[1];
            flow = ggml_add(ctx, flow, fd);

            flow1 = ggml_nn_slice(ctx, flow, 2/*dim*/, 0, 2, 1/*step*/);
            flow2 = ggml_nn_slice(ctx, flow, 2/*dim*/, 2, 4, 1/*step*/);
            W_I1 = warp(ctx, I1, flow1, grid);
            W_I2 = warp(ctx, I2, flow2, grid);
        }
        // tensor [W_I1] size: [1, 3, 1024, 1024], min: 2.8e-05, max: 1.0, mean: 0.483843
        // tensor [W_I2] size: [1, 3, 1024, 1024], min: 3.4e-05, max: 1.0, mean: 0.484579

        ggml_tensor_t *one_mask;
        {
            mask = ggml_sigmoid(ctx, mask);
            one_mask = ggml_dup(ctx, mask);
            one_mask = ggml_constant(ctx, one_mask, 1.0);
            one_mask = ggml_sub(ctx, one_mask, mask);
            // tensor [mask] size: [1, 1, 1024, 1024], min: 1.5e-05, max: 1.0, mean: 0.546369
        }

        // middle = W_I1 * mask + W_I2 * (1.0 - mask)
        ggml_tensor_t *middle;
        middle = ggml_add(ctx, ggml_mul(ctx, W_I1, mask), ggml_mul(ctx, W_I2, one_mask));
        // tensor [middle] size: [1, 3, 1024, 1024], min: 6.9e-05, max: 1.0, mean: 0.484024

        if (middle->ne[0] != W2 || middle->ne[1] != H2) {
            middle = ggml_nn_slice(ctx, middle, 0, 0, W2, 1/*step*/);
            middle = ggml_nn_slice(ctx, middle, 1, 0, H2, 1/*step*/);
        }

      	return middle;
    }
};

struct VideoSlowNetwork {
    IFNet net;
    GGMLModel model;

    int init(int device) {
        // -----------------------------------------------------------------------------------------
        net.set_device(device);
        net.start_engine();
        // net.dump();

        check_point(model.preload("models/video_slow_f16.gguf") == RET_OK);

        return RET_OK;
    }

    int load() {
        return net.load_weight(&model, "");
    }

    TENSOR *forward(TENSOR *input1_tensor, TENSOR *input2_tensor, TENSOR *timestep) {
        TENSOR *argv[3];
        argv[0] = input1_tensor ;
        argv[1] = input2_tensor ;
        argv[2] = timestep ;

        load();
        return net.engine_forward(ARRAY_SIZE(argv), argv);
    }

    void exit() {
        model.clear();
        net.stop_engine();
    }
};


#endif // __RIFE__H__
