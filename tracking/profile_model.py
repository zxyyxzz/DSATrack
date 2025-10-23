import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='0'


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='grm_stu', choices=['grm', 'grm_stu'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='vitb_d8', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate_vit(model, template, search, templates_num):
    '''Speed Test'''
    template_show = torch.cat((template.unsqueeze(0), template.unsqueeze(0), template.unsqueeze(0)), dim=0)
    macs1, params1 = profile(model, inputs=(template_show, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        instant_frame = template

        # overall
        # 预热阶段
        for i in range(T_w):
            _ = model(template, search)
        # from torch.profiler import record_function, ProfilerActivity
        # with torch.profiler.profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     profile_memory=True, record_shapes=True
        # ) as prof:
        start = time.time()
        # 正式测试阶段
        for i in range(T_t):
            if i <= 100:
                interval = 5
            elif i > 100 and i <= 200:
                interval = 10
            elif i > 200 and i <= 300:
                interval = 20
            elif i > 300 and i <= 400:
                interval = 40
            elif i > 400 and i <= 500:
                interval = 80
            elif i > 500:
                interval = 160

            if i%interval==0:
                new_template = template
            elif i!=0:
                new_template = None

            _ = model(new_template, search)
        torch.cuda.synchronize()
        end = time.time()
        # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # prof.export_chrome_trace(f"trace_rank{rank}.json")
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1000))
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    templates_num = cfg.TEST.TEMPLATE_NUMBER

    if args.script == "grm":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_grm
        model = model_constructor(cfg, training=False)
        # get the template and search
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        # transfer to device
        model = model.to(device).eval()
        template = template.to(device)
        search = search.to(device)

        search = [search]

        merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER
        if merge_layer <= 0:
            evaluate_vit(model, template, search, templates_num)
    
    elif args.script == "grm_stu":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_grm_stu
        model = model_constructor(cfg, training=False)
        # get the template and search
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        # transfer to device
        model = model.to(device).eval()
        template = template.to(device)
        search = search.to(device)

        search = [search]

        evaluate_vit(model, template, search, templates_num)

    else:
        raise NotImplementedError
