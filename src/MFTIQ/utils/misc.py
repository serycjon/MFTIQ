import numpy as np
import torch
import inspect
import datetime

def ensure_numpy(xs):
    if isinstance(xs, torch.Tensor):
        return xs.detach().cpu().numpy()
    else:
        return xs


def ensure_torch(xs):
    if not isinstance(xs, torch.Tensor):
        return torch.from_numpy(xs)
    else:
        return xs


def parse_scale_WH(scale_WH, frames_shape):
    """ Parse string argument scale_WH and scale img_shape according to scale_WH.
        Missing value will be computed keeping same ratio.

    Params: scale_WH       - string with resolution WxH, examples: fullres, 256x256, x1080, 512x,
                             if there is "_" in the scale_WH, the output will be sequence of rescaling operations:
                             i.e. 256x256_x480 -> first rescale to 256x256 then to x480
            frames_shape   - dict with current's frame resolution with keys W and H
    Output: new_shape_list - list of dicts with scaled resolution, video will be rescaled multiple times according
                             the new resolutions in list
    """
    if scale_WH == 'fullres':
        return [frames_shape]
    new_shape_list = []
    scale_WH_split = scale_WH.split('_')
    for c_scale_WH in scale_WH_split:
        if c_scale_WH == 'fullres':
            new_shape_list.append(frames_shape)
            continue
        new_shape = dict(frames_shape.items())
        W_str, H_str = c_scale_WH.split('x')
        W = int(W_str) if W_str != '' else None
        H = int(H_str) if H_str != '' else None
        assert W is not None or H is not None, 'at least one dimmension has to be set'
        new_shape['W'] = W if W is not None else int(round(frames_shape['W'] * (H / frames_shape['H'])))
        new_shape['H'] = H if H is not None else int(round(frames_shape['H'] * (W / frames_shape['W'])))
        new_shape_list.append(new_shape)
    return new_shape_list


def dummy_profile():
    """A possibly dummy profile decorator.

    For use with line_profiler.
    When called with: kernprof -l script... it will use the profile decorator,
    otherwise it will return an empty dummy
    """
    def dummy_decorator(fun):
        return fun

    try:
        from line_profiler import profile
        return profile
    except ImportError:
        return dummy_decorator

_debounce = {}

def debounce(seconds):
    """Debounce: return True when at least 'seconds' seconds has passed since last True

    the debouncing is based on the filename and line number where the debounce function is called.
    Can be used like:
    while True:
        debounce(5) and print('Only printing every 5 seconds')
    """
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    caller_line = caller_frame.lineno

    key = f'{caller_file}:{caller_line}'

    last_time = _debounce.get(key, datetime.datetime(2000, 1, 1))
    now = datetime.datetime.now()

    diff = now - last_time
    if diff < datetime.timedelta(seconds=seconds):
        # print(f'Debouncer @ {key} - {diff} - too soon')
        return False
    else:
        # print.debug(f'Debouncer @ {key} - {diff} - OK')
        _debounce[key] = now
        return True



    
