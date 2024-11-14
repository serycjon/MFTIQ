"""A PTLFlow wrapper"""
from ipdb import iex

import ptlflow
from ptlflow.utils.io_adapter import IOAdapter

from MFTIQ.utils.misc import ensure_numpy


class PTLFlowWrapper():
    def __init__(self, config):
        self.C = config

        # self.device = torch.device('cuda')
        model = ptlflow.get_model(self.C.model_name, self.C.checkpoint_name)
        model.eval()
        # model.to(self.device)

        model.requires_grad_(False)
        model.cuda()
        # model.eval()

        self.model = model

    @iex
    def compute_flow(self, src_img, dst_img, mode='TC', vis=False, numpy_out=False, **kwargs):
        """
        args:
            src_img: (H, W, 3) uint8 BGR opencv image
            dst_img: (H, W, 3) uint8 BGR opencv image
            mode: one of "flow" or "TC"
            init_flow: (xy H W) flow
        """
        assert kwargs.get('init_flow') is None

        io_adapter = IOAdapter(self.model, src_img.shape[:2], cuda=True)
        inputs = io_adapter.prepare_inputs([src_img, dst_img])

        predictions = self.model(inputs)

        flow = predictions['flows'][0, 0]  # Remove batch and sequence dimensions xy, H, W shape

        H, W = src_img.shape[:2]
        assert flow.shape == (2, H, W)

        if mode == 'flow':
            if numpy_out:
                flow = ensure_numpy(flow)

            extra_outputs = {'occlusion': None,
                             'sigma': None}
            return flow, extra_outputs
        else:
            raise NotImplementedError("TC mode not implemented")
