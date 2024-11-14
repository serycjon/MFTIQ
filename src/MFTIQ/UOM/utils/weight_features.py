import einops

def features_weighted_sum(features_list, weight_list):
    n_channels = features_list[0].shape[1]
    if n_channels != 1:
        weight_list = [einops.repeat(weight_list[i], 'B C H W -> B (C N) H W', N=n_channels, C=1) for i in range(len(weight_list))]

    # TODO: Investigate/validate: Does this approach affects autograd?
    assert features_list[0].shape == weight_list[0].shape
    data_out = features_list[0] * weight_list[0]
    for i in range(1, len(weight_list)):
        assert features_list[i].shape == weight_list[i].shape
        data_out = data_out + features_list[i] * weight_list[i]
    return data_out