
def load_model(model_config):
    """
    Loads the model specified by the model config.
    """
    model_name = model_config['architecture'].lower()

    if model_name == 'vrnn':
        from vrnn import VRNN
        return VRNN(model_config)
    # elif model_name == 'srnn':
    #     from srnn import SRNN
    #     return SRNN(model_config)
    # elif model_name == 'dvbf':
    #     from dvbf import DVBF
    #     return DVBF(model_config)
    elif model_name == 'svg':
        from svg import SVG
        return SVG(model_config)
    elif model_name in ['conv', 'convolutional']:
        from convolutional import ConvDLVM
        return ConvDLVM(model_config)
    elif model_name in ['fc', 'fully_connected']:
        from fully_connected import FullyConnectedDLVM
        return FullyConnectedDLVM(model_config)
    else:
        raise Exception('Model architecture name not found.')
