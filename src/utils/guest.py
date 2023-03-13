from collections import OrderedDict

def fix_pesr_da(modules: OrderedDict) -> OrderedDict:
    """Prepares the weights to be loaded from the original PESR module

    If DomainAdapter is created in the same way as described in the
    paper (uses pretrained normalised VGG and 6 ResNet blocks), and the
    original state dictionary is used to initialize its parameters, then
    this fixes the pretrained state dictionary to contain only the
    relevant weights for the domain adapter. In the original repository,
    weights of subsequent VGG layers were saved, even though they were
    not used in `forward` method.

    Args:
        modules (dict): The original moduels of the loaded DA checkpoint
    
    Returns:
        OrderedDict: A modified 'DA' checkpoint entry
    """
    # Init the DA modules
    modules_updated = OrderedDict()
    
    # Copy over the initial VGG encodings but change the naming
    modules_updated["vgg_encoding.0.weight"] = modules["enc_1.0.weight"]
    modules_updated["vgg_encoding.0.bias"] = modules["enc_1.0.bias"]
    modules_updated["vgg_encoding.2.weight"] = modules["enc_1.2.weight"]
    modules_updated["vgg_encoding.2.bias"] = modules["enc_1.2.bias"]

    for key in modules.keys():
        if "enc_" not in key:
            # Only keep non-enc DA modules
            modules_updated[key] = modules[key]

    return modules_updated