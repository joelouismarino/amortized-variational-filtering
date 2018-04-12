model_config = {
    'architecture': 'vrnn',
    'inference_procedure': 'direct',
    'modified': False
}

## SVG
if model_config['architecture'].lower() == 'svg':
    model_config['model_type'] = 'bair_robot_pushing'

## VRNN
if model_config['architecture'].lower() == 'vrnn':
    model_config['model_type'] = 'timit'
