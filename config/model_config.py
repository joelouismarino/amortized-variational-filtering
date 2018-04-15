model_config = {
    'architecture': 'vrnn',
    'inference_procedure': 'error',
    'modified': True
}

## SVG
if model_config['architecture'].lower() == 'svg':
    model_config['model_type'] = 'bair_robot_pushing'

## VRNN
if model_config['architecture'].lower() == 'vrnn':
    model_config['model_type'] = 'timit'
