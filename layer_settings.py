"""Resolve independent NR parameters without overriding pipeline/colour settings."""
import math

NR_DEFAULTS = dict(preset=1, style=0, intensity=1., local_tone=1., local_struct=1.,
                   skin_struct=1., use_auto_mask=1, ui_correction=0)


def resolve_layer_settings(settings, index):
    result = dict(settings)
    layers = settings.get('layer_settings') or []
    if not isinstance(layers, (list, tuple)):
        raise ValueError('layer_settings must be a list of NR parameter objects')
    if index > 0 and index < len(layers):
        override = layers[index]
        if not isinstance(override, dict) or override.keys() - NR_DEFAULTS.keys():
            raise ValueError('layer_settings only accepts NR model, style, strengths and mask options')
        for key, value in override.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'layer {index + 1}: {key} must be finite')
            if key == 'preset':
                valid = type(value) is int and value in (0, 1, 2, 3)
            elif key == 'style':
                valid = value in (0, 1, 2)
            elif key in ('use_auto_mask', 'ui_correction'):
                valid = value in (0, 1)
            else:
                valid = 0 <= value <= 2
            if not valid:
                raise ValueError(f'layer {index + 1}: invalid {key}')
        result.update(override)
    return result


def layer_parameters(settings):
    return {key: settings.get(key, default) for key, default in NR_DEFAULTS.items()}
