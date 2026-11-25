import re
from copy import deepcopy

import yaml

_FIELDS = {
    'raster_numerical': ('path',),
    'raster_categorical': ('path', 'categories'),
    'raster_numerical_categorized': ('path_numerical', 'path_categorical', 'categories'),
    'vector_line': ('path',),
    'vector_point': ('path',),
    'vector_polygon': ('path',),
    'double_proxy': ('base', 'second'),
}
_AREA_TYPES = {'raster_numerical', 'raster_categorical', 'raster_numerical_categorized', 'vector_polygon'}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys rather than silently replacing proxy definitions."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError('Proxy catalogue mapping keys must be strings')
            if key in mapping:
                raise ValueError(f'Duplicate proxy catalogue key: {key!r}')
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def load_proxy_catalogue(path, input_dir=None, data_path=None):
    """Read definitions without opening raster/vector data or choosing an NES API."""
    if path in (None, '', 'None'):
        return {}
    with open(path, encoding='utf-8') as stream:
        document = yaml.load(stream, Loader=_UniqueKeyLoader)
    if not isinstance(document, dict) or not isinstance(document.get('PROXIES'), dict):
        raise ValueError('Proxy catalogue must contain a PROXIES mapping')
    catalogue = document['PROXIES']
    for name, definition in catalogue.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
            raise ValueError(f'Invalid proxy name: {name!r}')
        if (not isinstance(definition, dict) or not isinstance(definition.get('type'), str)
                or definition['type'] not in _FIELDS):
            raise ValueError(f'Invalid proxy type or definition for {name!r}')
        for field in _FIELDS[definition['type']]:
            if field not in definition:
                raise ValueError(f'Proxy {name!r} requires {field!r}')
        if 'categories' in definition:
            categories = definition['categories']
            if not isinstance(categories, list) or not categories or any(type(code) is not int for code in categories):
                raise ValueError(f'Proxy {name!r} requires a non-empty integer categories list')
        for field in ('path', 'path_numerical', 'path_categorical', 'weight_column'):
            if field not in definition:
                continue
            value = definition[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Proxy {name!r} requires a non-empty string for {field!r}')
            if field != 'weight_column':
                for placeholder, replacement in (('<input_dir>', input_dir), ('<data_path>', data_path)):
                    if replacement is not None:
                        value = value.replace(placeholder, replacement)
                definition[field] = value
    return catalogue


def select_proxies(value, catalogue):
    """Return ordered named definitions, including double-proxy components."""
    if value is None or not isinstance(value, str) or not value.strip():
        return []

    def resolve(name):
        if name not in catalogue:
            raise ValueError(f'Unknown proxy {name!r}; names are case-sensitive')
        definition = deepcopy(catalogue[name])
        definition['name'] = name
        if definition['type'] == 'double_proxy':
            for field in ('base', 'second'):
                component = definition[field]
                if not isinstance(component, str) or component not in catalogue:
                    raise ValueError(f'Proxy {name!r} has an invalid {field!r} reference')
                component_type = catalogue[component]['type']
                if component_type == 'double_proxy':
                    raise ValueError('Nested double proxies are pending a design decision')
                if field == 'second' and component_type not in _AREA_TYPES:
                    raise ValueError(f'Proxy {name!r}: second cannot be a point or line proxy')
                definition[field] = resolve(component)
        return definition

    return [resolve(name.strip()) for name in value.split(',') if name.strip()]


def to_nes_proxies(proxies):
    """
    Helper function to convert ordered catalogue definitions of proxies defined from input CSV and
    proxy catalogue to NES priority-list entries.
    """
    types = {
        'raster_numerical': 'raster',
        'raster_categorical': 'raster_categorical',
        'raster_numerical_categorized': 'raster_categorical_numerical',
        'vector_line': 'lines',
        'vector_point': 'points',
        'vector_polygon': 'polygons',
    }
    result = []

    # Iterate over each proxy definition and convert it to the NES format.
    for definition in proxies:
        proxy_type = definition['type']
        if proxy_type == 'double_proxy':
            raise ValueError(f"Double proxy {definition['name']!r} is not supported for interpolation")
        if proxy_type not in types:
            raise ValueError(f"Unsupported proxy type: {proxy_type!r}")
        if proxy_type == 'raster_numerical_categorized':
            path = (definition['path_categorical'], definition['path_numerical'])
        else:
            path = definition['path']
        categories = definition.get('categories') if proxy_type.startswith('raster_') else None
        if proxy_type.startswith('vector_'):
            result.append((types[proxy_type], path, None, definition.get('weight_column', 'weight')))
        else:
            result.append((types[proxy_type], path, categories))
    return result
