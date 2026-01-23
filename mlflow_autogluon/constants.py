from types import MappingProxyType

"""Constants for AutoGluon MLflow flavor."""

FLAVOR_NAME = 'autogluon'
AUTODEPLOY_SUBPATH = 'model'
AUTODEPLOY_METADATA_FILE = 'autogluon_metadata.json'

MODEL_PACKAGES = MappingProxyType({
    'tabular': 'autogluon.tabular',
    'multimodal': 'autogluon.multimodal',
    'vision': 'autogluon.vision',
    'timeseries': 'autogluon.timeseries',
})
