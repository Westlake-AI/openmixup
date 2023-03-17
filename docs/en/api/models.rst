.. role:: hidden
    :class: hidden-section

openmixup.models
===================================

The ``models`` package contains several sub-packages for addressing the different components of a model.

- :ref:`augments`: Mixup augmentations for classification models.
- :ref:`backbones`: A feature extraction network, e.g., ResNet.
- :ref:`classifiers`: The top-level module which defines the whole process of a classification model.
- :ref:`heads`: The component for specific tasks, e.g., classification and self-supervised pre-training.
- :ref:`losses`: Loss functions.
- :ref:`memories`: The memory bank component for self-supervised learning.
- :ref:`necks`: The component between backbones and heads, e.g., GlobalAveragePooling.
- :ref:`selfsup`: The top-level module for a self-supervised learning method.
- :ref:`semisup`: The top-level module for a semi-supervised learning method.

.. currentmodule:: openmixup.models

.. autosummary::
    :toctree: generated
    :nosignatures:

    build_classifier
    build_backbone
    build_neck
    build_memory
    build_head
    build_loss
    build_model

.. _augments:

Augmentations
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

.. _backbones:

Backbones
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

.. _classifiers:

Classifier
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

.. _heads:

Heads
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

.. _losses:

Losses
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

.. _memories:

Memories
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

.. _necks:

Necks
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

.. _selfsup:

Selfsup
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

.. _semisup:

Semisup
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst
