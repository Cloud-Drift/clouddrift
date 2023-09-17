.. currentmodule:: clouddrift

API
===

Auto-generated summary of CloudDrift's API. For more details and examples, refer to the different Jupyter Notebooks.

Adapters
--------

.. automodule:: clouddrift.adapters
  :members:
  :undoc-members:

GDP (hourly)
^^^^^^^^^^^^

.. automodule:: clouddrift.adapters.gdp1h
  :members:
  :undoc-members:

GDP (6-hourly)
^^^^^^^^^^^^^^

.. automodule:: clouddrift.adapters.gdp6h
  :members:
  :undoc-members:

MOSAiC
^^^^^^

.. automodule:: clouddrift.adapters.mosaic
  :members:
  :undoc-members:

Analysis
--------

.. automodule:: clouddrift.analysis
  :members:
  :exclude-members: chunk
  :undoc-members:

  .. autofunction:: chunk

    .. image:: img/chunk.png
      :width: 800
      :align: center
      :alt: chunk schematic

    Combined with :func:`clouddrift.analysis.apply_ragged`, :func:`clouddrift.analysis.chunk`
    can be used to divide a ragged array into equal chunks.

    .. image:: img/chunk_ragged.png
      :width: 800
      :align: center
      :alt: ragged array chunk schematic

Datasets
--------

.. automodule:: clouddrift.datasets
  :members:
  :undoc-members:

RaggedArray
-----------

.. automodule:: clouddrift.raggedarray
  :members:
  :undoc-members:

Sphere
---------

.. automodule:: clouddrift.sphere
  :members:
  :undoc-members:

Signal
---------

.. automodule:: clouddrift.signal
  :members:
  :undoc-members:

Wavelet
---------

.. automodule:: clouddrift.wavelet
  :members:
  :undoc-members: