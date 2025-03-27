API
===========================================================================================

Toolkit functions
-------------------
Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: loki.preprocess
.. rubric:: Functions
.. autosummary::
   :toctree: .
   
   generate_gene_df
   segment_patches
   get_library_id
   get_scalefactors
   get_spot_diameter_in_pixels
   prepare_data_for_alignment
   load_data_for_annotation



Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: loki.utils
.. rubric:: Functions
.. autosummary::
   :toctree: .
   
   load_model
   encode_image
   encode_image_patches
   encode_text
   encode_text_df



Loki Align
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: loki.align
.. rubric:: Functions
.. autosummary::
   :toctree: .

   apply_homography
   align_tissue
   find_homography_translation_rotation


.. rubric:: Classes
.. autosummary::
   :toctree: .

   DeformableRegistration
   EMRegistration



Loki Annotate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: loki.annotate
.. rubric:: Functions
.. autosummary::
   :toctree: .

   annotate_with_bulk
   annotate_with_marker_genes
   load_image_annotation



Loki Decompose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: loki.decompose
.. rubric:: Functions
.. autosummary::
   :toctree: .

   generate_feature_ad
   cell_type_decompose



Loki Retrieve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: loki.retrieve
.. rubric:: Functions
.. autosummary::
   :toctree: .

   retrieve_st_by_image



Loki PredEx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: loki.predex
.. rubric:: Functions
.. autosummary::
   :toctree: .

   predict_st_gene_expr



Plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: loki.plot
.. rubric:: Functions
.. autosummary::
   :toctree: .
   
   plot_alignment
   plot_alignment_with_img
   plot_img_with_annotation
   plot_annotation_heatmap


