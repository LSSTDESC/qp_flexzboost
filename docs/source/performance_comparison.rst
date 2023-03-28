Performance comparison
======================

The following comparisons were observed using a basis set of 35 cosine functions, 
an x grid of 301 points, and an example data set of 20,449 distributions.
All measurements were made on a desktop computer. 
"Baseline" refers to a ``qp.Ensemble`` composed of ``qp.interp`` 
(x,y interpolated) distributions.
"qp_flexzboost" refers to a ``qp.Ensemble`` composed of ``qp_flexzboost`` 
distributions.

.. list-table:: Baseline vs. qp_flexzboost
    :widths: 50 25 25
    :header-rows: 1

    * - **Measurement**
      - **Baseline**
      - **qp_flexzboost**
    * - Storage size on disk
      - 48M
      - 2.9M (>16x smaller)
    * - Model estimation time
      - 17.4s +/- 0.3s
      - 16.8s +/- 0.1s
    * - Extract all median and mode values
      - 9.1s +/- 0.3s
      - 9.1s +/- 0.2s
    * - Plot all median values
      - 59ms +/- 3ms
      - 59ms +/- 3ms
    * - Convert to ``qp.histogram`` representation
      - 8.1s +/- 0.2s
      - 8.1s +/- 0.1s


Additional context
------------------

Prior to the creation of ``qp_flexzboost``, ``rail_flexzboost`` would store 
``Flexcode`` output as a ``qp.Ensemble`` of x,y interpolated distributions. 
This approach is computationally efficient, but also lossy, and the fidelity of 
the persisted data was proportional storage size on disk.

To make use of the x,y interpolated representations, the user must define 
a grid of x values for which corresponding y values of the ``Flexcode`` 
conditional density estimates would be stored. 
As the x resolution increases the fidelity and storage requirements also increase.

Using ``qp_flexzboost``, we store only the ``Flexcode`` output basis function 
weights for each conditional density estimate. 
This approach is lossless and there is minimal computational impact when 
compared to the baseline.

While additional parameters that will affect storage size come into play 
(the size of the basis set, the resolution of the x grid provided, 
the number of CDEs stored) the measurements shown in the table above 
are derived from a reasonable set of parameters a user might select.
