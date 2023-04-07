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
    * - Extract median values
      - 0.7s +/- 0.1s
      - 9.1s +/- 0.2s
    * - Plot median values
      - 59ms +/- 3ms
      - 59ms +/- 3ms
    * - Convert to ``qp.histogram`` representation
      - 0.04s +/- 0.01s
      - 8.1s +/- 0.1s
    * - Convert to ``qp.interp`` representation
      - Already ``qp.interp``
      - 10.6s +/- 0.1s

Other considerations
--------------------

While additional parameters that will affect storage size come into play 
(the size of the basis set, the resolution of the x grid provided, 
the number of CDEs stored) the measurements shown in the table above 
are derived from a reasonable set of parameters a user might select.

Upon further exploration, it appears that XGBoost, the modeling tool that 
``Flexcode`` uses, allows at most 38 basis functions. 
Any basis functions requested beyond that will be ignored. 
Thus for comparison above, the largest output file size for 20,449 distributions 
will be approximately 3.3M for the ``qp_flexzboost`` representation.

To probe the storage requirements for the baseline, the number of distributions 
was held constant at 20,449, while varying the number of x values. 
As expected, decreasing the number of x grid points from 301 to 101 (roughly 1/3) 
reduced the storage by 1/3 to 16M. 
Additionally, reducing this x grid points to 20 produced output files 3.3M in size.
It is feasible to reduce the size further with fewer x values, however the 
fidelity of the reconstructed distributions begins to suffer. 

It is also important to note that ``qp_flexzboost`` represented distributions 
allow the user to manipulate the post processing parameters without the need to 
rerun the model - x,y interpolated representation do not permit that kind of 
manipulation. 
Additionally ``qp_flexzboost`` represented distributions are lossless, whereas
the x,y interpolated representations become more lossy as the storage size 
approaches that of ``qp_flexzboost``.

The x,y interpolated representation is significantly more performant when 
operating on the data. One potential workflow may be to store the data in the 
native ``qp_flexzboost`` representation, and then convert to an interpolated 
grid when performing operations on it.

Converting the ``qp_flexzboost`` representation to x,y interpolated using with 
a grid = ``np.linspace(0.0, 3.0, 301)`` (to match the baseline) took approximately 
11 seconds. 

Given these all of these considerations, it is up to the user to determine when 
to use each storage representation.

If compact storage is required, or if the ability to experiment with different 
bump threshold or sharpening alpha parameters is needed, then using 
``qp_flexzboost`` makes sense.

Alternatively, if efficient storage is a lower priority compared to computation 
speed, the x,y interpolated representation is a better choice.

Finally, depending on the computation requirements, it may be practical to 
store the data using the compact ``qp_flexzboost`` representation, then unpack 
it into a x,y interpolated representation while working with it.

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
