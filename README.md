# ClusteringMetrics

I have developed some important clustering evaluation indexes in Spark 2.0.2,
using the Scala API. These indexes can help us to choose the best number of "k"
for our dataset. 

The internal indexes developed have been:

* Ball and Hall Index (Ball Index) - 1965
* Calinski and Harabasz Index (CH Index) - 1974
* Davies and Boulding Index (DB Index) - 1979
* Hartigan Index (Hartigan Index) - 1975
* Krzanowski and Lai Index (KL Index) - 1988
* Ratkowsky and Lance Index (Ratkowsky Index) - 1978

These indexes have been tested using the Iris dataset, and the R package NbClust.

I also have developed an external index (useful when you have some information 
about any elements that must go in the same group or in different groups):

* Rand Index (Rand Index) - 1971

Any suggestion or fix will be appreciated.

You can contact with me using my email: daniel.tizon@gmail.com

Thanks!
