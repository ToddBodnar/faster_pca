# faster_pca
PCA filter for Weka with optimization for large datasets

Right now this only works for weka's PCA's "centered" setting (normalize based on mean) and not the "normalized" setting (transform the data so min=-1 and max=1).

To use it in code, just construct a `faster_pca.faster_pca` instead of a `weka.attributeSelection.PrincipalComponents`. It's an extension of PrincipalComponents, so all of the configuration works the same.

You can run faster_pca.test.main() for a speed test. The results are below. Note that this is a stopwatch timer, which isn't necessarily the most accurate method. The 40 accuracy tests before the timing tests may help controll for JIT optimization. Also note that it's unlikely that the same PCA model would be built many times in a real world application. Run tests.calcTimeDistributions() to run 100 iterations of each model, if you want more accurate statistics (although this will take a while).

```
Running tests of faster_pca

In accuracy tests
Comparing centered results from PCA and faster_pca (1/10)	Mean Sq Difference: 1.3143370490992766E-24	PASS
Comparing centered results from PCA and faster_pca (2/10)	Mean Sq Difference: 1.3143370490992766E-24	PASS
Comparing centered results from PCA and faster_pca (3/10)	Mean Sq Difference: 1.3143370490992766E-24	PASS
Comparing centered results from PCA and faster_pca (4/10)	Mean Sq Difference: 1.3143370490992766E-24	PASS
Comparing centered results from PCA and faster_pca (5/10)	Mean Sq Difference: 1.3143370490992766E-24	PASS
Comparing centered results from PCA and faster_pca (6/10)	Mean Sq Difference: 1.3143370490992766E-24	PASS
Comparing centered results from PCA and faster_pca (7/10)	Mean Sq Difference: 1.3143370490992766E-24	PASS
Comparing centered results from PCA and faster_pca (8/10)	Mean Sq Difference: 1.3143370490992766E-24	PASS
Comparing centered results from PCA and faster_pca (9/10)	Mean Sq Difference: 1.3143370490992766E-24	PASS
Comparing centered results from PCA and faster_pca (10/10)	Mean Sq Difference: 1.3143370490992766E-24	PASS

Comparing non-centered results from PCA and faster_pca (1/10)	Mean Sq Difference: 1.0069116622185093E-4	FAIL
Comparing non-centered results from PCA and faster_pca (2/10)	Mean Sq Difference: 1.0069116622185093E-4	FAIL
Comparing non-centered results from PCA and faster_pca (3/10)	Mean Sq Difference: 1.0069116622185093E-4	FAIL
Comparing non-centered results from PCA and faster_pca (4/10)	Mean Sq Difference: 1.0069116622185093E-4	FAIL
Comparing non-centered results from PCA and faster_pca (5/10)	Mean Sq Difference: 1.0069116622185093E-4	FAIL
Comparing non-centered results from PCA and faster_pca (6/10)	Mean Sq Difference: 1.0069116622185093E-4	FAIL
Comparing non-centered results from PCA and faster_pca (7/10)	Mean Sq Difference: 1.0069116622185093E-4	FAIL
Comparing non-centered results from PCA and faster_pca (8/10)	Mean Sq Difference: 1.0069116622185093E-4	FAIL
Comparing non-centered results from PCA and faster_pca (9/10)	Mean Sq Difference: 1.0069116622185093E-4	FAIL
Comparing non-centered results from PCA and faster_pca (10/10)	Mean Sq Difference: 1.0069116622185093E-4	FAIL



In speed tests (25.0% improvement required to pass)
Comparing to wekaPCA (no center, 100x10000). weka takes: 5606 fast pca takes: 1488	PASS
Comparing to wekaPCA (center, 100x10000). weka takes: 7442 fast pca takes: 3044	PASS

Comparing to wekaPCA (no center, 1000x10000). weka takes: 435249 fast pca takes: 270854	PASS
Comparing to wekaPCA (center, 1000x10000). weka takes: 566167 fast pca takes: 272152	PASS
Passed 14 / 24 tests.

```
