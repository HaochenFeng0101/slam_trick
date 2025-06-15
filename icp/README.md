

## ICP Algorithm Logic


### 1. Initialization

* **Input Clouds:** The algorithm takes two `pcl::PointCloud<pcl::PointXYZ>` as input: `source_cloud_initial` (the cloud to be transformed) and `target_cloud` (the reference cloud).
* **Current Transformed Cloud:** A copy of the initial source cloud, `transformed_source_cloud`, is created. We'll iteratively update this cloud to bring it closer to the target.
* **Accumulated Transformation:** An `Eigen::Matrix4f` called `current_total_transform` is initialized to an **identity matrix**, $I_4$. This matrix will accumulate all the incremental transformations found in each iteration, eventually representing the final transformation from the `source_cloud_initial` to the `target_cloud`.
* **Kd-Tree for Target:** A `pcl::KdTreeFLANN<pcl::PointXYZ>` is built once on the `target_cloud`. This data structure allows for efficient nearest neighbor searches, which is crucial for finding correspondences quickly.

### 2. Iteration Loop

The ICP algorithm runs for a `max_iterations` or until a convergence criterion is met. Each iteration performs the following steps:

#### a. Correspondence Search (Nearest Neighbor)

For each point $p_i \in \text{transformed\_source\_cloud}$ (where $p_i$ is a 3D point represented as $\begin{pmatrix} x \\ y \\ z \end{pmatrix}$):
* We perform a **nearest neighbor search** in the **`target_cloud`** using the pre-built Kd-Tree to find the closest point, let's call it $q_k \in \text{target\_cloud}$.
* A correspondence pair $(p_i, q_k)$ is considered **valid** only if the squared Euclidean distance between $p_i$ and $q_k$ is less than or equal to `max_correspondence_distance^2`:
    $$\|p_i - q_k\|^2 \le \text{max\_correspondence\_distance}^2$$
    This `max_correspondence_distance` acts as a crucial outlier rejection mechanism, preventing distant points from forming incorrect correspondences.
* All valid $(p_i, q_k)$ pairs form the `correspondences` list for the current iteration.
* If no correspondences are found, the ICP process stops, as further alignment isn't possible with the current parameters.

#### b. Transformation Estimation (SVD / Kabsch Algorithm)

The `estimateTransformationSVD` function takes the list of $(p_i, q_k)$ correspondences (where $p_i$ is from the *current* `transformed_source_cloud` and $q_k$ is its target correspondent) and computes a $4 \times 4$ homogeneous transformation matrix ($\text{transformation\_delta}$). This $\text{transformation\_delta}$ represents the optimal rigid transformation that maps the `transformed_source_cloud` to the `target_cloud` based on the current correspondences.

Here's how `estimateTransformationSVD` works:

1.  **Calculate Centroids:** We compute the centroids ($\bar{p}$, $\bar{q}$) of the corresponding point sets $P = \{p_i\}$ and $Q = \{q_k\}$:
    $$\bar{p} = \frac{1}{N} \sum_{i=1}^{N} p_i \quad \text{and} \quad \bar{q} = \frac{1}{N} \sum_{i=1}^{N} q_k$$
    where $N$ is the number of valid correspondences.
2.  **Center the Point Sets:** All points in both sets are translated so that their respective centroids are at the origin:
    $$p'_i = p_i - \bar{p} \quad \text{and} \quad q'_k = q_k - \bar{q}$$
    This temporarily removes the translational component, allowing us to focus on rotation.
3.  **Compute Covariance Matrix H:** A $3 \times 3$ covariance matrix $H$ is calculated as the sum of outer products of the centered points:
    $$H = \sum_{i=1}^{N} p'_i (q'_k)^T$$
4.  **Singular Value Decomposition (SVD):** SVD is performed on $H$:
    $$H = U S V^T$$
    where $U$ and $V$ are orthogonal matrices and $S$ is a diagonal matrix of singular values.
5.  **Compute Rotation Matrix R:** The rotation matrix $R$ is derived from $U$ and $V$:
    $$R = V U^T$$
    A check is included for reflections (if $\det(R) < 0$). If a reflection is detected, the last column of $V$ is negated, and $R$ is recomputed to ensure it's a pure rotation.
6.  **Compute Translation Vector t:** The translation vector $t$ is calculated using the centroids and the rotation matrix:
    $$t = \bar{q} - R \bar{p}$$
7.  **Assemble Transformation Matrix:** Finally, $R$ and $t$ are combined into a $4 \times 4$ homogeneous transformation matrix $\text{T\_matrix}$:
    $$\text{T\_matrix} = \begin{pmatrix} R & t \\ 0^T & 1 \end{pmatrix}$$
    This $\text{T\_matrix}$ is the $\text{transformation\_delta}$ for the current iteration.

#### c. Apply Transformation

* The $\text{transformation\_delta}$ calculated in the previous step is applied directly to the `transformed_source_cloud` using `pcl::transformPointCloud`. This moves the source cloud closer to the target cloud.

#### d. Accumulate Transformation

* The `current_total_transform` (which tracks the total transformation from the initial source cloud to its current position) is updated by pre-multiplying it with the $\text{transformation\_delta}$:
    $$\text{current\_total\_transform} = \text{transformation\_delta} \times \text{current\_total\_transform}$$
    This correctly updates the accumulated transformation such that an `initial_source_point` transformed by $\text{current\_total\_transform}$ will yield the point in the *current* `transformed_source_cloud`'s frame.

#### e. Convergence Check

* The algorithm checks if the ICP process has converged. This is done by calculating the **Frobenius norm** of the difference between the $\text{transformation\_delta}$ and an identity matrix, $I_4$:
    $$\text{transform\_change} = \|\text{transformation\_delta} - I_4\|_F$$
* If this $\text{transform\_change}$ is less than a predefined `transformation_epsilon`, it means the transformation in the current iteration is very small, indicating that the clouds are sufficiently aligned, and the ICP loop terminates.

### 3. Output

* Upon convergence or reaching `max_iterations`, the `current_total_transform` matrix is returned. This is the final transformation matrix that aligns the `source_cloud_initial` to the `target_cloud`.
* The script then applies this final transformation to the original `source_cloud_initial` to generate an `aligned_source_cloud`, which is saved as a PCD file.

---

## Dependencies

This implementation relies on the following libraries:

* **Eigen:** For linear algebra operations, especially matrix and vector manipulations and SVD.
* **PCL (Point Cloud Library):** For point cloud data structures, Kd-Tree implementation, and point cloud I/O and transformations.
* **spdlog:** logging