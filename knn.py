"""Student implementation for Lab 03: k-NN classification.

Contains the three distance functions from Lab 02, plus label prediction
and the KnnClassifier class.
"""
import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses a naive set of nested loops over the training and
    test data.

    The input data may have any number of dimensions -- for example this
    function should be able to compute nearest neighbor between vectors, in
    which case the inputs will have shape (num_{train, test}, D); it should
    also be able to compute nearest neighbors between images, where the inputs
    will have shape (num_{train, test}, C, H, W). More generally, the inputs
    will have shape (num_{train, test}, D1, D2, ..., Dn); we flatten
    each element of shape (D1, D2, ..., Dn) into a vector of shape
    (D1 * D2 * ... * Dn) before computing distances.

    The input tensors are not modified.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It has the same dtype as x_train.
    """
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    # Flatten every sample into a vector (views, inputs are not modified).
    train_flat = x_train.reshape(num_train, -1)
    test_flat = x_test.reshape(num_test, -1)
    for i in range(num_train):
        for j in range(num_test):
            diff = train_flat[i] - test_flat[j]
            dists[i, j] = (diff * diff).sum()
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses only a single loop over the training data.

    Similar to `compute_distances_two_loops`, this handles inputs with any
    number of dimensions. The inputs are not modified.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It has the same dtype as x_train.
    """
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    train_flat = x_train.reshape(num_train, -1)
    test_flat = x_test.reshape(num_test, -1)
    for i in range(num_train):
        # (num_test, D) - (D,) broadcasts over all test samples at once.
        diff = test_flat - train_flat[i]
        dists[i] = (diff * diff).sum(dim=1)
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses no Python loops and does not create any
    intermediate tensor with O(num_train * num_test * D) elements. It uses

        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b

    so that only a single (num_train, num_test) matrix is formed by a matrix
    multiply and two broadcast sums.

    Args:
        x_train: Tensor of shape (num_train, C, H, W)
        x_test: Tensor of shape (num_test, C, H, W)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is
            the squared Euclidean distance between the i-th training point and
            the j-th test point.
    """
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    train_flat = x_train.reshape(num_train, -1)
    test_flat = x_test.reshape(num_test, -1)

    train_sq = (train_flat * train_flat).sum(dim=1, keepdim=True)  # (N, 1)
    test_sq = (test_flat * test_flat).sum(dim=1, keepdim=True)  # (M, 1)
    cross = train_flat.mm(test_flat.t())  # (N, M)

    dists = train_sq + test_sq.t() - 2.0 * cross
    # Floating point error can produce tiny negative values; clamp them.
    dists = dists.clamp(min=0)
    return dists


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    """
    Given distances between all pairs of training and test samples, predict a
    label for each test sample by taking a MAJORITY VOTE among its `k` nearest
    neighbors in the training set.

    In the event of a tie, this function returns the smallest label. For
    example, if k=5 and the 5 nearest neighbors to a test example have labels
    [1, 2, 1, 2, 3] then there is a tie between 1 and 2 (each have 2 votes),
    so we return 1 since it is the smallest label.

    This function does not modify any of its inputs.

    Args:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is the
            squared Euclidean distance between the i-th training point and the
            j-th test point.
        y_train: Tensor of shape (num_train,) giving labels for all training
            samples. Each label is an integer in the range [0, num_classes - 1]
        k: The number of nearest neighbors to use for classification.

    Returns:
        y_pred: int64 Tensor of shape (num_test,) giving predicted labels for
            the test data, where y_pred[j] is the predicted label for the j-th
            test example. Each label is an integer in the range
            [0, num_classes - 1].
    """
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64, device=dists.device)
    num_classes = int(y_train.max().item()) + 1
    for j in range(num_test):
        # Indices of the k smallest distances to test sample j.
        _, idx = torch.topk(dists[:, j], k, largest=False)
        neighbor_labels = y_train[idx]
        counts = torch.bincount(neighbor_labels, minlength=num_classes)
        # Smallest label among those with the maximal vote count.
        max_count = counts.max()
        y_pred[j] = (counts == max_count).nonzero()[0, 0]
    return y_pred


class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """
        Create a new K-Nearest Neighbor classifier with the specified training
        data. In the initializer we simply memorize the provided training data.

        Args:
            x_train: Tensor of shape (num_train, C, H, W) giving training data
            y_train: int64 Tensor of shape (num_train, ) giving training labels
        """
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test: torch.Tensor, k: int = 1):
        """
        Make predictions using the classifier.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            k: The number of neighbors to use for predictions.

        Returns:
            y_test_pred: Tensor of shape (num_test,) giving predicted labels
                for the test samples.
        """
        dists = compute_distances_no_loops(self.x_train, x_test)
        y_test_pred = predict_labels(dists, self.y_train, k=k)
        return y_test_pred

    def check_accuracy(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        k: int = 1,
        quiet: bool = False
    ):
        """
        Utility method for checking the accuracy of this classifier on test
        data. Returns the accuracy of the classifier on the test data, and
        also prints a message giving the accuracy.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            y_test: int64 Tensor of shape (num_test,) giving test labels.
            k: The number of neighbors to use for prediction.
            quiet: If True, don't print a message.

        Returns:
            accuracy: Accuracy of this classifier on the test data, as a
                percent. Python float in the range [0, 100]
        """
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy
