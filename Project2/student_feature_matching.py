import numpy as np


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    dists = []
    
    # CAN OPTIMIZE BY NOT USING FOR LOOPS
    for n in range(features1.shape[0]):
        inner = []
        for m in range(features2.shape[0]):
            x = features1[n]
            y = features2[m]
            d = np.linalg.norm(x-y)
            # x = np.multiply(x, x)
            # y = np.multiply(y, y)
            # print(x.shape)
            # print(y.shape)
            # print(x+y)
            # e = np.sum(x+y)
            # e = np.sqrt(e)
            # print("e", e)
            # print("d", d)
            inner.append(d)
        dists.append(inner)

    # f1 = np.multiply(features1, features1)
    # f2 = np.multiply(features2, features2)
    # a = np.add(f1, f2)
    # b = np.sqrt(a)
    # print(b)

    # print("dists", dists)
    dists = np.array(dists)

    # raise NotImplementedError('`match_features` function in ' +
    #     '`student_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    dists = compute_feature_distances(features1, features2)
    matches = []
    confidences = []

    for n in range(x1.shape[0]):
        row1 = dists[n]
        min1 = np.amin(row1)
        row2 = np.delete(row1, row1.argmin())
        min2 = np.amin(row2)
        ratio = min1/min2
        # low ratio is good
        if ratio <= 0.75:
            inner = [n, row1.argmin()]  # not sure about the correctness of this
            matches.append(inner)
            confidences.append(1-ratio)   # higher confidence is lower ratio confidence = 1-ratio?

    matches = np.array(matches)
    confidences = np.array(confidences)

    # raise NotImplementedError('`match_features` function in ' +
    #     '`student_feature_matching.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
