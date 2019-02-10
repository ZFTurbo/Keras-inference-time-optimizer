"""
Regression tests for KITO
Author: Roman Solovyev (ZFTurbo), IPPM RAS: https://github.com/ZFTurbo/
"""

import argparse

from keras.models import load_model

from kito import *
import time

def compare_two_models_results(m1, m2, test_number=10000, max_batch=10000):
    input_shape1 = m1.input_shape
    input_shape2 = m2.input_shape
    if tuple(input_shape1) != tuple(input_shape2):
        print('Different input shapes for models {} vs {}'.format(input_shape1, input_shape2))
    output_shape1 = m1.output_shape
    output_shape2 = m2.output_shape
    if tuple(output_shape1) != tuple(output_shape2):
        print('Different output shapes for models {} vs {}'.format(output_shape1, output_shape2))
    print(input_shape1, input_shape2, output_shape1, output_shape2)

    t1 = 0
    t2 = 0
    max_error = 0
    avg_error = 0
    count = 0
    for i in range(0, test_number, max_batch):
        tst = min(test_number - i, max_batch)
        print('Generate random images {}...'.format(tst))

        if type(input_shape1) is list:
            matrix = []
            for i1 in input_shape1:
                matrix.append(np.random.uniform(0.0, 1.0, (tst,) + i1[1:]))
        else:
            # None shape fix
            inp_shape_fix = list(input_shape1)
            for i in range(1, len(inp_shape_fix)):
                if inp_shape_fix[i] is None:
                    inp_shape_fix[i] = 224
            matrix = np.random.uniform(0.0, 1.0, (tst,) + tuple(inp_shape_fix[1:]))

        start_time = time.time()
        res1 = m1.predict(matrix)
        t1 += time.time() - start_time

        start_time = time.time()
        res2 = m2.predict(matrix)
        t2 += time.time() - start_time

        if type(res1) is list:
            for i1 in range(len(res1)):
                abs_diff = np.abs(res1[i1] - res2[i1])
                max_error = max(max_error, abs_diff.max())
                avg_error += abs_diff.sum()
                count += abs_diff.size
        else:
            abs_diff = np.abs(res1 - res2)
            max_error = max(max_error, abs_diff.max())
            avg_error += abs_diff.sum()
            count += abs_diff.size

    print("Initial model prediction time for {} random images: {:.2f} seconds".format(test_number, t1))
    print("Reduced model prediction time for {} same random images: {:.2f} seconds".format(test_number, t2))
    print('Models raw max difference: {} (Avg difference: {})'.format(max_error, avg_error/count))
    return max_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='model_filepath', required=True)
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False, required=False)
    args = parser.parse_args()

    model = load_model(args.model_filepath)
    verbose = args.verbose

    if verbose:
        print(model.summary())
    start_time = time.time()
    model_reduced = reduce_keras_model(model, verbose=verbose)
    print("Reduction time: {:.2f} seconds".format(time.time() - start_time))
    if verbose:
        print(model_reduced.summary())
    print('Initial model number layers: {}'.format(len(model.layers)))
    print('Reduced model number layers: {}'.format(len(model_reduced.layers)))
    print('Compare models...')
    max_error = compare_two_models_results(model, model_reduced, test_number=1000, max_batch=1000)
    if max_error > 1e-04:
        print('Possible error just happen! Max error value: {}'.format(max_error))
