import hashlib
import json
import typing
from math import floor, sqrt


##############################################
# Change this to your 9-digit Georgia Tech ID!
STUDENT_ID = '903401672'
##############################################


def print_tests_for_student_id() -> None:
    f = open('student_tests.json')
    student_tests = json.load(f)

    student_id_hash = hashlib.sha256(STUDENT_ID.encode()).hexdigest()
    try:
        tests = student_tests[student_id_hash]
        print('The tests for ID {} are:'.format(STUDENT_ID))
        print('========================================================')
        for test_id, test in tests.items():
            print('{} -> {}'.format(test_id, test))
        print('========================================================')
    except KeyError:
        print('ERROR: ID {} was not found in student_tests.'.format(STUDENT_ID))


# This function is only provided for your convenience.
# You are not required to use it.
def rsa_factor_64_bit_key(N: int, e: int) -> typing.Tuple[int, int]:
    p = 0
    q = 0

    root_n = floor(sqrt(N))
    if root_n % 2 == 0:
        root_n += 1
    for i in range(root_n , N, 2):
        if N % i == 0:
            p = i
            q = N // p
            break
    # TODO: Write the necessary code to get the factors p and q of the public key (N, e)

    if p > q:
        temp = q
        q = p
        p = temp

    return p, q


if __name__ == '__main__':
    print(rsa_factor_64_bit_key(e=65537, N=947116045342907509))
    print(rsa_factor_64_bit_key(e=65537, N=947631742231500539))
    print(rsa_factor_64_bit_key(e=65537, N=959198948936610359))
    print(rsa_factor_64_bit_key(e=65537, N=950440510780863919))
    print(rsa_factor_64_bit_key(e=65537, N=935649924689604829))
    print_tests_for_student_id()
