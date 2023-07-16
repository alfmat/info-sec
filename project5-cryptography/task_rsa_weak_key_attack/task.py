from math import gcd
import typing

def rsa_weak_key_attack(given_public_key_N: int, given_public_key_e: int, public_key_list: typing.List[int]) -> int:
    # TODO: Write the necessary code to retrieve the private key d from the given public
    # key (N, e) using only the list of public keys generated using the same flawed RNG
    d = 0
    N1 = given_public_key_N
    p = 1
    for N2 in public_key_list:
        p = gcd(N1,N2)
        if p != 1:
            break
    q = N1 // p
    d = pow(given_public_key_e, -1, (p-1) * (q-1))
    return d
