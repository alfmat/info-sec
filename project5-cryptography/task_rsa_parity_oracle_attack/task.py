from typing import Callable

# You may find these helpful
from math import ceil, log2
from decimal import *


def rsa_parity_oracle_attack(c: int, N: int, e: int, oracle: Callable[[int], bool]) -> str:

    # TODO: Write the necessary code to get the plaintext message from the cipher (c) using
    # the public key (N, e) and an oracle function - oracle(chosen_c) that will give you
    # the parity of the decrypted value of a chosen cipher (chosen_c) value using the hidden private key (d)

    iterations = int(ceil(log2(N)))
    getcontext().prec = iterations
    low = Decimal(0)
    high = Decimal(N)

    hack = pow(2, e, N)

    for _ in range(0, iterations):
        c = (c * hack) % N
        if not oracle(c):
            low = (high + low) / 2
        else:
            high = (high + low) / 2
    
    m_int = int(high)
    # Transform the integer value of the message into a human readable form
    message = bytes.fromhex(hex(int(m_int)).rstrip('L')[2:]).decode('utf-8')

    return message
