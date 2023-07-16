def rsa_decrypt_cipher(n: int, d: int, c: int) -> int:
    # TODO: Write the necessary code to get the message (m) from the cipher (c)
    m = pow(c,d,n)
    return m

def rsa_encrypt_message(m: int, e: int, n: int) -> int:
    # TODO: Write the necessary code to get the cipher (c) from the message (m)
    c = pow(m,e,n)
    return c

def rsa_calculate_private_key(e: int, p: int, q: int) -> int:
    # TODO: Write the necessary code to get the private key d from
    # the public exponent e and the factors p and q
    d = pow(e, -1, (p-1) * (q-1))
    return d
