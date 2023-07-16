def find_nth_root(num: int, power: int):
    min = 0
    max = num // power

    while min < max:
        center = (min + max) // 2
        if center**power < num:
            min = center + 1
        else:
            max = center
    return min

def rsa_broadcast_attack(N_1: int, c_1: int, N_2: int, c_2: int, N_3: int, c_3: int) -> int:
    # TODO: Write the necessary code to retrieve the decrypted message (m) using three different
    # ciphers (c_1, c_2, and c_3) created using three different public key N's (N_1, N_2, and N_3)
    m = 0

    m_3 = N_1 * N_2
    m_1 = N_2 * N_3
    m_2 = N_1 * N_3
    

    t_3 = c_3 * m_3 * pow(m_3, -1, N_3)
    t_1 = c_1 * m_1 * pow(m_1, -1, N_1)
    t_2 = c_2 * m_2 * pow(m_2, -1, N_2)

    final_c = (t_3 + t_2 + t_1) % (N_3 * N_2 * N_1)
    return find_nth_root(final_c, 3)
