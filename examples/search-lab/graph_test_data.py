"""Adversarial IDs for native graph hash-probe cancellation tests."""


def colliding_ids(count):
    # Invert graph_hash's SplitMix64 steps to make the final low 32 bits zero.
    modulus = 1 << 64
    inverse_second = pow(0x94D049BB133111EB, -1, modulus)
    inverse_first = pow(0xBF58476D1CE4E5B9, -1, modulus)
    for i in range(count):
        x = i << 32
        x ^= (x >> 31) ^ (x >> 62)
        x = x * inverse_second % modulus
        x ^= (x >> 27) ^ (x >> 54)
        x = x * inverse_first % modulus
        x ^= (x >> 30) ^ (x >> 60)
        x = (x - 0x9E3779B97F4A7C15) % modulus
        yield x if x < 1 << 63 else x - modulus
