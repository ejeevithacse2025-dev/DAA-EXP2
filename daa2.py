"""
String Matching Algorithm Comparison
--------------------------------------
This program implements THREE classic pattern matching algorithms:
    1. Naive (Brute Force) String Matching
    2. KMP (Knuth-Morris-Pratt) String Matching
    3. Rabin-Karp String Matching

For each algorithm, we count the number of CHARACTER COMPARISONS made
(not just execution time), because that's what the question asks us
to analyze -- comparisons are a hardware/CPU-independent way to judge
algorithm efficiency.

We test on:
    - Text length  (n) = 5000 characters
    - Pattern lengths (m) = 4, 8, 16, 32 characters
"""

import random
import string


# ============================================================
# 1. NAIVE STRING MATCHING
# ============================================================
def naive_search(text, pattern):
    """
    How it works (in plain English):
    ---------------------------------
    Slide the pattern over the text one position at a time.
    At each position, compare characters one by one.
    If a mismatch happens, stop comparing at THIS position
    and slide the pattern one step to the right.

    This is the "brute force" way -- simple but wasteful,
    because it forgets everything it learned from the previous
    comparison when it moves to the next position.
    """
    n = len(text)
    m = len(pattern)
    comparisons = 0          # counter for character comparisons
    matches = []              # store starting indices where pattern is found

    # We can only start a match at positions 0 to n-m (pattern must fit)
    for i in range(n - m + 1):
        j = 0                # index into the pattern
        # Compare pattern with text starting at position i
        while j < m:
            comparisons += 1                     # every char comparison counts
            if text[i + j] != pattern[j]:
                break                             # mismatch -> stop early
            j += 1
        if j == m:
            matches.append(i)                    # full pattern matched

    return matches, comparisons


# ============================================================
# 2. KMP (KNUTH-MORRIS-PRATT) STRING MATCHING
# ============================================================
def compute_lps(pattern):
    """
    LPS = "Longest Proper Prefix which is also a Suffix"

    Why do we need this?
    ---------------------
    This array tells KMP: "If a mismatch happens here, how far
    can I safely skip in the pattern WITHOUT re-checking
    characters I already know matched?"

    Example: pattern = "ABABC"
    lps = [0, 0, 1, 2, 0]

    Building this array itself takes some comparisons, and we
    count them too, because it's part of the algorithm's real cost.
    """
    m = len(pattern)
    lps = [0] * m
    length = 0        # length of the previous longest prefix-suffix
    i = 1
    comparisons = 0

    while i < m:
        comparisons += 1
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]   # fall back (no comparison, just a jump)
            else:
                lps[i] = 0
                i += 1

    return lps, comparisons


def kmp_search(text, pattern):
    """
    How it works (in plain English):
    ---------------------------------
    Just like Naive search, we slide and compare.
    BUT when a mismatch happens, instead of restarting the
    pattern from index 0, we use the LPS array to jump ahead
    intelligently -- skipping comparisons we already know
    would succeed.

    This is why KMP never has to "go backwards" in the text.
    """
    n = len(text)
    m = len(pattern)
    matches = []

    if m == 0:
        return matches, 0

    lps, lps_comparisons = compute_lps(pattern)
    comparisons = lps_comparisons     # start counting from LPS build cost

    i = 0   # index for text
    j = 0   # index for pattern

    while i < n:
        comparisons += 1
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                matches.append(i - j)      # found a match
                j = lps[j - 1]              # continue searching (use LPS to shift)
        else:
            if j != 0:
                j = lps[j - 1]              # shift pattern using LPS (no char comparison)
            else:
                i += 1                       # no match at all, move text pointer

    return matches, comparisons


# ============================================================
# 3. RABIN-KARP STRING MATCHING
# ============================================================
def rabin_karp_search(text, pattern, prime=101, base=256):
    """
    How it works (in plain English):
    ---------------------------------
    Instead of comparing pattern vs text character by character
    at every position, we first convert both the pattern and
    every window of the text into a NUMBER (a hash).

    If two numbers (hashes) are different, we KNOW for certain
    the strings are different -- so we skip character comparison
    entirely.

    Only when hashes MATCH do we do a character-by-character
    check (because different strings can rarely produce the
    same hash -- this is called a "collision").

    We use a "rolling hash" so we don't recompute the hash of
    the whole window from scratch every time -- we just slide it.
    """
    n = len(text)
    m = len(pattern)
    matches = []
    comparisons = 0     # we only count CHARACTER comparisons, not hash computations

    if m == 0 or m > n:
        return matches, 0

    # highest power of base needed for rolling hash removal, e.g. base^(m-1) % prime
    h = pow(base, m - 1, prime)

    pattern_hash = 0
    window_hash = 0

    # Compute initial hash for pattern and first window of text
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % prime
        window_hash = (base * window_hash + ord(text[i])) % prime

    for i in range(n - m + 1):
        # Step 1: compare hashes (cheap, not counted as char comparison)
        if pattern_hash == window_hash:
            # Step 2: hashes match -> verify with actual character comparison
            j = 0
            while j < m:
                comparisons += 1
                if text[i + j] != pattern[j]:
                    break
                j += 1
            if j == m:
                matches.append(i)

        # Roll the hash forward to the next window (remove leftmost char, add new rightmost char)
        if i < n - m:
            window_hash = (base * (window_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if window_hash < 0:
                window_hash += prime

    return matches, comparisons


# ============================================================
# EXPERIMENT: Compare all 3 algorithms
# ============================================================
def generate_text(n, alphabet="ABCD"):
    """Generate a random text of length n using a small alphabet.
    A SMALL alphabet (like ABCD) is used deliberately -- it creates
    more partial matches/false starts, which is what makes Naive
    search look bad and KMP/Rabin-Karp look good. A huge random
    alphabet would make almost every algorithm equally fast."""
    return ''.join(random.choice(alphabet) for _ in range(n))


def generate_pattern(text, length, from_text=True, alphabet="ABCD"):
    """
    from_text=True  -> guarantees the pattern actually exists in the text
                        (extracted from a random position)
    from_text=False -> generates a completely random pattern
                        (may or may not exist in text)
    """
    if from_text:
        start = random.randint(0, len(text) - length)
        return text[start:start + length]
    else:
        return ''.join(random.choice(alphabet) for _ in range(length))


def run_experiment():
    random.seed(42)   # fixed seed so results are reproducible for your report

    n = 5000
    text = generate_text(n)
    pattern_lengths = [4, 8, 16, 32]

    print("=" * 90)
    print(f"TEXT LENGTH (n) = {n}")
    print("=" * 90)

    results = []

    for m in pattern_lengths:
        pattern = generate_pattern(text, m, from_text=True)

        naive_matches, naive_cmp = naive_search(text, pattern)
        kmp_matches, kmp_cmp = kmp_search(text, pattern)
        rk_matches, rk_cmp = rabin_karp_search(text, pattern)

        results.append((m, naive_cmp, kmp_cmp, rk_cmp))

        print(f"\nPattern length m = {m}")
        print(f"  Pattern (sample) : {pattern}")
        print(f"  Naive matches found      : {len(naive_matches)}")
        print(f"  KMP matches found        : {len(kmp_matches)}")
        print(f"  Rabin-Karp matches found : {len(rk_matches)}")
        print(f"  Character Comparisons -> Naive: {naive_cmp:6d} | "
              f"KMP: {kmp_cmp:6d} | Rabin-Karp: {rk_cmp:6d}")

    # -------- Summary Table --------
    print("\n" + "=" * 90)
    print("SUMMARY TABLE: Character Comparisons vs Pattern Length")
    print("=" * 90)
    print(f"{'Pattern Length (m)':<20}{'Naive':<15}{'KMP':<15}{'Rabin-Karp':<15}")
    print("-" * 65)
    for m, nc, kc, rc in results:
        print(f"{m:<20}{nc:<15}{kc:<15}{rc:<15}")


if __name__ == "__main__":
    run_experiment()
