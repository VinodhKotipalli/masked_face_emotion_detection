def prime_number(n, d):
    if n/2 < d:
        return True
    else:
        if n%d == 0:
            return False
        else:
            return prime_number(n, d+1)

def find_primes(n,i):
    if i == n + 1:
        return
    else:
        if prime_number(i, 2):
            print(i)
            return find_primes(n, i+1)
        else:
            return find_primes(n, i + 1)

find_primes(32,2)
