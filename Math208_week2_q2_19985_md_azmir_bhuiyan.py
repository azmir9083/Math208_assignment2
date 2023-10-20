import numpy as np

def generate_normal_random_numbers(n, mu, sigma):
    return np.random.normal(mu, sigma, n)

def generate_uniform_random_numbers(n, low, high):
    return np.random.uniform(low, high, n)

def verify_Chebyshev_ineq(lst, mu, sigma, k):
    lower_bound = mu - k*sigma
    upper_bound = mu + k*sigma
    count = sum(lower_bound <= x <= upper_bound for x in lst)
    probability = count / len(lst)
    chebyshev_bound = 1 - 1/(k**2)
    
    print(f"Probability of |X-u| = {probability} ; 1-1/(k^2)= {chebyshev_bound}")
    print(f"When k = {k} , P(|X-u|<k*sd)>={1-1/k**2} is {'True' if probability >= chebyshev_bound else 'False'}")
    return count

def main():
    mu, sigma = 10, 0.5
    lst_normal = generate_normal_random_numbers(50, mu, sigma)
    lst_uniform = generate_uniform_random_numbers(50, -20, 20)

    for k in [1, 2**0.5, 1.5, 2, 3]:
        print("\nNormal distribution:")
        cnt_normal = verify_Chebyshev_ineq(lst_normal, mu, sigma, k)

        print("\nUniform distribution:")
        # For a uniform distribution, the standard deviation is not the same as for the normal distribution. 
        # The standard deviation for the uniform distribution in the range [a,b] is sqrt((b-a)^2/12).
        uniform_sigma = np.sqrt((20 - (-20))**2 / 12)
        uniform_mu = 0  # Because the range is symmetric around zero.
        cnt_uniform = verify_Chebyshev_ineq(lst_uniform, uniform_mu, uniform_sigma, k)

if __name__ == "__main__":
    main()
