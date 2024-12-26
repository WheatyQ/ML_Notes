import pandas as pd
import numpy as np

# Sample data (you should replace this with your actual data)
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 7, 6, 5, 4, 3, 2, 1]
index = list(range(1, len(data) + 1))
series = pd.Series(data, index=index)

# Step 3: Find a strictly increasing subsequence with a threshold
def longest_increasing_subsequence_with_threshold(arr, threshold):
    n = len(arr)
    dp = [1] * n
    prev = [-1] * n

    for i in range(1, n):
        for j in range(i):
            # Check if the difference between arr[i] and arr[j] is larger than (i-j) * threshold
            if arr[i] - arr[j] > (i - j) * threshold and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
                prev[i] = j

    # Find the index of the maximum length
    max_length = max(dp)
    index = dp.index(max_length)

    # Reconstruct the subsequence
    subsequence = []
    while index != -1:
        subsequence.append(arr[index])
        index = prev[index]

    return subsequence[::-1]  # Reverse to get the correct order

# Define the threshold (you can adjust this value)
threshold = 0.5

# Get the strictly increasing subsequence with the threshold
lis = longest_increasing_subsequence_with_threshold(series.values, threshold)

# Step 4: Extract the subsequence from the original series
new_series = pd.Series(lis, index=series[series.isin(lis)].index[:len(lis)])

# Add the expected value at the end
new_series[index[-1]] = index[-1]  # Setting the last index to its value

# Step 5: Interpolate missing values
interpolated_series = new_series.reindex(series.index).interpolate(method="linear")

print("Original Series:")
print(series)
print("\nInterpolated Series:")
print(interpolated_series)
