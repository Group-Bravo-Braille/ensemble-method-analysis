# Ensemble Method Analysis

Conclusion: for the given use case of autoprediction for Braille users, it was important to choose a method that was simple to optimise and run, but also gave effective results.

As such the following method was chosen: Run a model switching method between LSTMs of varying input length [5, 10, 18, 50] and the fivegram model.
