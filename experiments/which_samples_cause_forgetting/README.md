Setup:

1. Repeat:
   1. Train a model on task 1.
   2. For every sample from task 2, do a single step on the previous model.
   3. Save some statistics for every sample: how much accuracy was reduced, etc.
2. Average over multiple initialiazations.

(Task 1, Task2) tuple options to be tried:
- (MNIST 0-4, MNIST 5-9)
- (CIFAR 0-4, CIFAR 5-9)
- ...? 
