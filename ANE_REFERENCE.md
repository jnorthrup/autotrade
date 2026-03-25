# ANE Training

External ANE training code located in ./ANE/training/
See ./ANE/README.md for details on ANE backpropagation implementation.

To use:
```bash
cd ANE/training
xcrun clang -O2 -framework Foundation -framework IOSurface \
  -framework CoreML -framework Accelerate -ldl -lobjc \
  -o train_large train_large.m
./train_large
```
