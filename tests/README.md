# CAIF Neural Network Framework Test Suite

This directory contains comprehensive tests for the CAIF (Artificial Intelligence Framework) neural network library. The test suite validates all major components including loss functions, optimizers, neural network training, and integration scenarios.

## Test Structure

### Test Files

- **`test_loss_functions.cpp`** - Tests for MSE and Cross Entropy loss functions
- **`test_optimizers.cpp`** - Tests for SGD and Adam optimizers
- **`test_neural_network_training.cpp`** - Tests for neural network training functionality
- **`test_integration.cpp`** - End-to-end integration tests
- **`run_all_tests.cpp`** - Master test runner

### Test Coverage

#### Loss Functions Tests
- Mean Squared Error (MSE) loss computation and gradients
- Cross Entropy loss computation and gradients
- Error handling for shape mismatches
- Numerical accuracy verification

#### Optimizer Tests
- SGD basic functionality and momentum
- Adam optimizer with adaptive learning rates
- Parameter update correctness
- Convergence behavior
- Reset functionality
- Error handling

#### Neural Network Training Tests
- Network compilation and configuration
- Forward pass functionality
- Prediction capabilities
- Training with regression problems
- Training with classification problems
- Multi-layer network support
- Error handling for invalid configurations

#### Integration Tests
- Complete XOR problem solution
- Linear regression end-to-end
- Multi-class classification
- Optimizer comparison
- Real-world problem scenarios

## Building and Running Tests

### Prerequisites

1. **Build the CAIF library first:**
   ```bash
   cd ..  # Go to CAIF root directory
   make   # or scons
   ```

2. **Ensure you have a C++23 compatible compiler** (GCC 11+ or Clang 14+)

### Building Tests

```bash
# Build all tests
make all

# Or build individual tests
make test_loss_functions
make test_optimizers
make test_neural_network_training
make test_integration
```

### Running Tests

#### Run All Tests
```bash
make test
```
This builds and runs all test suites with a comprehensive summary.

#### Run Individual Test Suites
```bash
make run-loss-functions      # Test loss functions only
make run-optimizers          # Test optimizers only
make run-neural-network      # Test neural network training only
make run-integration         # Test integration scenarios only
```

#### Quick Integration Test
```bash
make quick-test
```
Runs only the integration test for a fast validation.

### Test Output

Each test provides detailed output including:
- Individual test case results (PASSED/FAILED)
- Error messages for failed tests
- Performance metrics
- Summary statistics

Example output:
```
=== CAIF Loss Functions Test Suite ===

Testing MSE Loss - Simple Case... PASSED
Testing MSE Gradient... PASSED
Testing Cross Entropy Loss... PASSED
Testing Cross Entropy Gradient... PASSED
Testing Loss Function Error Handling... PASSED

=== Test Summary ===
All tests PASSED!
```

## Test Details

### Loss Functions Tests

**MSE Loss Tests:**
- Validates correct loss computation for simple cases
- Verifies gradient calculation accuracy
- Tests numerical stability

**Cross Entropy Tests:**
- Tests with one-hot encoded targets
- Validates gradient computation
- Tests epsilon parameter for numerical stability

### Optimizer Tests

**SGD Tests:**
- Basic parameter updates
- Momentum accumulation
- Learning rate effects

**Adam Tests:**
- Adaptive learning rates
- Bias correction
- Convergence on quadratic functions

### Neural Network Tests

**Compilation Tests:**
- Network setup and configuration
- Layer addition and shape inference
- Optimizer and loss function assignment

**Training Tests:**
- Regression problems (linear relationships)
- Classification problems (binary and multi-class)
- Loss reduction over epochs
- Accuracy computation

### Integration Tests

**XOR Problem:**
- Classic non-linearly separable problem
- Tests network's ability to learn complex patterns
- Validates complete training pipeline

**Linear Regression:**
- Simple y = mx + b relationship
- Tests numerical accuracy
- Validates prediction capabilities

**Multi-Class Classification:**
- 3-class problem with spatial separation
- Tests softmax-like outputs
- Validates classification accuracy

## Troubleshooting

### Common Issues

1. **Library not found error:**
   ```
   Error: libcaif.a not found
   ```
   **Solution:** Build the CAIF library first with `make` or `scons` in the parent directory.

2. **Compilation errors:**
   - Ensure you have C++23 support
   - Check that all header files are accessible
   - Verify include paths in Makefile

3. **Test failures:**
   - Check that the CAIF library was built correctly
   - Verify that all implementations are complete
   - Review error messages for specific issues

### Debug Mode

For debugging failed tests, you can compile with debug symbols:
```bash
make clean
CXXFLAGS="-std=c++23 -Wall -Wextra -g -O0 -I../include/caif" make all
```

## Adding New Tests

To add new test cases:

1. **For existing test files:** Add new test functions following the existing pattern
2. **For new test files:** 
   - Create new `.cpp` file
   - Add to Makefile
   - Update `run_all_tests.cpp` to include the new test

### Test Function Template

```cpp
bool TestNewFeature()
{
  std::cout<<"Testing New Feature... ";
  
  try
  {
    // Test implementation
    // ...
    
    std::cout<<"PASSED\n";
    return true;
  }
  catch(const std::exception &e)
  {
    std::cout<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}
```

## Performance Expectations

- **Loss Functions Tests:** < 1 second
- **Optimizer Tests:** < 2 seconds  
- **Neural Network Tests:** < 5 seconds
- **Integration Tests:** < 10 seconds
- **Total Test Suite:** < 20 seconds

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. The test runner returns appropriate exit codes:
- `0` - All tests passed
- `1` - One or more tests failed

## Contributing

When contributing new features to CAIF:

1. **Add corresponding tests** for new functionality
2. **Ensure all existing tests pass** 
3. **Update test documentation** if needed
4. **Follow the existing test patterns** and coding style

## Support

For issues with the test suite:
1. Check this README for common solutions
2. Review test output for specific error messages
3. Ensure the CAIF library builds correctly
4. Verify your development environment meets requirements 