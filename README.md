# A parametrization method for B-spline curve interpolation via supervised regression

## Project Structure

The repository contains source code in two languages:
- **C++ source files**: Located in `src-cpp/`
- **Python source files**: Located in `src-py/`

### Compilation Instructions

The main C++ files are stored in `src-cpp/B-spline-curve-fitting/main_scripts/`. To compile and run a specific program:

1. Copy the desired `main.cpp` file from `src-cpp/B-spline-curve-fitting/main_scripts/`
2. Paste it into `src-cpp/B-spline-curve-fitting/`
3. Rename the file to `main.cpp`
4. Compile and execute the program

## Method Replication and Evaluation

Follow these steps to replicate the method described in the paper:

### 1. Dataset Generation
- **Regression dataset**: Run `src-cpp/B-spline-curve-fitting/main_scripts/main-0. generating data with regression labels.cpp`
- **Classification dataset**: Run `src-cpp/B-spline-curve-fitting/main_scripts/main-9. generating data with class labels.cpp`

### 2. Classifier Training and Inference
1. Train the classifier: `src-py/training/classifier_training.py`
2. Run inference: `src-py/inference/classifier_inference.py`

### 3. Method Testing (Paper Implementation)
1. Run training script: `src-py/training/main_del_feat.py`
2. Run prediction script: `src-py/inference/seq_pred_del_feat.py`
3. Execute test script: `src-cpp/B-spline-curve-fitting/main_scripts/main-5. test script.cpp`

### 4. Curve Visualization
1. Generate curve data: `src-cpp/B-spline-curve-fitting/main_scripts/main-6. record crvs and params.cpp`
2. Plot interpolating curves: `DrawNurbs2D.py`

### 5. GA-Optimized Knot Vectors Testing
1. Run training: `src-py/training/main_del_feat.py`
2. Run prediction: `src-py/inference/seq_pred_del_feat.py`
3. Test with optimized knots: `src-cpp/B-spline-curve-fitting/main_scripts/main-7. ga-optimized knots.cpp`

### 6. Classification Method Comparison
Compare with idealized classifier approach:
1. Run training: `src-py/training/main_del_feat.py`
2. Run prediction: `src-py/inference/seq_pred_del_feat.py`
3. Execute comparison: `src-cpp/B-spline-curve-fitting/main_scripts/main-8. superior count.cpp`

## Method Validation

These steps validate the method's performance and optimize parameters:

### 1. Parameter Optimization
Determine optimal parameters for regression models:
1. Grid search training: `src-py/training/grid_train.py`
2. Grid validation: `src-py/inference/grid_val.py`
3. Parameter testing: `src-cpp/B-spline-curve-fitting/main_scripts/main-1. param grid test.cpp`

### 2. Regression Model Evaluation
1. Run training: `src-py/training/main_del_feat.py`
2. Run prediction: `src-py/inference/seq_pred_model_val.py`
3. Execute tests: `src-cpp/B-spline-curve-fitting/main_scripts/main-2. testing regression models.cpp`

### 3. Feature Selection and Dataset Size Optimization
1. Run training: `src-py/training/test_dataset_size_feat_model.py`
2. Run prediction: `src-py/inference/seq_pred_datasize_val.py`
3. Execute tests: `src-cpp/B-spline-curve-fitting/main_scripts/main-3. feat and dataset size.cpp`

### 4. Local Stage Sequence Length Optimization
1. Run training: `src-py/training/local_length_test_del_feat.py`
2. Run prediction: `src-py/inference/local_len_metric_del_feat.py`
3. Execute tests: `src-cpp/B-spline-curve-fitting/main_scripts/main-4. testing local len.cpp`

## System Requirements
**Operating System**: Windows only

## Usage Notes

- Ensure all dependencies are installed for both Python and C++ components
- Follow the sequential execution order for each workflow
- The C++ compilation step must be repeated for each different main script you wish to run