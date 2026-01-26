#!/usr/bin/env python3
"""
Quick test script to compare detection performance with different optimization settings.
"""

import cv2
import numpy as np
import time

# Test configurations
configs = {
    "original": {
        "use_clahe": False,
        "use_bilateral": False,
        "use_multi_threshold": False,
        "adaptive_thresh_constant": 7,
        "corner_refinement_max_iterations": 30,
    },
    "optimized": {
        "use_clahe": True,
        "use_bilateral": False,
        "use_multi_threshold": True,
        "adaptive_thresh_constant": 5,
        "corner_refinement_max_iterations": 60,
    },
    "maximum": {
        "use_clahe": True,
        "use_bilateral": True,
        "use_multi_threshold": True,
        "adaptive_thresh_constant": 5,
        "corner_refinement_max_iterations": 80,
    }
}

def setup_detector(config_name):
    """Setup ArUco detector with given configuration."""
    cfg = configs[config_name]
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
    parameters = cv2.aruco.DetectorParameters_create()
    
    # Apply configuration
    parameters.adaptiveThreshConstant = cfg["adaptive_thresh_constant"]
    parameters.cornerRefinementMaxIterations = cfg["corner_refinement_max_iterations"]
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.03
    parameters.minCornerDistanceRate = 0.03
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMinAccuracy = 0.01
    parameters.errorCorrectionRate = 0.6
    parameters.detectInvertedMarker = True
    
    return dictionary, parameters, cfg

def preprocess_image(gray, config):
    """Preprocess image based on configuration."""
    processed = gray.copy()
    
    if config["use_clahe"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
    
    if config["use_bilateral"]:
        processed = cv2.bilateralFilter(processed, 5, 50, 50)
    
    return processed

def detect_markers(gray, dictionary, parameters, config):
    """Detect markers with given configuration."""
    processed = preprocess_image(gray, config)
    
    if config["use_multi_threshold"]:
        all_corners = []
        all_ids = []
        seen_ids = set()
        
        # Adaptive
        corners, ids, _ = cv2.aruco.detectMarkers(processed, dictionary, parameters=parameters)
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in seen_ids:
                    all_corners.append(corners[i])
                    all_ids.append(marker_id)
                    seen_ids.add(marker_id)
        
        # Otsu
        _, thresh_otsu = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        corners, ids, _ = cv2.aruco.detectMarkers(thresh_otsu, dictionary, parameters=parameters)
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in seen_ids:
                    all_corners.append(corners[i])
                    all_ids.append(marker_id)
                    seen_ids.add(marker_id)
        
        if len(all_ids) > 0:
            return tuple(all_corners), np.array(all_ids).reshape(-1, 1)
        else:
            return tuple(), None
    else:
        return cv2.aruco.detectMarkers(processed, dictionary, parameters=parameters)[:2]

def test_configuration(frame, config_name):
    """Test a configuration and return results."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary, parameters, cfg = setup_detector(config_name)
    
    # Warmup
    for _ in range(3):
        detect_markers(gray, dictionary, parameters, cfg)
    
    # Time detection
    num_iterations = 10
    start = time.time()
    for _ in range(num_iterations):
        corners, ids = detect_markers(gray, dictionary, parameters, cfg)
    elapsed = (time.time() - start) / num_iterations * 1000
    
    num_detected = len(ids) if ids is not None else 0
    
    return {
        "config": config_name,
        "num_markers": num_detected,
        "time_ms": elapsed,
        "fps": 1000.0 / elapsed if elapsed > 0 else 0
    }

def main():
    """Run comparison test."""
    print("ArUco Detection Optimization Comparison")
    print("=" * 60)
    print()
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Using test image instead...")
        # Create a test frame with some noise
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        cap.release()
        print(f"Using camera frame: {frame.shape}")
    
    print()
    print("Configuration Details:")
    print("-" * 60)
    for name, cfg in configs.items():
        print(f"\n{name.upper()}:")
        for key, value in cfg.items():
            print(f"  {key}: {value}")
    
    print()
    print("-" * 60)
    print("Testing configurations...")
    print("-" * 60)
    
    results = []
    for config_name in ["original", "optimized", "maximum"]:
        print(f"\nTesting {config_name}...", end=" ", flush=True)
        result = test_configuration(frame, config_name)
        results.append(result)
        print("Done!")
    
    print()
    print("-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"{'Configuration':<15} {'Markers':<10} {'Time (ms)':<12} {'FPS':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['config']:<15} {result['num_markers']:<10} "
              f"{result['time_ms']:<12.2f} {result['fps']:<10.1f}")
    
    # Calculate improvements
    print()
    print("-" * 60)
    print("Improvements vs Original:")
    print("-" * 60)
    
    original_time = results[0]['time_ms']
    for result in results[1:]:
        time_overhead = result['time_ms'] - original_time
        time_percent = (time_overhead / original_time) * 100
        
        print(f"\n{result['config'].upper()}:")
        print(f"  Markers detected: {result['num_markers']} "
              f"({'+' if result['num_markers'] > results[0]['num_markers'] else ''}"
              f"{result['num_markers'] - results[0]['num_markers']})")
        print(f"  Processing time: +{time_overhead:.2f}ms (+{time_percent:.1f}%)")
        print(f"  Still achieves {result['fps']:.1f} FPS")
    
    print()
    print("-" * 60)
    print("Recommendations:")
    print("-" * 60)
    print("• Use 'optimized' for best balance (good detection, acceptable speed)")
    print("• Use 'maximum' for challenging conditions (low light, motion blur)")
    print("• Use 'original' only if processing time is critical and lighting is perfect")
    print()
    print("To adjust settings, edit: config/aruco_advanced.yaml")
    print()

if __name__ == "__main__":
    main()
