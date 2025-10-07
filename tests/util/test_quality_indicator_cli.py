"""
Test cases for quality indicator CLI utility.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import io

import numpy as np

from jmetal.util.quality_indicator_cli import (
    _load_csv,
    _parse_ref_point,
    _auto_ref_point,
    _normalize_fronts,
    _compute_all_indicators,
    _print_results,
    main
)


class QualityIndicatorCLITestCases(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary CSV files for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Simple 2D front
        self.front_data = np.array([
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0]
        ])
        self.front_file = self.temp_dir / "front.csv"
        np.savetxt(self.front_file, self.front_data, delimiter=',')
        
        # Reference front
        self.reference_data = np.array([
            [0.1, 0.9],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.9, 0.1]
        ])
        self.reference_file = self.temp_dir / "reference.csv"
        np.savetxt(self.reference_file, self.reference_data, delimiter=',')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_csv_valid_file(self):
        """Test loading a valid CSV file."""
        data = _load_csv(str(self.front_file))
        np.testing.assert_array_equal(data, self.front_data)
    
    def test_load_csv_nonexistent_file(self):
        """Test loading a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            _load_csv("nonexistent.csv")
    
    def test_parse_ref_point_valid(self):
        """Test parsing a valid reference point."""
        ref_point = _parse_ref_point("1.0,2.0", 2)
        self.assertEqual(ref_point, [1.0, 2.0])
    
    def test_parse_ref_point_dimension_mismatch(self):
        """Test parsing reference point with wrong dimensions."""
        with self.assertRaises(ValueError):
            _parse_ref_point("1.0,2.0", 3)
    
    def test_parse_ref_point_invalid_format(self):
        """Test parsing invalid reference point format."""
        with self.assertRaises(ValueError):
            _parse_ref_point("1.0,invalid", 2)
    
    def test_auto_ref_point(self):
        """Test automatic reference point generation."""
        ref_point = _auto_ref_point(self.reference_data, margin=0.1)
        expected = [1.0, 1.0]  # max values + 0.1
        self.assertEqual(ref_point, expected)
    
    def test_normalize_fronts(self):
        """Test front normalization."""
        front_norm, ref_norm = _normalize_fronts(self.front_data, self.reference_data)
        
        # Reference front should be normalized to [0, 1] range
        self.assertAlmostEqual(np.min(ref_norm), 0.0, places=10)
        self.assertAlmostEqual(np.max(ref_norm), 1.0, places=10)
    
    def test_compute_all_indicators(self):
        """Test computing all indicators."""
        ref_point = [1.2, 1.2]
        results = _compute_all_indicators(self.front_data, self.reference_data, ref_point)
        
        expected_indicators = {"epsilon", "igd", "igdplus", "hv", "nhv"}
        self.assertEqual(set(results.keys()), expected_indicators)
        
        # Check that all results are numeric
        for value in results.values():
            self.assertIsInstance(value, (int, float))
    
    def test_print_results_text_format(self):
        """Test printing results in text format."""
        results = {"igd": 0.1414, "epsilon": 0.1}
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        _print_results(results, "text")
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("Result (epsilon): 0.1", output)
        self.assertIn("Result (igd): 0.1414", output)
    
    def test_print_results_json_format(self):
        """Test printing results in JSON format."""
        results = {"igd": 0.1414, "epsilon": 0.1}
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        _print_results(results, "json")
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue().strip()
        # Should be valid JSON
        import json
        parsed = json.loads(output)
        self.assertEqual(parsed, {"epsilon": 0.1, "igd": 0.1414})
    
    def test_main_with_valid_args(self):
        """Test main function with valid arguments."""
        test_args = [
            "quality_indicator_cli.py",
            str(self.front_file),
            str(self.reference_file),
            "igd"
        ]
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("Result (igd):", output)
    
    def test_main_with_all_indicators(self):
        """Test main function computing all indicators."""
        test_args = [
            "quality_indicator_cli.py",
            str(self.front_file),
            str(self.reference_file),
            "all",
            "--ref-point", "1.2,1.2"
        ]
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("Result (epsilon):", output)
        self.assertIn("Result (igd):", output)
        self.assertIn("Result (igdplus):", output)
        self.assertIn("Result (hv):", output)
        self.assertIn("Result (nhv):", output)
    
    def test_main_with_normalization(self):
        """Test main function with normalization."""
        test_args = [
            "quality_indicator_cli.py",
            str(self.front_file),
            str(self.reference_file),
            "igd",
            "--normalize"
        ]
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("Fronts normalized", output)
        self.assertIn("Result (igd):", output)


if __name__ == "__main__":
    unittest.main()