import pytest
from unittest.mock import Mock
from jmetal.metaopt.component.termination import TerminationByEvaluations


class TestTerminationByEvaluations:
    """Tests for the TerminationByEvaluations class."""
    
    class TestInitialization:
        """Tests for the TerminationByEvaluations initialization."""
        
        @pytest.mark.parametrize("invalid_count", [0, -1, 3.14, "100", None])
        def test_raises_error_when_max_evaluations_invalid(self, invalid_count):
            """Should raise ValueError when maximum_number_of_evaluations is not a positive integer."""
            with pytest.raises(ValueError, match="must be a positive integer"):
                TerminationByEvaluations(maximum_number_of_evaluations=invalid_count)
        
        def test_initializes_correctly_with_valid_count(self):
            """Should initialize correctly with a valid positive integer."""
            # When
            termination = TerminationByEvaluations(maximum_number_of_evaluations=100)
            
            # Then
            assert termination.maximum_number_of_evaluations == 100
    
    class TestIsMet:
        """Tests for the is_met() method."""
        
        @pytest.fixture
        def termination(self):
            """Create a TerminationByEvaluations instance for testing."""
            return TerminationByEvaluations(maximum_number_of_evaluations=100)
        
        def test_returns_false_when_evaluations_below_maximum(self, termination):
            """Should return False when current evaluations are below the maximum."""
            # Given
            status_data = {'EVALUATIONS': 50}
            
            # When
            result = termination.is_met(status_data)
            
            # Then
            assert result is False
        
        def test_returns_true_when_evaluations_equal_maximum(self, termination):
            """Should return True when current evaluations equal the maximum."""
            # Given
            status_data = {'EVALUATIONS': 100}
            
            # When
            result = termination.is_met(status_data)
            
            # Then
            assert result is True
        
        def test_returns_true_when_evaluations_above_maximum(self, termination):
            """Should return True when current evaluations exceed the maximum."""
            # Given
            status_data = {'EVALUATIONS': 150}
            
            # When
            result = termination.is_met(status_data)
            
            # Then
            assert result is True
        
        def test_raises_keyerror_when_evaluations_key_missing(self, termination):
            """Should raise KeyError when 'EVALUATIONS' key is missing from status data."""
            # Given
            status_data = {}
            
            # When/Then
            with pytest.raises(KeyError, match="must contain 'EVALUATIONS' key"):
                termination.is_met(status_data)
        
        @pytest.mark.parametrize("invalid_value", ["100", 100.0, None, [100], {"value": 100}])
        def test_raises_typeerror_when_evaluations_not_integer(self, termination, invalid_value):
            """Should raise TypeError when 'EVALUATIONS' value is not an integer."""
            # Given
            status_data = {'EVALUATIONS': invalid_value}
            
            # When/Then
            with pytest.raises(TypeError, match="must be an integer"):
                termination.is_met(status_data)
    
    class TestProperty:
        """Tests for the maximum_number_of_evaluations property."""
        
        def test_returns_correct_value(self):
            """Should return the maximum number of evaluations set during initialization."""
            # Given
            max_evaluations = 500
            termination = TerminationByEvaluations(maximum_number_of_evaluations=max_evaluations)
            
            # When
            result = termination.maximum_number_of_evaluations
            
            # Then
            assert result == max_evaluations
