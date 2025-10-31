"""
This module implements the Observer design pattern for event handling in JMetalPy.

The Observer pattern allows objects to notify other objects about changes in their state.
This is particularly useful for monitoring algorithm progress, logging, and visualization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeVar, Generic, Optional


class Observer(ABC):
    """Abstract base class for observers in the Observer pattern.
    
    Observers are objects that receive updates from Observable objects they are
    registered with. The type of the observable subject is not strictly enforced
    for better compatibility with Python's method resolution order (MRO).
    
    Subclasses must implement the update() method to define how they handle
    notifications from observables.
    """
    
    @abstractmethod
    def update(self, subject: Any, *args: Any, **kwargs: Any) -> None:
        """Receive an update from an observable subject.
        
        This method is called whenever the observed subject changes state.
        
        Args:
            subject: The observable object that sent the update.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments containing update data.
                    Common keys include:
                    - 'evaluations': Current number of evaluations
                    - 'solutions': Current population or solution set
                    - 'computing_time': Elapsed computation time
        """
        pass


class Observable(ABC):
    """Abstract base class for observable subjects in the Observer pattern.
    
    Observable objects maintain a list of observers and notify them when their
    state changes. This implementation is thread-safe and supports multiple observers.
    """
    
    def __init__(self) -> None:
        """Initialize the observable with an empty list of observers."""
        self._observers: List[Observer[T]] = []
    
    @abstractmethod
    def register(self, observer: 'Observer') -> None:
        """Register an observer to receive updates.
        
        Args:
            observer: The observer to register.
            
        Raises:
            TypeError: If the observer is not an instance of Observer.
        """
        pass

    @abstractmethod
    def deregister(self, observer: 'Observer') -> None:
        """Remove an observer from the notification list.
        
        Args:
            observer: The observer to remove.
            
        Note:
            If the observer is not in the list, this method does nothing.
        """
        pass

    @abstractmethod
    def deregister_all(self) -> None:
        """Remove all observers from the notification list."""
        pass

    @abstractmethod
    def notify_all(self, *args: Any, **kwargs: Any) -> None:
        """Notify all registered observers.
        
        This method calls the update() method on each registered observer,
        passing along any provided arguments.
        
        Args:
            *args: Variable length argument list to pass to observers.
            **kwargs: Arbitrary keyword arguments to pass to observers.
        """
        pass
