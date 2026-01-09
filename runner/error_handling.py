"""
Robuste Fehlerbehandlung für tile-compile

Stellt Mechanismen zur Fehlerbehandlung und Fehlerprotokollierung bereit.
"""

import logging
import traceback
from typing import Callable, Any, Optional

class ProcessingError(Exception):
    """Basisklasse für Verarbeitungsfehler"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
        self.log_error()
    
    def log_error(self):
        """Protokolliert Fehlerdetails"""
        logger = logging.getLogger('ProcessingError')
        logger.error(f"Processing Error: {self}")
        if self.original_error:
            logger.error(f"Original Error: {self.original_error}")
            logger.error(traceback.format_exc())

class MemoryManagementError(ProcessingError):
    """Speichermanagement-Fehler"""
    pass

class ResourceAllocationError(ProcessingError):
    """Ressourcenzuweisungsfehler"""
    pass

def robust_processing(func: Callable) -> Callable:
    """
    Decorator für robuste Fehlerbehandlung
    
    Args:
        func: Zu dekorierende Funktion
    
    Returns:
        Dekoriete Funktion mit Fehlerbehandlung
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except MemoryError as e:
            logger.error(f"Memory Error in {func.__name__}: {e}")
            raise MemoryManagementError(
                f"Nicht genügend Arbeitsspeicher: {e}", 
                original_error=e
            )
        except ResourceWarning as e:
            logger.error(f"Resource Warning in {func.__name__}: {e}")
            raise ResourceAllocationError(
                f"Ressourcen-Problem: {e}", 
                original_error=e
            )
        except Exception as e:
            logger.error(f"Unerwarteter Fehler in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise ProcessingError(
                f"Unerwarteter Verarbeitungsfehler: {e}", 
                original_error=e
            )
    return wrapper

def log_exception(func: Callable) -> Callable:
    """
    Decorator zum Protokollieren von Ausnahmen
    
    Args:
        func: Zu dekorierende Funktion
    
    Returns:
        Dekoriete Funktion mit Fehlerprotokollierung
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fehler in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

def safe_execution(
    func: Callable, 
    *args, 
    fallback: Optional[Callable] = None,
    **kwargs
) -> Any:
    """
    Sichere Funktionsausführung mit optionalem Fallback
    
    Args:
        func: Primäre Funktion
        fallback: Fallback-Funktion
        *args, **kwargs: Funktionsargumente
    
    Returns:
        Funktionsergebnis
    """
    logger = logging.getLogger('SafeExecution')
    
    try:
        return func(*args, **kwargs)
    except Exception as primary_error:
        logger.warning(f"Primäre Funktion fehlgeschlagen: {primary_error}")
        
        if fallback is None:
            raise
        
        try:
            logger.info("Fallback-Strategie wird ausgeführt")
            return fallback(*args, **kwargs)
        except Exception as fallback_error:
            logger.error(f"Fallback fehlgeschlagen: {fallback_error}")
            raise primary_error