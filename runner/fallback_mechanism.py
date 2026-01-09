"""
Fallback-Mechanismen für tile-compile

Implementiert robuste Fehlerbehandlungs- und Ausweichstrategien.
"""

import logging
from typing import Callable, Any, Optional

class FallbackMechanism:
    """
    Zentrale Fallback-Strategie-Implementierung
    """
    def __init__(self, logger_name: str = 'FallbackMechanism'):
        """
        Initialisiert Fallback-Mechanismus
        
        Args:
            logger_name: Name des Loggers
        """
        self.logger = logging.getLogger(logger_name)
    
    def execute_with_fallback(
        self, 
        primary_func: Callable[..., Any],
        fallback_func: Optional[Callable[..., Any]] = None,
        fallback_handler: Optional[Callable[[Exception], Any]] = None,
        *args, 
        **kwargs
    ) -> Any:
        """
        Führt Funktion mit optionalen Fallback-Strategien
        
        Args:
            primary_func: Primäre Verarbeitungsfunktion
            fallback_func: Alternative Verarbeitungsfunktion
            fallback_handler: Fehler-Behandlungsfunktion
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
        
        Returns:
            Verarbeitungsergebnis
        """
        try:
            # Primäre Funktion ausführen
            return primary_func(*args, **kwargs)
        
        except Exception as primary_error:
            self.logger.warning(f"Primäre Funktion fehlgeschlagen: {primary_error}")
            
            # Fallback-Funktion vorhanden
            if fallback_func is not None:
                try:
                    self.logger.info("Fallback-Strategie wird ausgeführt")
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback fehlgeschlagen: {fallback_error}")
            
            # Fehler-Behandlungsfunktion vorhanden
            if fallback_handler is not None:
                try:
                    self.logger.info("Fehler-Behandlungsfunktion wird ausgeführt")
                    return fallback_handler(primary_error)
                except Exception as handler_error:
                    self.logger.error(f"Fehler-Handler fehlgeschlagen: {handler_error}")
            
            # Ursprünglichen Fehler neu werfen
            raise primary_error
    
    def retry_with_backoff(
        self, 
        func: Callable[..., Any], 
        max_attempts: int = 3,
        base_delay: float = 1.0,
        *args, 
        **kwargs
    ) -> Any:
        """
        Funktion mit exponentieller Backoff-Strategie wiederholen
        
        Args:
            func: Zu wiederholende Funktion
            max_attempts: Maximale Anzahl der Versuche
            base_delay: Basis-Verzögerung zwischen Versuchen
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
        
        Returns:
            Funktionsergebnis
        
        Raises:
            Exception, wenn alle Versuche fehlschlagen
        """
        import time
        
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                if attempt == max_attempts:
                    raise
                
                delay = base_delay * (2 ** (attempt - 1))
                self.logger.warning(
                    f"Versuch {attempt} fehlgeschlagen. "
                    f"Warte {delay:.2f} Sekunden. Fehler: {e}"
                )
                time.sleep(delay)
    
    def safe_generator(
        self, 
        generator_func: Callable[..., Any],
        *args, 
        **kwargs
    ) -> Callable[..., Any]:
        """
        Sicherer Generator-Wrapper
        
        Args:
            generator_func: Generator-Funktion
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
        
        Returns:
            Sicherer Generator
        """
        def safe_gen(*gen_args, **gen_kwargs):
            try:
                for item in generator_func(*gen_args, **gen_kwargs):
                    yield item
            except Exception as e:
                self.logger.error(f"Generator-Fehler: {e}")
                # Optional: Fehler-Handling-Strategie
        
        return safe_gen