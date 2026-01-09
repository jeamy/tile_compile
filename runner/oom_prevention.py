"""
Zentralisierte OOM-Prävention für tile-compile

Umfassende Speichermanagement- und Verarbeitungsstrategie
"""

import logging
import time
import traceback
import psutil
import numpy as np
from typing import (
    List, Dict, Any, Optional, 
    Callable, Generator, Union, Tuple
)
from pathlib import Path

class ResourceManager:
    """
    Zentralisierte Ressourcenmanagement-Klasse mit erweiterten Funktionen
    """
    def __init__(
        self, 
        memory_threshold: float = 80.0,
        cpu_threshold: float = 90.0,
        disk_threshold: float = 90.0
    ):
        """
        Initialisiert Ressourcenmanager mit konfigurierbaren Schwellwerten
        
        Args:
            memory_threshold: Speicher-Schwellwert in Prozent
            cpu_threshold: CPU-Auslastungs-Schwellwert
            disk_threshold: Festplatten-Schwellwert
        """
        self.logger = logging.getLogger(__name__)
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.disk_threshold = disk_threshold
        
        # Zusätzliche Tracking-Mechanismen
        self.initial_memory = psutil.virtual_memory().total
        self.peak_memory_usage = 0
    
    def check_resources(self) -> bool:
        """
        Überprüft Systemressourcen mit detaillierter Analyse
        
        Returns:
            bool: Ressourcen ausreichend
        """
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # Speicher-Tracking
        current_memory_usage = memory.percent
        self.peak_memory_usage = max(self.peak_memory_usage, current_memory_usage)
        
        checks = [
            current_memory_usage <= self.memory_threshold,
            cpu <= self.cpu_threshold,
            disk.percent <= self.disk_threshold
        ]
        
        if not all(checks):
            self.logger.warning(
                f"Ressourcenlimits überschritten: "
                f"Speicher={current_memory_usage}%, "
                f"CPU={cpu}%, "
                f"Disk={disk.percent}%"
            )
            return False
        
        return True
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Liefert detaillierten Ressourcenstatus
        
        Returns:
            Dict mit Ressourcendetails
        """
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            'memory_total_gb': self.initial_memory / (1024**3),
            'memory_used_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'peak_memory_usage_percent': self.peak_memory_usage,
            'cpu_usage_percent': cpu,
            'disk_usage_percent': disk.percent
        }

class ChunkedProcessor:
    """
    Chunk-basierte Datenverarbeitung mit adaptiver Strategie
    """
    def __init__(
        self, 
        resource_manager: ResourceManager,
        initial_chunk_size: int = 50,
        min_chunk_size: int = 10,
        max_chunk_size: int = 200
    ):
        """
        Initialisiert Chunk-Processor
        
        Args:
            resource_manager: Ressourcenmanager
            initial_chunk_size: Initiale Chunk-Größe
            min_chunk_size: Minimale Chunk-Größe
            max_chunk_size: Maximale Chunk-Größe
        """
        self.logger = logging.getLogger(__name__)
        self.resource_manager = resource_manager
        self.initial_chunk_size = initial_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def process_in_chunks(
        self, 
        data: List[Any], 
        processing_func: Callable[[List[Any]], List[Any]]
    ) -> List[Any]:
        """
        Verarbeitung in Chunks mit dynamischer Anpassung
        
        Args:
            data: Zu verarbeitende Daten
            processing_func: Verarbeitungsfunktion
        
        Returns:
            Verarbeitete Daten
        """
        results = []
        current_chunk_size = min(self.initial_chunk_size, len(data))
        
        for start in range(0, len(data), current_chunk_size):
            # Ressourcen-Check
            if not self.resource_manager.check_resources():
                self.logger.warning("Ressourcen erschöpft. Breche Verarbeitung ab.")
                break
            
            end = min(start + current_chunk_size, len(data))
            chunk = data[start:end]
            
            try:
                processed_chunk = processing_func(chunk)
                results.extend(processed_chunk)
                
                # Dynamische Chunk-Größenanpassung
                current_chunk_size = self._adjust_chunk_size(
                    current_chunk_size, 
                    len(processed_chunk)
                )
            
            except Exception as e:
                self.logger.error(f"Chunk-Verarbeitung fehlgeschlagen: {e}")
        
        return results
    
    def _adjust_chunk_size(
        self, 
        current_size: int, 
        processed_size: int
    ) -> int:
        """
        Passt Chunk-Größe dynamisch an
        
        Args:
            current_size: Aktuelle Chunk-Größe
            processed_size: Größe des verarbeiteten Chunks
        
        Returns:
            Angepasste Chunk-Größe
        """
        # Heuristik zur Chunk-Größenanpassung
        if processed_size > current_size * 1.5:
            # Chunk-Größe reduzieren
            return max(self.min_chunk_size, current_size // 2)
        elif processed_size < current_size // 2:
            # Chunk-Größe erhöhen
            return min(self.max_chunk_size, current_size * 2)
        
        return current_size

def prevent_oom(
    memory_limit_mb: int = 8192,
    log_level: int = logging.WARNING
) -> Callable:
    """
    Decorator zur OOM-Prävention
    
    Args:
        memory_limit_mb: Speicherlimit in MB
        log_level: Logging-Level
    
    Returns:
        Decorator-Funktion
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.setLevel(log_level)
            
            resource_manager = ResourceManager(
                memory_threshold=memory_limit_mb / psutil.virtual_memory().total * 100
            )
            
            try:
                if not resource_manager.check_resources():
                    logger.warning("Nicht genügend Systemressourcen")
                    return None
                
                result = func(*args, **kwargs)
                
                # Ressourcennutzung protokollieren
                status = resource_manager.get_resource_status()
                logger.info(f"Ressourcennutzung nach {func.__name__}: {status}")
                
                return result
            
            except MemoryError as e:
                logger.error(f"Speicherfehler: {e}")
                logger.error(traceback.format_exc())
                return None
            except Exception as e:
                logger.error(f"Unerwarteter Fehler: {e}")
                logger.error(traceback.format_exc())
                return None
        
        return wrapper
    
    return decorator

# Kompatibilitäts-Wrapper für bestehende Funktionen
def safe_processing(
    processing_func: Callable,
    data: Any,
    memory_limit_mb: int = 8192
) -> Optional[Any]:
    """
    Sichere Verarbeitung mit OOM-Schutz
    
    Args:
        processing_func: Zu verarbeitende Funktion
        data: Eingabedaten
        memory_limit_mb: Speicherlimit
    
    Returns:
        Verarbeitetes Ergebnis oder None
    """
    @prevent_oom(memory_limit_mb)
    def safe_func(input_data):
        return processing_func(input_data)
    
    return safe_func(data)