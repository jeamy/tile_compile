"""
Memory Mapping Utilities für tile-compile

Stellt Mechanismen zur speichereffizienten Datenverarbeitung bereit.
"""

import os
import tempfile
import numpy as np
import logging
from typing import Optional, Tuple, Any
from pathlib import Path

class MemoryMappedArray:
    """
    Speicher-gemappte Array-Verwaltung mit erweiterten Funktionen
    """
    def __init__(
        self, 
        shape: Tuple[int, ...], 
        dtype: np.dtype = np.float32, 
        tempdir: Optional[str] = None
    ):
        """
        Initialisiert memory-mapped Array
        
        Args:
            shape: Form des Arrays
            dtype: Datentyp
            tempdir: Temporäres Verzeichnis
        """
        self.logger = logging.getLogger(__name__)
        self.tempdir = tempdir or tempfile.gettempdir()
        self.shape = shape
        self.dtype = dtype
        
        # Temporäre Datei erstellen
        self.filename = os.path.join(
            self.tempdir, 
            f"memmap_{os.getpid()}_{id(self)}.dat"
        )
        
        try:
            # Memory-mapped Array erstellen
            self.array = np.memmap(
                self.filename, 
                dtype=self.dtype, 
                mode='w+', 
                shape=self.shape
            )
            self.logger.info(f"Memory-mapped array created: {self.filename}")
        except Exception as e:
            self.logger.error(f"Memory mapping failed: {e}")
            raise
    
    def __del__(self):
        """Ressourcen sicher freigeben"""
        try:
            # Array löschen
            del self.array
            
            # Temporäre Datei entfernen
            if os.path.exists(self.filename):
                os.unlink(self.filename)
                self.logger.debug(f"Memory-mapped file deleted: {self.filename}")
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")
    
    def load_chunk(
        self, 
        start: int, 
        end: Optional[int] = None
    ) -> np.ndarray:
        """
        Lädt einen Chunk des Arrays
        
        Args:
            start: Startindex
            end: Endindex
        
        Returns:
            Array-Chunk
        """
        end = end or self.shape[0]
        return self.array[start:end]
    
    def write_chunk(
        self, 
        chunk: np.ndarray, 
        start: int
    ):
        """
        Schreibt einen Chunk in das Array
        
        Args:
            chunk: Zu schreibender Chunk
            start: Startindex
        """
        end = start + chunk.shape[0]
        self.array[start:end] = chunk
    
    @classmethod
    def from_array(
        cls, 
        array: np.ndarray, 
        tempdir: Optional[str] = None
    ) -> 'MemoryMappedArray':
        """
        Erstellt Memory-Mapped Array aus existierendem Array
        
        Args:
            array: Ursprüngliches Array
            tempdir: Temporäres Verzeichnis
        
        Returns:
            MemoryMappedArray
        """
        memmap_array = cls(array.shape, array.dtype, tempdir)
        memmap_array.array[:] = array[:]
        return memmap_array
    
    def as_numpy_array(self) -> np.ndarray:
        """
        Konvertiert Memory-Mapped Array zu Standard-NumPy-Array
        
        Returns:
            NumPy-Array
        """
        return np.array(self.array)

def memory_mapped_processing(
    data: np.ndarray, 
    processing_func: callable,
    chunk_size: int = 1000
) -> np.ndarray:
    """
    Verarbeitung großer Arrays durch Chunk-basiertes Memory Mapping
    
    Args:
        data: Eingabedaten
        processing_func: Verarbeitungsfunktion
        chunk_size: Größe der Verarbeitungs-Chunks
    
    Returns:
        Verarbeitete Daten
    """
    logger = logging.getLogger(__name__)
    
    # Memory-Mapped Array erstellen
    mapped_data = MemoryMappedArray.from_array(data)
    result = MemoryMappedArray(data.shape, data.dtype)
    
    try:
        for start in range(0, data.shape[0], chunk_size):
            end = min(start + chunk_size, data.shape[0])
            
            # Chunk laden und verarbeiten
            chunk = mapped_data.load_chunk(start, end)
            processed_chunk = processing_func(chunk)
            
            # Verarbeiteten Chunk speichern
            result.write_chunk(processed_chunk, start)
        
        return result.as_numpy_array()
    
    except Exception as e:
        logger.error(f"Memory-mapped processing error: {e}")
        raise