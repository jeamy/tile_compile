"""
Zentralisierte Logging-Konfiguration für tile-compile

Stellt umfassende Logging-Funktionalitäten bereit.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(
    log_level=logging.INFO, 
    log_dir='/var/log/tile-compile',
    log_prefix='tile_compile'
):
    """
    Umfassendes Logging-Setup mit konfigurierbaren Parametern
    
    Args:
        log_level: Logging-Level (default: INFO)
        log_dir: Verzeichnis für Logdateien
        log_prefix: Präfix für Logdateien
    """
    # Sicherstellen, dass Log-Verzeichnis existiert
    os.makedirs(log_dir, exist_ok=True)
    
    # Eindeutiger Dateiname mit Zeitstempel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")
    
    # Root Logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Formatter für strukturierte Logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File Handler mit Rotation
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Alte Handler entfernen
    logger.handlers.clear()
    
    # Neue Handler hinzufügen
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Zusätzliche Konfigurationen
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logger

def get_logger(name):
    """
    Erstellt einen konfigurierten Logger
    
    Args:
        name: Name des Loggers
    
    Returns:
        Konfigurierter Logger
    """
    return logging.getLogger(name)

# Logging beim Import konfigurieren
default_logger = setup_logging()