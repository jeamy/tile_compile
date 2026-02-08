import os
import requests
import numpy as np
from astropy.io import fits
from astroquery.sdss import SDSS
from astroquery.mast import Observations
import logging

class ReferenceDatasetManager:
    def __init__(self, output_dir='reference_datasets'):
        """
        Initialize Reference Dataset Manager
        
        Args:
            output_dir: Directory to store reference datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def download_sdss_images(
        self, 
        ra: float, 
        dec: float, 
        radius: float = 0.1,
        filters: list = ['r', 'g', 'i']
    ):
        """
        Download SDSS images around a specific celestial coordinate
        
        Args:
            ra: Right Ascension
            dec: Declination
            radius: Search radius in degrees
            filters: Color filters to download
        """
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        coord = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        
        try:
            # Search for objects
            result_table = SDSS.query_region(
                coord, 
                radius=radius*u.deg, 
                photoobj_fields=['ra', 'dec', 'modelMag_r', 'modelMag_g', 'modelMag_i']
            )
            
            if result_table is None or len(result_table) == 0:
                self.logger.warning(f"No SDSS objects found near coordinates (RA: {ra}, Dec: {dec})")
                return []
            
            # Download images
            downloaded_images = []
            for obj in result_table:
                try:
                    # Download each filter
                    images = SDSS.get_images(
                        ra=obj['ra'], 
                        dec=obj['dec'], 
                        filters=filters
                    )
                    
                    for image in images:
                        # Save image
                        filename = f"sdss_{obj['ra']}_{obj['dec']}_{image[0].header['FILTER']}.fits"
                        filepath = os.path.join(self.output_dir, filename)
                        image[0].writeto(filepath, overwrite=True)
                        downloaded_images.append(filepath)
                        
                        self.logger.info(f"Downloaded: {filename}")
                
                except Exception as img_err:
                    self.logger.error(f"Error downloading image: {img_err}")
            
            return downloaded_images
        
        except Exception as e:
            self.logger.error(f"SDSS download error: {e}")
            return []
    
    def download_hubble_images(
        self, 
        target: str, 
        instrument: str = 'WFC3'
    ):
        """
        Download Hubble Space Telescope images
        
        Args:
            target: Astronomical target name
            instrument: Specific HST instrument
        """
        try:
            # Search Hubble observations
            obs_table = Observations.query_object(
                target, 
                instruments=instrument
            )
            
            if len(obs_table) == 0:
                self.logger.warning(f"No Hubble observations found for {target}")
                return []
            
            # Download suitable images
            downloaded_images = []
            for obs in obs_table:
                # Download data products
                data_products = Observations.get_product_list(obs)
                
                # Filter for fits images
                fits_products = data_products[
                    (data_products['productType'] == 'SCIENCE') & 
                    (data_products['productSubType'] == 'IMAGE')
                ]
                
                for product in fits_products:
                    try:
                        # Download and save
                        local_path = Observations.download_file(product['dataURI'])
                        filename = os.path.basename(local_path)
                        new_path = os.path.join(self.output_dir, filename)
                        os.rename(local_path, new_path)
                        
                        downloaded_images.append(new_path)
                        self.logger.info(f"Downloaded Hubble image: {filename}")
                    
                    except Exception as download_err:
                        self.logger.error(f"Download error: {download_err}")
            
            return downloaded_images
        
        except Exception as e:
            self.logger.error(f"Hubble data retrieval error: {e}")
            return []
    
    def preprocess_fits_images(self, fits_paths):
        """
        Preprocess FITS images for validation
        
        Args:
            fits_paths: List of FITS image paths
        
        Returns:
            Processed image data
        """
        processed_images = []
        
        for path in fits_paths:
            try:
                with fits.open(path) as hdul:
                    # Extract primary image data
                    image_data = hdul[0].data
                    
                    # Basic preprocessing
                    if image_data is None:
                        self.logger.warning(f"No image data in {path}")
                        continue
                    
                    # Normalize image
                    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
                    
                    processed_images.append(image_data)
                    self.logger.info(f"Processed: {path}")
            
            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}")
        
        return processed_images

def main():
    # Example usage
    manager = ReferenceDatasetManager()
    
    # Download SDSS images near M31 (Andromeda Galaxy)
    sdss_images = manager.download_sdss_images(
        ra=10.6847,   # Andromeda Galaxy coordinates
        dec=41.2692,
        radius=0.5    # Search radius
    )
    
    # Download Hubble images of Orion Nebula
    hubble_images = manager.download_hubble_images('M42')
    
    # Preprocess images
    processed_sdss = manager.preprocess_fits_images(sdss_images)
    processed_hubble = manager.preprocess_fits_images(hubble_images)

if __name__ == '__main__':
    main()